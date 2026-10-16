
import argparse
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per process: fair timing
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

SPAMBASE_URL = ("https://media.githubusercontent.com/media/EpistasisLab/pmlb/"
                "master/datasets/spambase/spambase.tsv.gz")
H, BOUND = 10, 10.0  # hidden neurons, weight range [-BOUND, BOUND]


# ----------------------------------------------------------------- MLP + fitness
class Problem:
    """n-H-1 sigmoid MLP; fitness = training MSE. Evaluates a whole population at once."""

    def __init__(self, X, y):
        self.X, self.y = X.astype(np.float32), y.astype(np.float32)
        self.n = X.shape[1]
        self.D = self.n * H + 2 * H + 1
        self.fe, self.max_fe = 0, np.inf

    def proba(self, W, X=None):
        X = self.X if X is None else X
        W = np.atleast_2d(W).astype(np.float32)
        n = self.n
        W1 = W[:, :n * H].reshape(-1, n, H)
        b1 = W[:, n * H:n * H + H]
        W2 = W[:, n * H + H:n * H + 2 * H]
        b2 = W[:, -1]
        h = 1 / (1 + np.exp(-np.clip(np.einsum("nf,pfh->pnh", X, W1, optimize=True) + b1[:, None], -50, 50)))
        o = np.einsum("pnh,ph->pn", h, W2) + b2[:, None]
        return 1 / (1 + np.exp(-np.clip(o, -50, 50)))

    def __call__(self, W):
        """Budgeted evaluation: requests beyond max_fe return +inf (not evaluated)."""
        W = np.atleast_2d(W)
        f = np.full(len(W), np.inf)
        k = int(min(len(W), self.max_fe - self.fe))
        if k > 0:
            f[:k] = np.mean((self.proba(W[:k]) - self.y) ** 2, axis=1)
            self.fe += k
        return f

    @property
    def done(self):
        return self.fe >= self.max_fe


def clip(x):
    return np.clip(x, -BOUND, BOUND)


# ------------------------------------------------------------------- optimizers
def kma(prob, rng, pop_size=15, d=0.5, sparse=None, alpha=0.1, a=5, pmin=10, pmax=200,
        state=None, until=None):
    """Komodo Mlipir Algorithm (Suyanto et al., 2022).
    sparse=None -> original KMA; sparse=rho -> random steps move only a fraction rho of weights.
    state/until let PKMA run an island in epochs."""
    D, span = prob.D, 2 * BOUND
    until = prob.max_fe if until is None else until

    def rand_step(m):
        s = (2 * rng.random((m, D)) - 1) * alpha * span
        return s * (rng.random((m, D)) < sparse) if sparse else s

    if state is None:
        pop = rng.uniform(-BOUND, BOUND, (pop_size, D)); fit = prob(pop)
        o = np.argsort(fit); state = (pop[o], fit[o], [fit.min()])
    pop, fit, hist = state
    while prob.fe < until and not prob.done:
        q = max(2, min(int(0.5 * (len(pop) - 1)), 3, len(pop) - 3))
        big, bf, fem, ff, small = pop[:q], fit[:q], pop[q].copy(), fit[q], pop[q + 1:]
        # big males: attract to better, repel from worse with prob 0.5 (HILE)
        nb = np.empty_like(big)
        for i in range(q):
            sh = np.zeros(D)
            for j in range(q):
                if i != j:
                    att = bf[j] < bf[i] or rng.random() < 0.5
                    sh += rng.random(D) * ((big[j] - big[i]) if att else (big[i] - big[j]))
            nb[i] = clip(big[i] + sh)
        allb, allf = np.vstack([big, nb]), np.concatenate([bf, prob(nb)])
        k = np.argsort(allf)[:q]; big, bf = allb[k], allf[k]
        # female: mating with the winner or parthenogenesis
        if rng.random() < 0.5:
            r = rng.random(D)
            off = clip(np.vstack([r * big[0] + (1 - r) * fem, r * fem + (1 - r) * big[0]]))
        else:
            off = clip(fem + rand_step(1))
        of = prob(off); b = np.argmin(of)
        if of[b] < ff:
            fem, ff = off[b], of[b]
        # small males: mlipir (follow big males on a random subset of dimensions)
        sh = np.zeros_like(small)
        for bm in big:
            sh += (rng.random(small.shape) < d) * rng.random(small.shape) * (bm - small)
        small = clip(small + sh); sf = prob(small)
        pop = np.vstack([big, fem[None], small]); fit = np.concatenate([bf, [ff], sf])
        # self-adaptive population size
        hist = (hist + [min(hist[-1], fit.min())])[-3:]
        if len(hist) == 3:
            same1, same2 = np.isclose(hist[2], hist[1]), np.isclose(hist[1], hist[0])
            if not (same1 or same2) and len(pop) - a >= pmin:
                k = np.argsort(fit)[:-a]; pop, fit = pop[k], fit[k]
            elif same1 and same2 and len(pop) + a <= pmax:
                new = clip(pop[np.argmin(fit)] + rand_step(a))
                pop, fit = np.vstack([pop, new]), np.concatenate([fit, prob(new)])
        o = np.argsort(fit); pop, fit = pop[o], fit[o]
    return pop[0], fit[0], (pop, fit, hist)


def ga(prob, rng, n=50, pc=0.9, pm=0.02, elites=2):
    pop = rng.uniform(-BOUND, BOUND, (n, prob.D)); fit = prob(pop)
    while not prob.done:
        o = np.argsort(fit); m = n - elites
        c = rng.integers(0, n, (2 * m, 3))
        par = pop[c[np.arange(2 * m), np.argmin(fit[c], 1)]]          # tournament
        p1, p2 = par[:m], par[m:]
        lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
        ch = rng.uniform(lo - 0.5 * (hi - lo), hi + 0.5 * (hi - lo))  # BLX-0.5
        nox = rng.random(m) >= pc; ch[nox] = p1[nox]                  # no crossover
        ch = clip(ch + (rng.random(ch.shape) < pm) * rng.normal(0, 0.1 * 2 * BOUND, ch.shape))
        pop = np.vstack([pop[o[:elites]], ch]); fit = np.concatenate([fit[o[:elites]], prob(ch)])
    i = np.argmin(fit); return pop[i], fit[i]


def pso(prob, rng, n=50, w=0.7298, c=1.49618):
    vmax = 0.2 * 2 * BOUND
    x = rng.uniform(-BOUND, BOUND, (n, prob.D)); v = rng.uniform(-vmax, vmax, x.shape)
    f = prob(x); pb, pf = x.copy(), f.copy(); g = pb[np.argmin(pf)]
    while not prob.done:
        v = np.clip(w * v + c * rng.random(x.shape) * (pb - x) + c * rng.random(x.shape) * (g - x),
                    -vmax, vmax)
        x = clip(x + v); f = prob(x)
        imp = f < pf; pb[imp], pf[imp] = x[imp], f[imp]; g = pb[np.argmin(pf)]
    i = np.argmin(pf); return pb[i], pf[i]


def sa(prob, rng, frac=0.1):
    x = rng.uniform(-BOUND, BOUND, prob.D); fx = prob(x)[0]
    best, bf, T0, steps, k = x, fx, 0.1 * fx, prob.max_fe - 1, 0
    while not prob.done:
        tau = k / steps; T = T0 * 1e-3 ** tau; sig = 0.1 * 0.1 ** tau
        y = clip(x + (rng.random(prob.D) < frac) * rng.normal(0, sig * 2 * BOUND, prob.D))
        fy = prob(y)[0]
        if fy <= fx or rng.random() < np.exp(-(fy - fx) / max(T, 1e-12)):
            x, fx = y, fy
            if fx < bf:
                best, bf = x, fx
        k += 1
    return best, bf


# --------------------------------------------------- PKMA: parallel island KMA
_P = None


def _init(X, y):
    global _P
    _P = Problem(X, y)


def _epoch(args):
    """Run one island until it has used `until` evaluations (executes in a worker)."""
    isl, until = args
    _P.fe, _P.max_fe = isl["fe"], isl["budget"]
    x, f, isl["state"] = kma(_P, isl["rng"], d=isl["d"], sparse=0.05,
                             state=isl["state"], until=until)
    isl["fe"] = _P.fe
    return isl


def pkma(prob, seed, K=2, epochs=50, workers=2):
    """K islands (mlipir rates spread over [0.3, 0.7]), sparse parthenogenesis,
    synchronous ring migration after every epoch. Same result for any `workers`."""
    budget = int(prob.max_fe) // K
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(K)]
    isl = [{"rng": rngs[k], "d": float(np.linspace(0.3, 0.7, K)[k]), "state": None,
            "fe": 0, "budget": budget} for k in range(K)]
    with Pool(workers, initializer=_init, initargs=(prob.X, prob.y)) as pool:
        for e in range(epochs):
            until = round(budget * (e + 1) / epochs)
            isl = pool.map(_epoch, [(i, until) for i in isl])
            if e < epochs - 1:  # ring migration: best of k replaces worst of k+1
                mig = [(i["state"][0][0].copy(), i["state"][1][0]) for i in isl]
                for k, i in enumerate(isl):
                    x, f = mig[(k - 1) % K]
                    pop, fit, h = i["state"]
                    if f < fit[-1]:
                        pop, fit = pop.copy(), fit.copy(); pop[-1], fit[-1] = x, f
                        o = np.argsort(fit); i["state"] = (pop[o], fit[o], h)
    prob.fe = sum(i["fe"] for i in isl)
    b = min(isl, key=lambda i: i["state"][1][0])
    return b["state"][0][0], b["state"][1][0]


# ------------------------------------------------------------------------ main
def load(args):
    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        df = pd.read_csv(SPAMBASE_URL, sep="\t", compression="gzip")
    return df.iloc[:, :-1].values.astype(float), df.iloc[:, -1].values.astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--fe", type=int, default=20000); ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    X, y = load(args)
    print(f"data: {X.shape[0]} samples, {X.shape[1]} features, spam={y.mean():.1%}; "
          f"budget={args.fe} FE, runs={args.runs}\n")

    algos = {"PKMA": lambda p, s: pkma(p, s, workers=args.workers),
             "KMA": lambda p, s: kma(p, np.random.default_rng(s))[:2],
             "GA": lambda p, s: ga(p, np.random.default_rng(s)),
             "PSO": lambda p, s: pso(p, np.random.default_rng(s)),
             "SA": lambda p, s: sa(p, np.random.default_rng(s))}
    rows = []
    for r in range(args.runs):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=r)
        sc = MinMaxScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), np.clip(sc.transform(Xte), 0, 1).astype(np.float32)
        for name, fn in algos.items():
            prob = Problem(Xtr, ytr); prob.max_fe = args.fe
            t = time.perf_counter(); w, mse = fn(prob, 1000 * r + len(name))
            pred = (prob.proba(w, Xte)[0] >= 0.5).astype(int)
            rows.append(dict(run=r, alg=name, mse=mse, acc=accuracy_score(yte, pred),
                             f1=f1_score(yte, pred), time=time.perf_counter() - t, fe=prob.fe))
            print(f"run {r} {name:5s} mse={mse:.4f} acc={rows[-1]['acc']:.4f} "
                  f"time={rows[-1]['time']:.1f}s fe={prob.fe}")

    df = pd.DataFrame(rows)
    print("\nmean (std) over runs")
    summ = df.groupby("alg")[["mse", "acc", "f1", "time"]].agg(["mean", "std"])
    print(summ.loc[list(algos)].round(4).to_string())
    if args.runs >= 5:
        print("\nWilcoxon signed-rank, PKMA vs others (test accuracy)")
        piv = df.pivot(index="run", columns="alg", values="acc")
        for b in ["KMA", "GA", "PSO", "SA"]:
            p = 1.0 if np.allclose(piv.PKMA, piv[b]) else wilcoxon(piv.PKMA, piv[b]).pvalue
            print(f"  vs {b:4s} mean diff={100*(piv.PKMA-piv[b]).mean():+.2f} pp  p={p:.3f}")
    df.to_csv("benchmark_results.csv", index=False)
    print("\nper-run results saved to benchmark_results.csv")


if __name__ == "__main__":
    main()
