"""Smoke-test micro-benchmark: K-recursion vs alternatives (honest, CPU, float64).

Two comparisons:
  (A) Forward covariance time: K-recursion `k_full` vs the explicit closed form
      `(I-A)^{-1} Σ (I-A)^{-H}` (full_covariance_closed_form, uses torch.linalg.inv),
      on a sparse chain and a dense DAG, sweeping the node count M.
  (B) Gradient time: reverse-mode AD (one backward sweep -> all q gradients) vs
      central finite differences (2q forward evaluations).

All numbers are wall-clock medians from actual runs. Interpretation is printed
honestly: the closed form is a single optimized LAPACK call, while k_full is a
pure-Python block loop, so for small problems Python overhead can dominate; the
AD-vs-FD gap is the robust, implementation-independent result.

Run: uv run python bench/benchmark_krecursion.py
"""

from __future__ import annotations

import statistics
import time

import torch

import gaussian_bn as gbn

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


def gen(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def chain_model(M, d, seed=0):
    g = gen(seed)
    edges = {(i, i + 1): 0.5 * torch.randn(d, d, generator=g) for i in range(M - 1)}
    noise = [torch.eye(d) for _ in range(M)]
    return gbn.GaussianDAG([d] * M, edges, noise)


def dense_model(M, d, seed=0):
    g = gen(seed)
    s = 0.4 / max(M, 1)
    edges = {(i, j): s * torch.randn(d, d, generator=g) for j in range(M) for i in range(j)}
    noise = [torch.eye(d) for _ in range(M)]
    return gbn.GaussianDAG([d] * M, edges, noise)


def timeit(fn, reps=15, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def forward_benchmark(d=2):
    print(f"\n(A) Forward covariance time  [node dim d={d}]   (median ms)")
    print(f"  {'topology':8s} {'M':>4s} {'#edges':>7s} {'k_full':>10s} {'inverse':>10s} {'ratio inv/krec':>15s}")
    for name, builder in [("chain", chain_model), ("dense", dense_model)]:
        for M in [8, 16, 32, 64, 128]:
            m = builder(M, d)
            E = len(m.edges)
            # correctness check (once)
            assert torch.linalg.norm(gbn.k_full(m) - gbn.full_covariance_closed_form(m)) < 1e-8
            t_k = timeit(lambda: gbn.k_full(m)) * 1e3
            t_i = timeit(lambda: gbn.full_covariance_closed_form(m)) * 1e3
            print(f"  {name:8s} {M:4d} {E:7d} {t_k:10.3f} {t_i:10.3f} {t_i / t_k:15.2f}")


def gradient_benchmark(M=24, d=2):
    print(f"\n(B) Gradient time: AD (one backward) vs finite differences (2q forward)  [chain M={M}, d={d}]")
    m0 = chain_model(M, d, seed=1)
    eta0, pk = gbn.pack(m0, noise_param="fixed")     # edge params only
    q = eta0.numel()

    def f(e):
        return (gbn.k_full(gbn.unpack(e, pk, m0)) ** 2).sum()

    def ad_grad():
        e = eta0.detach().clone().requires_grad_(True)
        val = f(e)
        val.backward()
        return e.grad

    def fd_grad():
        e0 = eta0.detach()
        g = torch.zeros_like(e0)
        eps = 1e-6
        for a in range(q):
            de = torch.zeros_like(e0)
            de[a] = eps
            g[a] = (f(e0 + de) - f(e0 - de)) / (2 * eps)
        return g

    # correctness: AD vs FD agree
    rel = float(torch.linalg.norm(ad_grad() - fd_grad()) / torch.linalg.norm(fd_grad()))
    t_ad = timeit(ad_grad, reps=10, warmup=2) * 1e3
    t_fd = timeit(fd_grad, reps=3, warmup=1) * 1e3
    print(f"  #parameters q            = {q}")
    print(f"  AD (1 fwd + 1 bwd)       = {t_ad:.3f} ms   -> all {q} gradients")
    print(f"  finite diff (2q forward) = {t_fd:.3f} ms")
    print(f"  speedup  FD/AD           = {t_fd / t_ad:.1f}x")
    print(f"  AD vs FD rel. error      = {rel:.2e}")

    # how AD cost scales with q (vary M) vs FD
    print(f"\n  scaling of gradient cost with q (chain, d={d}):")
    print(f"  {'M':>4s} {'q':>5s} {'AD ms':>9s} {'FD ms':>9s} {'FD/AD':>8s}")
    for Mi in [8, 16, 32, 64]:
        mi = chain_model(Mi, d, seed=2)
        eta_i, pk_i = gbn.pack(mi, noise_param="fixed")
        qi = eta_i.numel()

        def fi(e, pk_i=pk_i, mi=mi):
            return (gbn.k_full(gbn.unpack(e, pk_i, mi)) ** 2).sum()

        def adi(eta_i=eta_i, fi=fi):
            e = eta_i.detach().clone().requires_grad_(True)
            fi(e).backward()
            return e.grad

        def fdi(eta_i=eta_i, fi=fi, qi=qi):
            e0 = eta_i.detach()
            g = torch.zeros_like(e0)
            for a in range(qi):
                de = torch.zeros_like(e0); de[a] = 1e-6
                g[a] = (fi(e0 + de) - fi(e0 - de)) / 2e-6
            return g

        tad = timeit(adi, reps=8, warmup=2) * 1e3
        tfd = timeit(fdi, reps=2, warmup=1) * 1e3
        print(f"  {Mi:4d} {qi:5d} {tad:9.3f} {tfd:9.3f} {tfd / tad:8.1f}")


def main():
    print("=" * 72)
    print("K-recursion micro-benchmark (CPU, float64, single process)")
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}")
    print("=" * 72)
    forward_benchmark(d=2)
    gradient_benchmark(M=24, d=2)
    print("\nNotes:")
    print(" - Forward: k_full is a pure-Python block loop; the closed form is one")
    print("   LAPACK inv call. For sparse/large M the recursion does fewer FLOPs")
    print("   (chain O(M^2 d^3) vs inverse O(M^3 d^3)); for small M Python overhead")
    print("   can dominate. Both are exact (checked to 1e-8).")
    print(" - Gradient: AD returns ALL q gradients in one backward sweep; finite")
    print("   differences cost 2q forward passes. This gap is the robust result and")
    print("   is what 'no per-topology gradient derivation' buys at run time.")


if __name__ == "__main__":
    main()
