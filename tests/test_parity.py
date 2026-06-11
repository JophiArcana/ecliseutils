"""Parity / correctness tests for the shared numerics.

These validate ``ecliseutils`` implementations against independent references
(scipy, naive loops) rather than against the original project copies, so they
double as a regression guard for the consolidated package.
"""

import importlib.util
import os

import numpy as np
import pytest
import torch

import ecliseutils as eu

eu.configure(device="cpu", dtype=torch.float64, precision=10, seed=0)

# Optional path to the original KF_RNN infrastructure, used for true
# "parity-vs-current-implementation" checks on the verbatim-copied numerics.
_KF_INFRA = os.path.expanduser(
    "~/Desktop/College/KF_RNN/src/kf_rnn/infrastructure"
)


def _load_standalone(path, name):
    """Import a single-file module by path (only for files with no intra-package imports)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rand_spd(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(n, n, generator=g, dtype=torch.float64)
    return M @ M.mT + n * torch.eye(n, dtype=torch.float64)


def test_sqrtm_roundtrip():
    M = _rand_spd(5, seed=1)
    S = eu.sqrtm(M)
    assert torch.allclose(S @ S, M, atol=1e-8)
    assert torch.allclose(S, S.mT, atol=1e-10)


def test_solve_discrete_are_matches_scipy():
    import scipy.linalg

    g = torch.Generator().manual_seed(2)
    n, m = 4, 2
    A = 0.5 * torch.randn(n, n, generator=g, dtype=torch.float64)
    B = torch.randn(n, m, generator=g, dtype=torch.float64)
    Q = _rand_spd(n, seed=3)
    R = _rand_spd(m, seed=4)

    P = eu.solve_discrete_are(A, B, Q, R)
    # Riccati residual should be ~0.
    res = eu.test_discrete_are(A, B, Q, R, P)
    assert res.abs().max().item() < 1e-6

    P_scipy = torch.tensor(scipy.linalg.solve_discrete_are(
        A.numpy(), B.numpy(), Q.numpy(), R.numpy()
    ))
    assert torch.allclose(P, P_scipy, atol=1e-5)


def test_solve_continuous_are_matches_scipy():
    import scipy.linalg

    g = torch.Generator().manual_seed(5)
    n, m = 4, 2
    A = torch.randn(n, n, generator=g, dtype=torch.float64)
    B = torch.randn(n, m, generator=g, dtype=torch.float64)
    Q = _rand_spd(n, seed=6)
    R = _rand_spd(m, seed=7)

    P = eu.solve_continuous_are(A, B, Q, R)
    res = eu.test_continuous_are(A, B, Q, R, P)
    assert res.abs().max().item() < 1e-6

    P_scipy = torch.tensor(scipy.linalg.solve_continuous_are(
        A.numpy(), B.numpy(), Q.numpy(), R.numpy()
    ))
    assert torch.allclose(P, P_scipy, atol=1e-5)


def test_solve_are_batched_matches_scipy():
    import scipy.linalg

    g = torch.Generator().manual_seed(11)
    K, n, m = 5, 4, 2
    A = 0.5 * torch.randn(K, n, n, generator=g, dtype=torch.float64)
    B = torch.randn(K, n, m, generator=g, dtype=torch.float64)
    Q = torch.stack([_rand_spd(n, 100 + i) for i in range(K)])
    R = torch.stack([_rand_spd(m, 200 + i) for i in range(K)])

    for solver, scipy_fn, test_fn in (
        (eu.solve_discrete_are, scipy.linalg.solve_discrete_are, eu.test_discrete_are),
        (eu.solve_continuous_are, scipy.linalg.solve_continuous_are, eu.test_continuous_are),
    ):
        P = solver(A, B, Q, R)
        assert P.shape == (K, n, n)
        res = test_fn(A, B, Q, R, P).abs().amax(dim=(-2, -1))
        assert res.max().item() < 1e-6
        P_scipy = torch.stack([torch.tensor(scipy_fn(
            A[i].numpy(), B[i].numpy(), Q[i].numpy(), R[i].numpy())) for i in range(K)])
        assert torch.allclose(P, P_scipy, atol=1e-5)


def test_solve_discrete_are_handles_singular_R():
    """DARE with ``R = 0`` (deadbeat) / rank-deficient ``R`` is well-posed and solved
    via the disk-function pencil path; scipy / an ``R^{-1}`` formulation cannot run."""
    g = torch.Generator().manual_seed(21)
    n, m = 4, 2
    A = 0.5 * torch.randn(n, n, generator=g, dtype=torch.float64)
    B = torch.randn(n, m, generator=g, dtype=torch.float64)
    Q = _rand_spd(n, 22)

    for R in (
        torch.zeros(m, m, dtype=torch.float64),
        torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64),
    ):
        P = eu.solve_discrete_are(A, B, Q, R)
        res = eu.test_discrete_are(A, B, Q, R, P).abs().max().item()
        assert res < 1e-6, res


def test_solve_are_handles_defective_pencil():
    """A nilpotent Jordan-block ``A`` makes the eigenvector basis rank-deficient, so an
    ``eig`` subspace extraction fails; the sign/disk function does not."""
    A = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
    B = torch.tensor([[1.], [0.], [0.]], dtype=torch.float64)
    Q = torch.eye(3, dtype=torch.float64)
    R = torch.eye(1, dtype=torch.float64)

    # The eigenvector matrix of A is genuinely rank-deficient (defective).
    _, V = torch.linalg.eig(torch.complex(A, torch.zeros_like(A)))
    assert torch.linalg.matrix_rank(V).item() < 3

    Pc = eu.solve_continuous_are(A, B, Q, R)
    assert eu.test_continuous_are(A, B, Q, R, Pc).abs().max().item() < 1e-6
    Pd = eu.solve_discrete_are(A, B, Q, R)
    assert eu.test_discrete_are(A, B, Q, R, Pd).abs().max().item() < 1e-6


def test_solve_are_gradcheck():
    """Implicit-differentiation backward matches numerical gradients (float64)."""
    g = torch.Generator().manual_seed(31)
    n, m = 3, 2
    A = 0.3 * torch.randn(n, n, generator=g, dtype=torch.float64)
    B = 0.5 * torch.randn(n, m, generator=g, dtype=torch.float64)
    Q = _rand_spd(n, 40)
    R = _rand_spd(m, 41)

    def inputs():
        return tuple(t.clone().requires_grad_(True) for t in (A, B, Q, R))

    assert torch.autograd.gradcheck(eu.solve_discrete_are, inputs(), atol=1e-5, rtol=1e-4)
    assert torch.autograd.gradcheck(
        lambda *x: eu.solve_continuous_are(*x, precision=None), inputs(), atol=1e-5, rtol=1e-4)


def test_conv_scan_matches_naive_recurrence():
    g = torch.Generator().manual_seed(8)
    shape = (3, 7)  # [... x L]
    L = shape[-1]
    A = 0.1 * torch.randn(*shape, generator=g, dtype=torch.float64)
    B = torch.randn(*shape[:-1], L + 1, generator=g, dtype=torch.float64)

    out = eu.conv_scan(A, B, chunk_size=4)

    # Naive recurrence: y_0 = B[0]; y_k = exp(A[k-1]) y_{k-1} + B[k].
    naive = B.clone()
    for k in range(1, L + 1):
        naive[..., k] = torch.exp(A[..., k - 1]) * naive[..., k - 1] + B[..., k]
    assert torch.allclose(out, naive, atol=1e-9)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(_KF_INFRA, "fast_conv_scan.py")),
    reason="original KF_RNN fast_conv_scan.py not available for parity check",
)
def test_conv_scan_parity_with_original_kf_rnn():
    """Forward AND backward must match the original KF_RNN implementation verbatim.

    (The original custom backward is intentionally an efficient, non-exact
    gradient; this refactor preserves it byte-for-byte rather than changing it.)
    """
    orig = _load_standalone(os.path.join(_KF_INFRA, "fast_conv_scan.py"), "_orig_fcs")

    g = torch.Generator().manual_seed(9)
    L = 6
    A0 = 0.1 * torch.randn(L, generator=g, dtype=torch.float64)
    B0 = torch.randn(L + 1, generator=g, dtype=torch.float64)

    def run(conv_scan_fn):
        A = A0.clone().requires_grad_(True)
        B = B0.clone().requires_grad_(True)
        out = conv_scan_fn(A, B, 4)
        (out ** 2).sum().backward()
        return out.detach(), A.grad.detach(), B.grad.detach()

    out_eu, dA_eu, dB_eu = run(eu.conv_scan)
    out_orig, dA_orig, dB_orig = run(orig.conv_scan)
    assert torch.allclose(out_eu, out_orig, atol=1e-12)
    assert torch.allclose(dA_eu, dA_orig, atol=1e-12)
    assert torch.allclose(dB_eu, dB_orig, atol=1e-12)


def test_hadamard_conjugation_matches_loop():
    g = torch.Generator().manual_seed(10)
    m, n, p, q = 2, 3, 2, 3

    def crandn(*shape):
        return torch.complex(
            torch.randn(*shape, generator=g, dtype=torch.float64),
            torch.randn(*shape, generator=g, dtype=torch.float64),
        )

    A = crandn(m, n)
    B = crandn(p, q)
    alpha = 0.3 * crandn(m, n)
    beta = 0.3 * crandn(p, q)
    C = crandn(m, p)

    result = eu.hadamard_conjugation(A, B, alpha, beta, C)

    ref = torch.zeros(n, q, dtype=torch.complex128)
    for i in range(n):
        for j in range(q):
            acc = 0.0
            for mm in range(m):
                for pp in range(p):
                    coeff = 1.0 / (1.0 - alpha[mm, i] * beta[pp, j])
                    acc = acc + A[mm, i] * B[pp, j] * C[mm, pp] * coeff
            ref[i, j] = acc
    assert torch.allclose(result.to(torch.complex128), ref, atol=1e-9)


def test_labeled_array_take_and_broadcast():
    from collections import OrderedDict

    vals = np.arange(6).reshape(2, 3)
    la = eu.LabeledArray(vals, ("a", "b"))
    taken = la.take({"a": 1})
    assert taken.dims == ("b",)
    assert np.array_equal(taken.values, vals[1])

    bc = la.broadcast(OrderedDict([("a", 2), ("b", 3), ("c", 4)]))
    assert bc.dims == ("a", "b", "c")
    assert bc.shape == (2, 3, 4)


def test_multi_map_and_zip():
    arr = np.empty(3, dtype=object)
    for i in range(3):
        arr[i] = i + 1
    mapped = eu.multi_map(lambda x: x * 10, arr, dtype=object)
    assert [mapped[i] for i in range(3)] == [10, 20, 30]

    zipped = eu.multi_zip(arr, arr)
    assert zipped[2] == (3, 3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
