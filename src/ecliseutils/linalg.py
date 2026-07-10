"""Linear-algebra and small numerical helpers (torch)."""

from __future__ import annotations

from typing import Union

import einops
import torch
from tensordict import TensorDict

from .modules import multi_vmap


__all__ = [
    "pow_series",
    "batch_trace",
    "kl_div",
    "sqrtm",
    "complex",
    "ceildiv",
    "ceil",
    "T",
    "hadamard_conjugation",
    "hadamard_conjugation_diff_order1",
    "hadamard_conjugation_diff_order2",
    "inverse",
    "eig_some",
]


def _pow_series_eig(D: torch.Tensor, V: torch.Tensor, Vinv: torch.Tensor, n: int, complex_M: bool) -> torch.Tensor:
    """``[M^0, ..., M^{n-1}]`` from the eigendecomposition ``M = V diag(D) Vinv``.

    Uses ``M^k = V diag(exp(log(D) k)) Vinv`` (parallel over ``k``). The ``k = 0``
    column is set explicitly so a zero eigenvalue (``log(D) = -inf``) does not
    produce ``-inf * 0 = nan``; for ``k >= 1`` a zero eigenvalue correctly gives 0.
    """
    k = torch.arange(n, device=D.device)
    Dk = torch.exp(torch.log(D)[..., :, None] * k)          # complex: [... x N x n]  (D_j^k)
    Dk[..., 0] = 1
    powers = torch.einsum("...ij,...jt,...jl->...til", V, Dk, Vinv)     # [... x n x N x N]
    return powers if complex_M else powers.real


def _pow_series_squaring(M: torch.Tensor, n: int) -> torch.Tensor:
    """``[M^0, ..., M^{n-1}]`` by truncated repeated-squaring doubling.

    Needs no eigenbasis, so it is used both as the general fallback for defective /
    non-diagonalizable ``M`` and to keep the all-matrices robustness of the original
    ``pow_series``. Computes exactly ``n`` powers (no pad-to-power-of-two waste).
    """
    N = M.shape[-1]
    I = torch.eye(N, dtype=M.dtype, device=M.device).expand(*M.shape[:-2], 1, N, N)
    P = I                                       # [... x 1 x N x N]  (M^0)
    gain = M                                    # M^(2^i)
    while P.shape[-3] < n:
        take = min(P.shape[-3], n - P.shape[-3])
        P = torch.cat([P, P[..., :take, :, :] @ gain[..., None, :, :]], dim=-3)
        if P.shape[-3] < n:
            gain = gain @ gain
    return P


def _pow_series_grad_modal(G: torch.Tensor, P: torch.Tensor, D: torch.Tensor, V: torch.Tensor, Vinv: torch.Tensor) -> torch.Tensor:
    """``grad_M`` via the adjoint recurrence solved with the shared modal scan.

    ``bar_P_k = G_k + M^T bar_P_{k+1}`` is a linear recurrence with dense transition
    ``M^T = V^{-T} diag(D) V^T``; it is run (reversed) through
    :func:`fast_conv_scan._conv_scan_fwd_gain` reusing the forward's ``(D, V, Vinv)``,
    treating the matrix state's column index as a batch dim. Then
    ``grad_M = sum_{k>=1} bar_P_k P_{k-1}^T``.
    """
    from . import fast_conv_scan

    n = P.shape[-3]
    L = n - 1
    # [B... x k x i x j] -> [B... x j(cols, batched) x k(time) x i(state)], time reversed.
    b = complex(einops.rearrange(G, "... k i j -> ... j k i")).flip(dims=(-2,))

    Vb, Vinvb = V[..., None, :, :], Vinv[..., None, :, :]        # broadcast eig factors over the column batch
    beta = (b @ Vb).mT                                          # complex: [B... x j x N_mode x n]  (project onto M^T basis)
    gains = D[..., None, :, None].expand(*beta.shape[:-1], L)   # complex: [B... x j x N_mode x L]
    z = fast_conv_scan._conv_scan_fwd_gain(gains, beta)         # complex: [B... x j x N_mode x n]
    u = (Vinvb.mT @ z).mT                                       # complex: [B... x j x k x i]

    barP = einops.rearrange(u.flip(dims=(-2,)), "... j k i -> ... k i j")   # [B... x n x N x N]
    if not torch.is_complex(P):
        barP = barP.real

    # grad_M = sum_{k=1}^{n-1} bar_P_k @ P_{k-1}^T
    return torch.einsum("...kir,...kjr->...ij", barP[..., 1:, :, :], P[..., :-1, :, :])


def _pow_series_grad_loop(G: torch.Tensor, P: torch.Tensor, M: torch.Tensor, n: int) -> torch.Tensor:
    """``grad_M`` via a plain reverse adjoint loop (defective / fallback path)."""
    MT = M.mT
    grad_M = torch.zeros_like(M)
    bar_next = torch.zeros_like(M)              # bar_P_n = 0
    for k in range(n - 1, 0, -1):
        bar_k = G[..., k, :, :] + MT @ bar_next
        grad_M = grad_M + bar_k @ P[..., k - 1, :, :].mT
        bar_next = bar_k
    return grad_M


class _PowSeriesFn(torch.autograd.Function):
    """Series of matrix powers ``[M^0, M^1, ..., M^{n-1}]`` with an explicit,
    eig-gradient-free backward.

    The forward diagonalizes ``M = V diag(D) Vinv`` once and evaluates the powers
    from ``D^k = exp(log(D) k)`` (parallel over ``k``). Defective / non-diagonalizable
    ``M`` (an ill-conditioned eigenvector matrix ``V``) falls back to repeated
    squaring, which needs no eigenbasis.

    The backward is the exact adjoint of ``P_k = M P_{k-1}``:
    ``bar_P_k = G_k + M^T bar_P_{k+1}`` (reverse) then ``grad_M = sum_{k>=1} bar_P_k P_{k-1}^T``.
    The eigendecomposition is held fixed, so ``torch.linalg.eig``'s (unstable)
    gradient is never invoked -- the same trick :class:`DenseLinearScanFn` uses.
    """

    @staticmethod
    def forward(ctx, M: torch.Tensor, n: int) -> torch.Tensor:
        with torch.no_grad():
            D, V = torch.linalg.eig(M)                          # complex: [... x N], [... x N x N]
            Vinv = inverse(V)                                   # complex: [... x N x N]

            eps = torch.finfo(V.real.dtype).eps
            cond = torch.linalg.cond(V)
            used_eig = bool(torch.all(torch.isfinite(cond) & (cond < eps ** -0.5)).item())

            if used_eig:
                P = _pow_series_eig(D, V, Vinv, n, torch.is_complex(M))
                ctx.save_for_backward(M, P, D, V, Vinv)
            else:
                P = _pow_series_squaring(M, n)
                ctx.save_for_backward(M, P)

        ctx.used_eig = used_eig
        ctx.n = n
        return P

    @staticmethod
    def backward(ctx, G: torch.Tensor) -> "tuple[torch.Tensor, None]":
        with torch.no_grad():
            if ctx.used_eig:
                M, P, D, V, Vinv = ctx.saved_tensors
                grad_M = _pow_series_grad_modal(G, P, D, V, Vinv)
            else:
                M, P = ctx.saved_tensors
                grad_M = _pow_series_grad_loop(G, P, M, ctx.n)
        return grad_M, None


def pow_series(M: torch.Tensor, n: int) -> torch.Tensor:
    """Stack of matrix powers ``[M^0, M^1, ..., M^{n-1}]`` -> ``[... x n x N x N]``.

    Differentiable in ``M`` via an explicit adjoint backward (no gradient flows
    through the eigensolver). Non-diagonalizable ``M`` is handled by a
    repeated-squaring fallback.
    """
    return _PowSeriesFn.apply(M, n)


def batch_trace(x: torch.Tensor) -> torch.Tensor:
    return x.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


def kl_div(cov1: torch.Tensor, cov2: torch.Tensor) -> torch.Tensor:
    """KL divergence ``KL(N(0, cov1) || N(0, cov2))`` for SPD covariances (batched).

    Cholesky-based: the two SPD factorizations are about half the flops of the LU
    used by ``det`` / ``inverse``, the log-det ratio is a stable sum of log-diagonals
    (no overflow-prone determinant product), and the trace term reuses one factor via
    a triangular solve.
    """
    L1 = torch.linalg.cholesky(cov1)
    L2 = torch.linalg.cholesky(cov2)
    log_det_ratio = 2 * (
        torch.log(torch.diagonal(L2, dim1=-2, dim2=-1)).sum(-1)
        - torch.log(torch.diagonal(L1, dim1=-2, dim2=-1)).sum(-1)
    )
    trace_term = batch_trace(torch.cholesky_solve(cov1, L2))    # tr(cov2^{-1} cov1)
    return (log_det_ratio - cov1.shape[-1] + trace_term) / 2


def sqrtm(t: torch.Tensor) -> torch.Tensor:
    """Symmetric matrix square root for symmetric positive-semidefinite ``t``.

    Uses the Hermitian eigensolver (orthonormal eigenvectors, so no explicit inverse
    and no complex arithmetic), which is substantially more stable than the general
    ``eig`` + ``inverse`` form. Eigenvalues are clamped at 0 to absorb small negative
    numerical drift.
    """
    L, V = torch.linalg.eigh(t)
    return V @ torch.diag_embed(L.clamp_min(0.0).sqrt()) @ V.mT


def complex(t: "torch.Tensor | TensorDict") -> Union[torch.Tensor, TensorDict]:
    fn = lambda t_: t_ if torch.is_complex(t_) else torch.complex(t_, torch.zeros_like(t_))
    return fn(t) if isinstance(t, torch.Tensor) else t.apply(fn)


def ceildiv(a: int, b: int) -> int:
    return -(-a // b)


def ceil(a: int) -> int:
    return ceildiv(a, 1)


def T(t: torch.Tensor) -> torch.Tensor:
    return t.permute((*range(t.ndim - 1, -1, -1),))


def hadamard_conjugation(
        A: torch.Tensor,        # [B... x m x n]
        B: torch.Tensor,        # [B... x p x q]
        alpha: torch.Tensor,    # [B... x m x n]
        beta: torch.Tensor,     # [B... x p x q]
        C: torch.Tensor         # [B... x m x p]
) -> torch.Tensor:              # [B... x n x q]
    coeff = 1 / (1 - alpha[..., :, None, :, None] * beta[..., None, :, None, :])            # [B... x m x p x n x q]
    return torch.einsum("...mn, ...pq, ...mp, ...mpnq -> ...nq", A, B, complex(C), coeff,)  # [B... x n x q]


def hadamard_conjugation_diff_order1(
        A: torch.Tensor,        # [B... x m x n]
        B: torch.Tensor,        # [B... x p x q]
        alpha: torch.Tensor,    # [B... x m x n]
        beta1: torch.Tensor,    # [B... x p x q]
        beta2: torch.Tensor,    # [B... x p x q]
        C: torch.Tensor         # [B... x m x p]
) -> torch.Tensor:              # [B... x n x q]
    alpha_ = alpha[..., :, None, :, None]                                                   # [B... x m x 1 x n x 1]
    _beta1, _beta2 = beta1[..., None, :, None, :], beta2[..., None, :, None, :]             # [B... x 1 x p x 1 x q]
    coeff = alpha_ / ((1 - alpha_ * _beta1) * (1 - alpha_ * _beta2))                        # [B... x m x p x n x q]
    return torch.einsum("...mn, ...pq, ...mp, ...mpnq -> ...nq", A, B, complex(C), coeff,)  # [B... x n x q]


def hadamard_conjugation_diff_order2(
        B: torch.Tensor,        # [B... x p x q]
        beta1: torch.Tensor,    # [B... x p x q]
        beta2: torch.Tensor,    # [B... x p x q]
        C: torch.Tensor         # [B... x p x p]
) -> torch.Tensor:              # [B... x q x q]
    beta1_, _beta1 = beta1[..., :, None, :, None], beta1[..., None, :, None, :]             # b1_ik, b1_jl
    beta2_, _beta2 = beta2[..., :, None, :, None], beta2[..., None, :, None, :]             # b2_ik, b2_jl

    beta12 = beta1_ * _beta2                                                                # b1_ik * b2_jl
    beta21 = einops.rearrange(beta12, "... i j k l -> ... j i l k")                         # b2_ik * b1_jl
    beta11, beta22 = (beta1_ * _beta1), (beta2_ * _beta2),                                  # b1_ik * b1_jl, b2_ik * b2_jl,

    coeff = 1 - beta12 * beta21
    for t in (beta11, beta12, beta21, beta22,):
        coeff.div_(1 - t)
    return torch.einsum("...mn, ...pq, ...mp, ...mpnq -> ...nq", B, B, C, coeff,)


def inverse(A: torch.Tensor) -> torch.Tensor:
    """Matrix inverse with an SVD-based pseudo-inverse fallback for singular ``A``."""
    try:
        return torch.inverse(A)
    except torch._C._LinAlgError:
        U, S, VT = torch.linalg.svd(A)
        VSinv: torch.Tensor = VT.mT / S[..., None, :]
        VSinv.nan_to_num_(nan=0.0, posinf=torch.inf, neginf=-torch.inf)
        VSinvUT = VSinv @ U.mT
        VSinvUT.nan_to_num_(nan=0.0, posinf=torch.inf, neginf=-torch.inf)
        return VSinvUT


def eig_some(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eigendecomposition keeping only the nonzero-eigenvalue subspace."""
    bsz = A.shape[:-2]
    L, V = torch.linalg.eig(A)
    Vinv = torch.inverse(V)
    n_nonzero = torch.sum(L != 0, dim=-1)
    max_n_nonzero = torch.max(n_nonzero).item()
    indices = torch.topk(torch.abs(L), k=max_n_nonzero, dim=-1).indices
    L = torch.gather(L, -1, indices)

    vmap_gather = multi_vmap(lambda t, idx: t[:, idx], n=len(bsz))
    V = vmap_gather(V, indices)
    Vinv = vmap_gather(Vinv.mT, indices).mT
    return V, L, Vinv
