"""Parallel exponential-weighted cumulative scan (Mamba/S4-style) with a custom
autograd function."""

import einops
import torch

from .linalg import complex, inverse


__all__ = ["ConvScanFn", "conv_scan", "DenseLinearScanFn", "dense_linear_scan"]


class ConvScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            A: torch.Tensor,        # float: [... x L]
            B: torch.Tensor,        # float: [... x (L + 1)]
            chunk_size: int,        # int: C (accepted for API compatibility; unused)
    ) -> torch.Tensor:              # float: [... x (L + 1)]
        with torch.no_grad():
            exp_A = torch.exp(A)                # float: [... x L]  (per-step gain)
            out = _conv_scan_fwd_gain(exp_A, B)
        # Cache the gains (not A): the backward needs exp(A) twice and never needs A.
        ctx.save_for_backward(exp_A, out)
        return out

    @staticmethod
    def backward(
            ctx,
            dout: torch.Tensor,     # float: [... x (L + 1)]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        exp_A, out = ctx.saved_tensors
        with torch.no_grad():
            dA, dB = _conv_scan_bwd(dout, {"exp_A": exp_A, "out": out,})
        return dA, dB, None


def _conv_scan_fwd_gain(
        exp_A: torch.Tensor,    # [... x L]        (multiplicative gain per step)
        B: torch.Tensor,        # [... x (L + 1)]
) -> torch.Tensor:              # [... x (L + 1)]
    """Doubling cumulative scan ``out[k] = exp_A[k-1] * out[k-1] + B[k]`` (``out[0] = B[0]``).

    Operates directly in the *gain* domain (``exp_A`` are the per-step multipliers),
    so a caller that already holds gains -- eigenvalues of a transition matrix, a
    matrix-power ratio -- need not round-trip through ``log`` then ``exp``.
    """
    *bsz, L = exp_A.shape
    A_, out = torch.ones_like(B), B.clone()     # [... x (L + 1)]
    A_[..., 1:] = exp_A

    k = 1
    while k <= L:
        out[..., k:].addcmul_(A_[..., k:], out[..., :-k].clone())
        A_[..., k:] *= A_[..., :-k].clone()
        k <<= 1
    return out


def _conv_scan_fwd(
        A: torch.Tensor,    # float: [... x L]
        B: torch.Tensor,    # float: [... x (L + 1)]
) -> torch.Tensor:          # float: [... x (L + 1)]
    """Log-domain wrapper: ``A`` holds log-gains, so the per-step gain is ``exp(A)``."""
    return _conv_scan_fwd_gain(torch.exp(A), B)


def _conv_scan_bwd(
        dout: torch.Tensor,     # float: [... x (L + 1)]
        cache: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:     # float: [... x L], [... x (L + 1)]
    exp_A = cache["exp_A"]                                                  # float: [... x L]
    dB = _conv_scan_fwd_gain(exp_A.flip(dims=(-1,)), dout).flip(dims=(-1,)) # float: [... x (L + 1)]
    out = cache["out"]                      # float: [... x (L + 1)]

    dA = einops.einsum(dB[..., 1:], out[..., :-1], exp_A, "..., ..., ... -> ...")   # float: [... x L]
    return dA, dB


def conv_scan(
        A: torch.Tensor,    # float: [B x L x H x D]
        B: torch.Tensor,    # float: [B x L x H x D]
        chunk_size: int,    # int: C (accepted for API compatibility; unused)
) -> torch.Tensor:          # float: [B x L x H x D]
    return ConvScanFn.apply(A, B, chunk_size)


def _sum_to(t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Reduce ``t`` back onto ``shape`` by summing the dimensions that were
    broadcast (the inverse of ``torch.broadcast_to``), as autograd would."""
    while t.ndim > len(shape):
        t = t.sum(dim=0)
    for i, s in enumerate(shape):
        if s == 1 and t.shape[i] != 1:
            t = t.sum(dim=i, keepdim=True)
    return t


class DenseLinearScanFn(torch.autograd.Function):
    """Linear recurrence with a *dense* transition matrix, ``s[t] = M @ s[t-1] + b[t]``.

    The recurrence is diagonalized once (``M = V diag(D) Vinv``) and evaluated as a
    complex diagonal scan via :func:`_conv_scan_fwd`, so the cost is the same
    ``O(... x n x L log L)`` doubling as :class:`ConvScanFn` plus two ``[n x n]``
    projections. The backward is the analytic adjoint recurrence
    ``lambda[t] = M^T lambda[t+1] + grad_s[t]`` run with ``(D, V, Vinv)`` held
    constant (same eigenvalues, basis ``Vinv^T``); the unstable gradient of
    ``torch.linalg.eig`` is therefore never invoked, and the gradient w.r.t. ``M``
    is the well-defined ``sum_t lambda[t] s[t-1]^T``.
    """

    @staticmethod
    def forward(
            ctx,
            M: torch.Tensor,    # float: [Bm... x n x n]
            B: torch.Tensor,    # float: [Bb... x (L + 1) x n]
    ) -> torch.Tensor:          # float: [B... x L x n]
        with torch.no_grad():
            L = B.shape[-2] - 1
            D, V = torch.linalg.eig(M)                                          # complex: [Bm... x n], [Bm... x n x n]
            Vinv = inverse(V)                                                   # complex: [Bm... x n x n]

            beta = (complex(B) @ Vinv.mT).mT                                    # complex: [B... x n x (L + 1)]
            gains = D[..., :, None].expand(*beta.shape[:-1], L)                 # complex: [B... x n x L]  (eigenvalue per step)
            z = _conv_scan_fwd_gain(gains, beta)                                # complex: [B... x n x (L + 1)]
            states = torch.real((V @ z).mT)                                     # float:   [B... x (L + 1) x n]

        ctx.save_for_backward(M, D, V, Vinv, states)
        ctx.M_shape, ctx.B_shape = M.shape, B.shape
        return states[..., 1:, :]

    @staticmethod
    def backward(
            ctx,
            grad_states: torch.Tensor,      # float: [B... x L x n]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        M, D, V, Vinv, states = ctx.saved_tensors
        with torch.no_grad():
            # Adjoint recurrence lambda[t] = M^T lambda[t+1] + grad_s[t], solved in the
            # modal basis of M^T (= Vinv^T diag(D) V^T) by a reverse diagonal scan.
            gamma = (complex(grad_states) @ V).mT                               # complex: [B... x n x L]  (V^T grad_s)
            gains = D[..., :, None].expand(*gamma.shape)                        # complex: [B... x n x L]  (eigenvalue per step)
            zeros = torch.zeros_like(gamma[..., :1])
            nu = _conv_scan_fwd_gain(gains, torch.cat([zeros, gamma.flip(dims=(-1,))], dim=-1))  # complex: [B... x n x (L + 1)]
            mu = nu[..., 1:].flip(dims=(-1,))                                   # complex: [B... x n x L]  (V^T lambda)
            lam = Vinv.mT @ mu                                                  # complex: [B... x n x L]  (lambda columns)

            grad_M = torch.real(lam @ complex(states[..., :-1, :]))            # float: [B... x n x n]

            lam_rows = lam.mT                                                   # complex: [B... x L x n]  (lambda[1..L] rows)
            grad_s0 = torch.real(lam_rows[..., :1, :] @ complex(M))[..., 0, :]  # float: [B... x n]  (M^T lambda[1])
            grad_B = torch.cat([grad_s0[..., None, :], torch.real(lam_rows)], dim=-2)  # float: [B... x (L + 1) x n]

            # Reduce over any dims that were broadcast from the declared input shapes.
            grad_M = _sum_to(grad_M, ctx.M_shape)
            grad_B = _sum_to(grad_B, ctx.B_shape)

        return grad_M, grad_B


def dense_linear_scan(
        M: torch.Tensor,    # float: [... x n x n]
        B: torch.Tensor,    # float: [... x (L + 1) x n]
) -> torch.Tensor:          # float: [... x L x n]
    """Solve ``s[t] = M @ s[t-1] + b[t]`` for ``t = 1..L`` and return ``s[1..L]``.

    Following :func:`conv_scan`'s leading-element convention, ``B[..., 0, :]`` holds
    the initial state ``s[0]`` and ``B[..., t, :]`` holds the per-step input ``b[t]``.
    """
    return DenseLinearScanFn.apply(M, B)
