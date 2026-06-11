"""Linear-algebra and small numerical helpers (torch)."""

from __future__ import annotations

import math
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


def pow_series(M: torch.Tensor, n: int) -> torch.Tensor:
    N = M.shape[0]
    I = torch.eye(N, device=M.device)
    if n == 1:
        return I[None]
    else:
        k = int(math.ceil(math.log2(n)))
        bits = [M]
        for _ in range(k - 1):
            bits.append(bits[-1] @ bits[-1])

        result = I
        for bit in bits:
            augmented_bit = torch.cat([I, bit], dim=1)
            blocked_result = result @ augmented_bit
            result = torch.cat([blocked_result[:, :N], blocked_result[:, N:]], dim=0)
        return result.reshape(1 << k, N, N)[:n]


def batch_trace(x: torch.Tensor) -> torch.Tensor:
    return x.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


def kl_div(cov1: torch.Tensor, cov2: torch.Tensor) -> torch.Tensor:
    return ((torch.det(cov2) / torch.det(cov1)).log() - cov1.shape[-1] + (torch.inverse(cov2) * cov1).sum(dim=(-2, -1))) / 2


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
