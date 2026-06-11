"""Algebraic Riccati equation solvers (batched, PyTorch, differentiable).

A single in-house implementation for both the discrete (DARE) and continuous
(CARE) algebraic Riccati equations that is:

* **batched / parallel** -- everything is batched torch, no per-element scipy
  loop;
* **robust to degenerate ``R`` (e.g. ``R = 0``) for the DARE** -- the forward
  builds the van Dooren *extended pencil*, which never forms ``R^{-1}``, then
  works on the deflated pencil ``(Lhat, Mhat)`` via the disk-function matrix
  ``(Lhat + Mhat)^{-1}(Lhat - Mhat)``. Inverting neither ``Lhat`` nor ``Mhat``
  means a deadbeat / ``R = 0`` spectrum (closed-loop eigenvalues at ``mu = 0``
  with symplectic partners at ``mu = inf``) is handled cleanly. (The CARE with
  singular ``R`` is genuinely ill-posed -- ``B R^{-1} B^T`` diverges and the
  closed-loop eigenvalues escape to ``inf`` *on* the imaginary-axis splitting
  boundary -- so CARE requires a nonsingular ``R``.)
* **robust to defective / non-diagonalizable matrices** -- the stable subspace
  is extracted with the matrix **sign** function, a spectral projector that
  never computes eigenvectors, so a rank-deficient or Jordan-block eigenvector
  basis (which breaks ``torch.linalg.eig``) cannot break it;
* **differentiable** -- the (non-differentiable) forward is wrapped in a custom
  :class:`torch.autograd.Function` whose backward solves a Lyapunov/Sylvester
  adjoint via implicit differentiation. Gradient stability is therefore
  decoupled from the forward method.

Why not an in-house Schur/QZ? No CUDA library (cuSOLVER, MAGMA, ``torch.linalg``)
exposes a batched nonsymmetric real-Schur / QZ with invariant-subspace
reordering, and hand-rolling Francis double-shift iterations is branchy,
sequential and GPU-hostile. We only need the stable invariant/deflating
subspace, which the sign/disk function computes with batched ``matmul`` /
``solve`` / ``svd`` (all well-supported on CUDA, unlike ``eig``).

Caveats:

* **Solvability assumption.** The deflated pencil ``(Lhat, Mhat)`` must have no
  eigenvalue exactly on the splitting boundary (imaginary axis for CARE, unit
  circle for DARE). This is the standard existence condition for a stabilizing
  solution; the sign/disk function is well-defined precisely under it.
  Off-boundary defectiveness / Jordan blocks are fine. (Singular factors --
  eigenvalues at ``0`` or ``inf`` -- are fine for the DARE disk form but not for
  the CARE sign form, hence the CARE nonsingular-``R`` requirement above.)
* **Backward w.r.t. ``R`` when ``R`` is exactly singular** is ill-defined (the
  gradient involves ``R^{-1}``); the forward ``P`` is still produced and a
  pseudo-inverse convention is used.

``test_discrete_are`` / ``test_continuous_are`` return the Riccati residual and
are handy for accuracy checks (especially in the degenerate cases scipy cannot
even run).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


__all__ = [
    "solve_discrete_are",
    "solve_continuous_are",
    "test_discrete_are",
    "test_continuous_are",
]


# SECTION: small batched linear-algebra helpers
def _sym(X: torch.Tensor) -> torch.Tensor:
    return 0.5 * (X + X.mT)


def _safe_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """``A^{-1} B`` with a pseudo-inverse fallback when ``A`` is (numerically) singular."""
    try:
        return torch.linalg.solve(A, B)
    except torch._C._LinAlgError:
        return torch.linalg.pinv(A) @ B


def _bkron(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Batched Kronecker product matching ``torch.kron`` on the trailing 2 dims.

    ``A`` is ``[B... x p x q]`` and ``B`` is ``[B... x r x s]`` (batch dims
    broadcast, and either operand may be unbatched). Returns ``[B... x pr x qs]``.
    """
    p, q = A.shape[-2:]
    r, s = B.shape[-2:]
    out = torch.einsum("...ij,...kl->...ikjl", A, B)
    return out.reshape(*out.shape[:-4], p * r, q * s)


def _commutation(n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Commutation matrix ``K`` with ``K vec(X) = vec(X^T)`` (column-major vec)."""
    idx = torch.arange(n * n, device=device)
    perm = idx // n + n * (idx % n)
    return torch.eye(n * n, dtype=dtype, device=device)[perm]


def _vecc_row(X: torch.Tensor) -> torch.Tensor:
    """Column-major ``vec(X)`` returned as a row vector ``[B... x 1 x (a*b)]``."""
    a, b = X.shape[-2:]
    return X.mT.reshape(*X.shape[:-2], 1, a * b)


def _unvec(v: torch.Tensor, a: int, b: int) -> torch.Tensor:
    """Inverse of a row-vector ``[B... x 1 x (a*b)]`` via ``.view(a, b)`` (no transpose)."""
    v = v.squeeze(-2)
    return v.reshape(*v.shape[:-1], a, b)


def _unvecT(v: torch.Tensor, a: int, b: int) -> torch.Tensor:
    """Inverse via ``.view(a, b).T`` (i.e. column-major un-vec) -> ``[B... x b x a]``."""
    return _unvec(v, a, b).mT


# SECTION: forward -- deflated extended pencil + matrix sign/disk projector
def _extended_pencil(
        A: torch.Tensor,
        B: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        discrete: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Van Dooren extended pencil ``L - lambda M`` (size ``2m + n``); no ``R^{-1}``."""
    bsz = A.shape[:-2]
    m, n = B.shape[-2:]

    I_m = torch.eye(m, dtype=A.dtype, device=A.device).expand(*bsz, m, m)
    Zmm = A.new_zeros((*bsz, m, m))
    Zmn = A.new_zeros((*bsz, m, n))
    Znm = A.new_zeros((*bsz, n, m))
    Znn = A.new_zeros((*bsz, n, n))

    if discrete:
        L = torch.cat([
            torch.cat([A, Zmm, B], dim=-1),
            torch.cat([-Q, I_m, Zmn], dim=-1),
            torch.cat([Znm, Znm, R], dim=-1),
        ], dim=-2)
        M = torch.cat([
            torch.cat([I_m, Zmm, Zmn], dim=-1),
            torch.cat([Zmm, A.mT, Zmn], dim=-1),
            torch.cat([Znm, -B.mT, Znn], dim=-1),
        ], dim=-2)
    else:
        L = torch.cat([
            torch.cat([A, Zmm, B], dim=-1),
            torch.cat([-Q, -A.mT, Zmn], dim=-1),
            torch.cat([Znm, B.mT, R], dim=-1),
        ], dim=-2)
        M = torch.cat([
            torch.cat([I_m, Zmm, Zmn], dim=-1),
            torch.cat([Zmm, I_m, Zmn], dim=-1),
            torch.cat([Znm, Znm, Znn], dim=-1),
        ], dim=-2)
    return L, M


def _deflate(L: torch.Tensor, M: torch.Tensor, m: int, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress the trailing ``n`` columns -> regular ``2m x 2m`` pencil ``(Lhat, Mhat)``.

    The QR of ``L``'s trailing control columns and projection onto their
    orthogonal complement eliminates the control variable without ever forming
    ``R^{-1}`` -- this is what makes singular / zero ``R`` admissible.
    """
    q_of_qr = torch.linalg.qr(L[..., :, -n:], mode="complete")[0]    # [B... x (2m+n) x (2m+n)]
    defl = q_of_qr[..., :, n:]                                       # [B... x (2m+n) x 2m]
    Lhat = defl.mT @ L[..., :, :2 * m]                               # [B... x 2m x 2m]
    Mhat = defl.mT @ M[..., :, :2 * m]                               # [B... x 2m x 2m]
    return Lhat, Mhat


def _matrix_sign(C: torch.Tensor, max_iter: int = 64, tol: float = 1e-12) -> torch.Tensor:
    """Matrix sign function via scaled Newton iteration ``X <- 1/2 (cX + (cX)^{-1})``."""
    sz = C.shape[-1]
    eps = torch.finfo(C.dtype).eps
    X = C
    for _ in range(max_iter):
        Xinv = torch.linalg.inv(X)
        det = torch.linalg.det(X).abs().clamp_min(eps)
        c = det.pow(-1.0 / sz)[..., None, None]
        Xnew = 0.5 * (c * X + Xinv / c)
        denom = X.norm(dim=(-2, -1)).clamp_min(eps)
        diff = (Xnew - X).norm(dim=(-2, -1)) / denom
        X = Xnew
        if torch.all(diff < tol):
            break
    return X


def _stable_subspace_P(Lhat: torch.Tensor, Mhat: torch.Tensor, discrete: bool) -> torch.Tensor:
    """Stable spectral projector of the pencil ``(Lhat, Mhat)`` -> ``P = U21 U11^{-1}``.

    A single matrix sign function is used in both cases; the only difference is the
    matrix it acts on, chosen so that the **stable** eigenvalues map to ``Re < 0``:

    * CARE (stable ``Re mu < 0``): ``Chat = Mhat^{-1} Lhat`` (eigenvalues ``mu``).
    * DARE (stable ``|mu| < 1``): the disk-function matrix
      ``Chat = (Lhat + Mhat)^{-1} (Lhat - Mhat)`` (eigenvalues ``(mu - 1)/(mu + 1)``).
      This never inverts ``Mhat`` (nor ``Lhat``), so deadbeat / ``R = 0`` spectra
      with eigenvalues at ``mu = 0`` and ``mu = inf`` are handled gracefully.
    """
    sz = Lhat.shape[-1]
    m = sz // 2
    I = torch.eye(sz, dtype=Lhat.dtype, device=Lhat.device)

    if discrete:
        Chat = _safe_solve(Lhat + Mhat, Lhat - Mhat)
    else:
        Chat = _safe_solve(Mhat, Lhat)

    S = _matrix_sign(Chat)
    Pm = 0.5 * (I - S)                                  # projector onto the stable subspace
    U = torch.linalg.svd(Pm)[0][..., :, :m]             # leading m left singular vectors
    U11 = U[..., :m, :]
    U21 = U[..., m:, :]
    P = _safe_solve(U11.mT, U21.mT).mT
    return _sym(P)


def _are_forward(A: torch.Tensor, B: torch.Tensor, Q: torch.Tensor, R: torch.Tensor, discrete: bool) -> torch.Tensor:
    m, n = B.shape[-2:]
    L, M = _extended_pencil(A, B, Q, R, discrete)
    Lhat, Mhat = _deflate(L, M, m, n)
    return _stable_subspace_P(Lhat, Mhat, discrete)


# SECTION: backward -- implicit differentiation (shared adjoint machinery)
def _solve_continuous_lyapunov(A_cl: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Solve ``A_cl W + W A_cl^T = C`` (batched, via Kronecker linear system)."""
    m = A_cl.shape[-1]
    I = torch.eye(m, dtype=A_cl.dtype, device=A_cl.device)
    Lk = _bkron(I, A_cl) + _bkron(A_cl, I)                          # [B... x m^2 x m^2]
    rhs = C.mT.reshape(*C.shape[:-2], m * m, 1)
    x = torch.linalg.solve(Lk, rhs)
    return x.reshape(*C.shape[:-2], m, m).mT


def _care_backward(
        P: torch.Tensor, A: torch.Tensor, B: torch.Tensor, Q: torch.Tensor, R: torch.Tensor,
        grad_output: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implicit-diff gradients of CARE. ``A_cl = A - B R^{-1} B^T P``; adjoint is a
    continuous Lyapunov solve ``A_cl W + W A_cl^T = sym(grad_output)``."""
    Rinv = torch.linalg.pinv(R)
    K = Rinv @ B.mT @ P                                            # [B... x n x m]
    A_cl = A - B @ K                                               # [B... x m x m]

    W = _solve_continuous_lyapunov(A_cl, _sym(grad_output))
    PW = P @ W

    dA = -2.0 * PW
    dB = 2.0 * PW @ K.mT
    dQ = -_sym(W)
    dR = -_sym(K @ W @ K.mT)
    return dA, dB, dQ, dR


def _dare_backward(
        P: torch.Tensor, A: torch.Tensor, B: torch.Tensor, Q: torch.Tensor, R: torch.Tensor,
        grad_output: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implicit-diff gradients of DARE.

    Batched port of the exact closed-form expressions in ``are_temp.txt``
    (``Riccati.backward``); ``m`` is the state dim and ``n`` the control dim.
    """
    m, n = B.shape[-2:]
    dtype, device = P.dtype, P.device

    go = _vecc_row(grad_output)                                    # [B... x 1 x m^2]

    M2 = torch.linalg.inv(R + B.mT @ P @ B)                        # [B... x n x n]
    PB = P @ B                                                     # [B... x m x n]
    PBM2 = PB @ M2                                                 # [B... x m x n]
    PBM2BT = PBM2 @ B.mT                                           # [B... x m x m]
    M1 = P - PBM2BT @ P                                            # [B... x m x m]

    I_m = torch.eye(m, dtype=dtype, device=device)
    I_n = torch.eye(n, dtype=dtype, device=device)
    I_m2 = torch.eye(m * m, dtype=dtype, device=device)
    I_n2 = torch.eye(n * n, dtype=dtype, device=device)
    Vp_m = _commutation(m, dtype, device)
    Vp_n = _commutation(n, dtype, device)

    AT_kron = _bkron(A.mT, A.mT)                                   # [B... x m^2 x m^2]
    PB_kron = _bkron(PB, PB)                                       # [B... x m^2 x n^2]
    M2_kron = _bkron(M2, M2)                                       # [B... x n^2 x n^2]

    LHS = PB_kron @ M2_kron @ _bkron(B.mT, B.mT)
    LHS = LHS - _bkron(I_m, PBM2BT) - _bkron(PBM2BT, I_m) + I_m2
    LHS = I_m2 - AT_kron @ LHS
    invLHS = torch.linalg.inv(LHS)                                 # [B... x m^2 x m^2]

    # dA
    rhs = Vp_m + I_m2
    dA_mat = invLHS @ rhs @ _bkron(I_m, A.mT @ M1)
    dA = _unvecT(go @ dA_mat, m, m)

    # dB
    rhs = _bkron(I_n, B.mT @ P)                                    # [B... x n^2 x (n*m)]
    rhs = (I_n2 + Vp_n) @ rhs
    rhs = PB_kron @ M2_kron @ rhs                                  # [B... x m^2 x (n*m)]
    rhs = rhs - (I_m2 + Vp_m) @ _bkron(PBM2, P)
    dB_mat = invLHS @ AT_kron @ rhs
    dB = _unvecT(go @ dB_mat, n, m)

    # dQ
    dQ = _sym(_unvec(go @ invLHS, m, m))

    # dR
    rhs = AT_kron @ PB_kron @ M2_kron                             # [B... x m^2 x n^2]
    dR_mat = invLHS @ rhs
    dR = _sym(_unvec(go @ dR_mat, n, n))

    return dA, dB, dQ, dR


# SECTION: autograd.Function wrappers
class _DiscreteARE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, Q, R):
        Q, R = _sym(Q), _sym(R)
        with torch.no_grad():
            P = _are_forward(A, B, Q, R, discrete=True)
        ctx.save_for_backward(P, A, B, Q, R)
        return P

    @staticmethod
    def backward(ctx, grad_output):
        P, A, B, Q, R = ctx.saved_tensors
        return _dare_backward(P, A, B, Q, R, grad_output)


class _ContinuousARE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, Q, R):
        Q, R = _sym(Q), _sym(R)
        with torch.no_grad():
            P = _are_forward(A, B, Q, R, discrete=False)
        ctx.save_for_backward(P, A, B, Q, R)
        return P

    @staticmethod
    def backward(ctx, grad_output):
        P, A, B, Q, R = ctx.saved_tensors
        return _care_backward(P, A, B, Q, R, grad_output)


# SECTION: public API
def solve_discrete_are(
        A: torch.Tensor,
        B: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        precision: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Solve the discrete-time algebraic Riccati equation (batched, differentiable).

    ``A^T P A - P - A^T P B (R + B^T P B)^{-1} B^T P A + Q = 0``.
    """
    if precision is None:
        return _DiscreteARE.apply(A, B, Q, R)
    original_dtype = A.dtype
    A, B, Q, R = A.to(precision), B.to(precision), Q.to(precision), R.to(precision)
    return _DiscreteARE.apply(A, B, Q, R).to(original_dtype)


def solve_continuous_are(
        A: torch.Tensor,
        B: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        precision: Optional[torch.dtype] = torch.float64,
) -> torch.Tensor:
    """Solve the continuous-time algebraic Riccati equation (batched, differentiable).

    ``A^T P + P A - P B R^{-1} B^T P + Q = 0``.
    """
    if precision is None:
        return _ContinuousARE.apply(A, B, Q, R)
    original_dtype = A.dtype
    A, B, Q, R = A.to(precision), B.to(precision), Q.to(precision), R.to(precision)
    return _ContinuousARE.apply(A, B, Q, R).to(original_dtype)


def test_discrete_are(
        A: torch.Tensor,
        B: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        P: torch.Tensor,
) -> torch.Tensor:
    P = _sym(P)
    ATP = A.mT @ P
    ATPB = ATP @ B
    return ATP @ A - P - ATPB @ torch.linalg.pinv(R + B.mT @ P @ B) @ ATPB.mT + Q


def test_continuous_are(
        A: torch.Tensor,
        B: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        P: torch.Tensor,
) -> torch.Tensor:
    P = _sym(P)
    ATP = A.mT @ P
    PB = P @ B
    return ATP + ATP.mT - PB @ torch.linalg.pinv(R) @ PB.mT + Q
