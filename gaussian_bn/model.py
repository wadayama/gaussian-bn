"""The GaussianDAG model container and its autograd parametrization.

``GaussianDAG`` holds a linear Gaussian Bayesian network -- node dimensions,
edge matrices, and noise / root covariances -- validates them on construction,
and exposes the K-recursion via :meth:`GaussianDAG.k_blocks`. ``k_full`` and
``full_covariance_closed_form`` assemble / cross-check the full covariance.

``pack`` / ``unpack`` turn a model into a flat vector of unconstrained leaf
parameters and back, with a positive-definite-preserving parametrization of the
noise covariances. This single abstraction is shared by gradient training
(:mod:`gaussian_bn.training`), the Fisher metric
(:mod:`gaussian_bn.identifiability`), and natural-gradient optimization
(:mod:`gaussian_bn.optimize`).

Everything is dtype/device agnostic: tensors are cast to the model's ``dtype``
and ``device`` (default ``torch.float64`` / CPU), and the same code runs under
``torch.complex128``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from gaussian_bn.krecursion import compute_k_blocks, get_K
from gaussian_bn.linalg import hermitianize


def _first_tensor(noise, edges) -> torch.Tensor | None:
    """First torch.Tensor found among noise then edge values (for dtype inference).

    The mean does not participate: the model's dtype is set by its covariance
    structure, and the offsets are cast to match (so a stray float32 offset never
    downcasts an otherwise float64 model).
    """
    for s in noise:
        if isinstance(s, torch.Tensor):
            return s
    for v in edges.values():
        if isinstance(v, torch.Tensor):
            return v
    return None


@dataclass
class GaussianDAG:
    """A linear Gaussian Bayesian network in topological order.

    Args:
        dims: ``dims[j]`` is the dimension of node ``j`` (length ``M``).
        edges: ``edges[(i, j)] = A_{ji}`` for each edge ``i -> j`` with ``i < j``
            (shape ``dims[j] x dims[i]``).
        noise: ``noise[j] = Sigma_j`` (length ``M``); for a parentless root this
            is the root covariance ``Sigma_r``, otherwise the innovation
            covariance.
        dtype: Tensor dtype; defaults to ``torch.get_default_dtype()``.
        device: Tensor device; defaults to CPU.
        validate: If ``True`` (default), run constructor validation (topological
            order, edge shapes, square SPD noise, at least one root). Pass
            ``"psd"`` to relax the noise check to positive *semi*-definite,
            which admits deterministic nodes (zero or singular covariance,
            e.g. exact ODE state updates observed through noisy children);
            ``False`` skips validation entirely. Functional updates
            (:meth:`to`, :meth:`replace_edges`, ...) keep the model's mode.
        mean: optional per-node additive offset ``c_j`` in the structural equation
            ``V_j = c_j + sum_i A_{ji} V_i + Z_j`` (each a length-``dims[j]``
            vector). ``None`` (default) is a zero-mean model — fully
            backward-compatible. The implied marginal means are obtained from
            :func:`mean_all`.

    Attributes (derived):
        M, parents, roots, offset, D, has_mean.
    """

    dims: list[int]
    edges: dict[tuple[int, int], torch.Tensor]
    noise: list[torch.Tensor]
    dtype: torch.dtype | None = None
    device: torch.device | None = None
    validate: bool | Literal["psd"] = True
    mean: list[torch.Tensor] | None = None

    def __post_init__(self) -> None:
        self.dims = list(self.dims)
        self.M = len(self.dims)
        # Infer dtype/device from the provided tensors when not given explicitly,
        # so that rebuilding a model from existing tensors never silently downcasts
        # (e.g. to the global float32 default).
        if self.dtype is None or self.device is None:
            ref = _first_tensor(self.noise, self.edges)
            if self.dtype is None:
                # Infer from provided tensors; otherwise default to float64 (this is a
                # scientific covariance library, not the global float32 default).
                self.dtype = ref.dtype if ref is not None else torch.float64
            if self.device is None:
                self.device = ref.device if ref is not None else torch.device("cpu")
        self.edges = {k: torch.as_tensor(v, dtype=self.dtype, device=self.device)
                      for k, v in self.edges.items()}
        self.noise = [torch.as_tensor(s, dtype=self.dtype, device=self.device)
                      for s in self.noise]
        # Per-node offset c_j. None -> zero-mean (the offsets are zeros and
        # has_mean is False, so every mean-aware path is a numeric no-op).
        if self.mean is None:
            self.has_mean = False
            self.mean = [torch.zeros(d, dtype=self.dtype, device=self.device)
                         for d in self.dims]
        else:
            # Explicitly providing a mean marks the model as affine, even if the
            # offsets are currently zero — so the mean path stays active (and
            # differentiable, e.g. for learning offsets from a zero init). The
            # numerics are identical to zero-mean when the offsets are zero.
            self.has_mean = True
            self.mean = [torch.as_tensor(c, dtype=self.dtype, device=self.device).reshape(-1)
                         for c in self.mean]
        self.parents: dict[int, list[int]] = {j: [] for j in range(self.M)}
        for (i, j) in self.edges:
            self.parents[j].append(i)
        for j in self.parents:
            self.parents[j].sort()
        self.roots = [j for j in range(self.M) if not self.parents[j]]
        self.offset = [0]
        for d in self.dims:
            self.offset.append(self.offset[-1] + d)
        self.D = self.offset[-1]
        if self.validate:
            self._validate()

    # -- validation --------------------------------------------------------
    def _validate(self) -> None:
        if len(self.noise) != self.M:
            raise ValueError(f"noise has length {len(self.noise)}, expected M={self.M}.")
        if not self.roots:
            raise ValueError("DAG has no root (every node has a parent).")
        for (i, j), A in self.edges.items():
            if not i < j:
                raise ValueError(f"edge ({i},{j}) violates topological order i<j.")
            if tuple(A.shape) != (self.dims[j], self.dims[i]):
                raise ValueError(
                    f"edge ({i},{j}) has shape {tuple(A.shape)}, expected "
                    f"{(self.dims[j], self.dims[i])}."
                )
        for j in range(self.M):
            S = self.noise[j]
            if tuple(S.shape) != (self.dims[j], self.dims[j]):
                raise ValueError(
                    f"noise[{j}] has shape {tuple(S.shape)}, expected "
                    f"{(self.dims[j], self.dims[j])}."
                )
            if self.validate == "psd":
                w = torch.linalg.eigvalsh(hermitianize(S))
                tol = 1e-10 * max(float(w.max().abs()), 1.0)
                if float(w.min()) < -tol:
                    raise ValueError(f"noise[{j}] is not positive semi-definite.")
            else:
                try:
                    torch.linalg.cholesky(hermitianize(S))
                except RuntimeError as exc:  # not positive definite
                    raise ValueError(f"noise[{j}] is not positive definite.") from exc
        for j in range(self.M):
            if tuple(self.mean[j].shape) != (self.dims[j],):
                raise ValueError(
                    f"mean[{j}] has shape {tuple(self.mean[j].shape)}, expected "
                    f"{(self.dims[j],)}."
                )

    # -- K-recursion -------------------------------------------------------
    def edge_mats(self) -> dict[tuple[int, int], torch.Tensor]:
        """Edge dict keyed ``(child, parent) = (j, i)`` for :func:`compute_k_blocks`."""
        return {(j, i): self.edges[(i, j)] for (i, j) in self.edges}

    def k_blocks(self) -> dict[tuple[int, int], torch.Tensor]:
        """All canonical K-blocks ``K_{jk}`` (``j >= k``)."""
        return compute_k_blocks(
            self.M,
            self.parents,
            self.edge_mats(),
            root_covs={r: self.noise[r] for r in self.roots},
            noise_covs={j: self.noise[j] for j in range(self.M) if j not in self.roots},
            dtype=self.dtype,
            device=self.device,
        )

    # -- indexing helpers --------------------------------------------------
    def slc(self, j: int) -> slice:
        """Slice of node ``j`` inside the stacked ``V_all`` vector."""
        return slice(self.offset[j], self.offset[j + 1])

    def node_index(self, nodes: Sequence[int]) -> torch.Tensor:
        """Long index tensor selecting a set of nodes from ``V_all`` (device-aware)."""
        idx: list[int] = []
        for j in nodes:
            idx.extend(range(self.offset[j], self.offset[j + 1]))
        return torch.tensor(idx, dtype=torch.long, device=self.device)

    # -- functional updates ------------------------------------------------
    def _mean_arg(self):
        """The mean list to pass on when rebuilding (None if zero-mean)."""
        return list(self.mean) if self.has_mean else None

    def to(self, *, dtype: torch.dtype | None = None,
           device: torch.device | None = None) -> "GaussianDAG":
        """Return a copy re-cast to ``dtype`` / ``device``."""
        return GaussianDAG(
            self.dims, dict(self.edges), list(self.noise),
            dtype=dtype or self.dtype, device=device or self.device,
            validate=self.validate, mean=self._mean_arg(),
        )

    def replace_edges(self, new: dict[tuple[int, int], torch.Tensor]) -> "GaussianDAG":
        """Return a copy with some edge matrices replaced/removed.

        Keys mapping to ``None`` are deleted (used by hard interventions).
        """
        edges = dict(self.edges)
        for k, v in new.items():
            if v is None:
                edges.pop(k, None)
            else:
                edges[k] = v
        return GaussianDAG(self.dims, edges, list(self.noise),
                           dtype=self.dtype, device=self.device,
                           validate=self.validate, mean=self._mean_arg())

    def replace_noise(self, j: int, Sigma: torch.Tensor) -> "GaussianDAG":
        """Return a copy with ``noise[j]`` replaced."""
        noise = list(self.noise)
        noise[j] = torch.as_tensor(Sigma, dtype=self.dtype, device=self.device)
        return GaussianDAG(self.dims, dict(self.edges), noise,
                           dtype=self.dtype, device=self.device,
                           validate=self.validate, mean=self._mean_arg())

    def replace_mean(self, j: int, c: torch.Tensor) -> "GaussianDAG":
        """Return a copy with the node-``j`` offset ``c_j`` replaced."""
        mean = list(self.mean)
        mean[j] = torch.as_tensor(c, dtype=self.dtype, device=self.device).reshape(-1)
        return GaussianDAG(self.dims, dict(self.edges), list(self.noise),
                           dtype=self.dtype, device=self.device,
                           validate=self.validate, mean=mean)


# --------------------------------------------------------------------------
# Full covariance assembly + closed-form oracle
# --------------------------------------------------------------------------
def k_full(model: GaussianDAG, K: dict[tuple[int, int], torch.Tensor] | None = None) -> torch.Tensor:
    """Assemble the full ``D x D`` covariance from canonical K-blocks."""
    if K is None:
        K = model.k_blocks()
    Kf = torch.zeros((model.D, model.D), dtype=model.dtype, device=model.device)
    for j in range(model.M):
        for k in range(model.M):
            Kf[model.slc(j), model.slc(k)] = get_K(K, j, k)
    return Kf


def mean_all(model: GaussianDAG) -> torch.Tensor:
    """Stacked marginal means ``m_all`` (shape ``D``) via the mean recursion.

    The linear Gaussian mean decouples from the covariance and follows the same
    topological recursion as the K-recursion, but on vectors:
        ``m_j = c_j + sum_{i in Pa(j)} A_{ji} m_i``  (equivalently ``(I - A)^{-1} c``).
    Returns the zero vector for a zero-mean model.
    """
    m = torch.zeros(model.D, dtype=model.dtype, device=model.device)
    if not model.has_mean:
        return m
    per: dict[int, torch.Tensor] = {}
    for j in range(model.M):
        acc = model.mean[j].clone()
        for i in model.parents[j]:
            acc = acc + model.edges[(i, j)] @ per[i]
        per[j] = acc
        m[model.slc(j)] = acc
    return m


def full_covariance_closed_form(model: GaussianDAG) -> torch.Tensor:
    """Reference covariance via ``V_all = (I - Acal)^{-1} Z``; ``Cov = T Sigma T^H``.

    Non-differentiable test oracle: uses an explicit ``torch.linalg.inv`` as an
    independent cross-check of :func:`k_full` (do not use on the autograd path).
    """
    D = model.D
    Acal = torch.zeros((D, D), dtype=model.dtype, device=model.device)
    Sig = torch.zeros((D, D), dtype=model.dtype, device=model.device)
    for j in range(model.M):
        Sig[model.slc(j), model.slc(j)] = model.noise[j]
        for i in model.parents[j]:
            Acal[model.slc(j), model.slc(i)] = model.edges[(i, j)]
    T = torch.linalg.inv(torch.eye(D, dtype=model.dtype, device=model.device) - Acal)
    return T @ Sig @ T.mH


# --------------------------------------------------------------------------
# Autograd parametrization: pack / unpack
# --------------------------------------------------------------------------
NoiseParam = Literal["chol", "logdiag", "fixed"]


@dataclass
class ParamPack:
    """Bookkeeping that maps a flat parameter vector back to a GaussianDAG.

    Created by :func:`pack`; consumed by :func:`unpack`. Holds the trainable
    edge keys, the noise parametrization mode, and the flat-vector layout.
    """

    free_edges: list[tuple[int, int]]
    edge_shapes: dict[tuple[int, int], tuple[int, int]]
    noise_param: NoiseParam
    noise_nodes: list[int]
    slices: list[tuple[str, object, slice]]  # (kind, key, slice into eta)
    total: int


def _tril_indices(d: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tril_indices(d, d, offset=-1, device=device)


def pack(
    model: GaussianDAG,
    *,
    free_edges: Sequence[tuple[int, int]] | None = None,
    noise_param: NoiseParam = "chol",
    requires_grad: bool = True,
) -> tuple[torch.Tensor, ParamPack]:
    """Flatten a model into a single unconstrained leaf vector ``eta``.

    Free edge matrices contribute their raw entries. Noise covariances contribute
    a positive-definite-preserving parametrization:

    - ``"chol"``: ``Sigma_j = L L^H`` with strictly-lower entries of ``L`` plus a
      ``log`` diagonal -> SPD for any real ``eta``.
    - ``"logdiag"``: ``Sigma_j = diag(exp(.))`` (diagonal noise).
    - ``"fixed"``: noise excluded from ``eta`` (edges-only training).

    Args:
        model: Base model providing the current values and structure.
        free_edges: Edge keys ``(i, j)`` to make trainable; ``None`` means all.
        noise_param: Noise parametrization mode.
        requires_grad: Whether the returned ``eta`` is a leaf requiring grad.

    Returns:
        ``(eta, pack)`` where ``eta`` is a 1-D real-valued (or complex for complex
        edges) tensor and ``pack`` reconstructs the model via :func:`unpack`.
    """
    free_edges = list(model.edges.keys()) if free_edges is None else list(free_edges)
    chunks: list[torch.Tensor] = []
    slices: list[tuple[str, object, slice]] = []
    pos = 0
    edge_shapes: dict[tuple[int, int], tuple[int, int]] = {}

    for key in free_edges:
        A = model.edges[key]
        edge_shapes[key] = tuple(A.shape)
        flat = A.reshape(-1).clone()
        chunks.append(flat)
        slices.append(("edge", key, slice(pos, pos + flat.numel())))
        pos += flat.numel()

    noise_nodes = [] if noise_param == "fixed" else list(range(model.M))
    for j in noise_nodes:
        d = model.dims[j]
        S = hermitianize(model.noise[j])
        if noise_param == "logdiag":
            raw = torch.log(torch.clamp(torch.diagonal(S).real, min=1e-12))
            raw = raw.to(model.dtype)
            chunks.append(raw)
            slices.append(("noise_logdiag", j, slice(pos, pos + d)))
            pos += d
        else:  # "chol"
            L = torch.linalg.cholesky(S)
            li, lj = _tril_indices(d, model.device)
            strict = L[li, lj].clone()  # strictly-lower entries
            logdiag = torch.log(torch.diagonal(L).real).to(model.dtype)
            chunks.append(logdiag)
            slices.append(("noise_chol_diag", j, slice(pos, pos + d)))
            pos += d
            if strict.numel() > 0:
                chunks.append(strict)
                slices.append(("noise_chol_strict", j, slice(pos, pos + strict.numel())))
                pos += strict.numel()

    eta = (torch.cat(chunks) if chunks
           else torch.zeros(0, dtype=model.dtype, device=model.device))
    eta = eta.detach().clone()
    if requires_grad:
        eta.requires_grad_(True)
    pk = ParamPack(free_edges, edge_shapes, noise_param, noise_nodes, slices, pos)
    return eta, pk


def unpack(eta: torch.Tensor, pack: ParamPack, base: GaussianDAG) -> GaussianDAG:
    """Rebuild a :class:`GaussianDAG` from a flat parameter vector.

    Edges not in ``pack.free_edges`` and (for ``noise_param="fixed"``) all noise
    covariances are taken from ``base``. The reconstructed model skips SPD
    re-validation because the parametrization guarantees positive definiteness.
    """
    edges = dict(base.edges)
    noise = list(base.noise)
    chol_diag: dict[int, torch.Tensor] = {}
    chol_strict: dict[int, torch.Tensor] = {}

    for kind, key, sl in pack.slices:
        seg = eta[sl]
        if kind == "edge":
            edges[key] = seg.reshape(pack.edge_shapes[key])
        elif kind == "noise_logdiag":
            noise[key] = torch.diag(torch.exp(seg.real).to(base.dtype))
        elif kind == "noise_chol_diag":
            chol_diag[key] = seg
        elif kind == "noise_chol_strict":
            chol_strict[key] = seg
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown pack segment kind {kind!r}")

    for j, diag in chol_diag.items():
        d = base.dims[j]
        L = torch.diag(torch.exp(diag.real).to(base.dtype))
        if j in chol_strict:
            li, lj = _tril_indices(d, base.device)
            L = L.clone()
            L[li, lj] = chol_strict[j].to(base.dtype)
        noise[j] = L @ L.mH

    return GaussianDAG(base.dims, edges, noise, dtype=base.dtype,
                       device=base.device, validate=False, mean=base._mean_arg())
