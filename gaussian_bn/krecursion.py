"""K-recursion: forward covariance computation for linear Gaussian DAGs.

Model (0-based indexing, topological order so every parent index is smaller
than its child index):

    V_r = root with cov Sigma_r           for every parentless node r
    V_j = sum_{i in Pa(j)} A_{ji} V_i + Z_j,   Z_j ~ N(0, Sigma_j)   otherwise

with the innovations ``Z_j`` mutually independent and independent of the roots.
The K-recursion maps the local conditional parameters ``{A_{ji}, Sigma_j}`` to
all node-pair covariance blocks ``K_{jk} = E[V_j V_k^H]``.

Canonical storage: ``K`` holds only ``K_{jk}`` for ``j >= k`` (lower-triangular
block matrix). Access to ``K_{ab}`` with ``a < b`` uses the Hermitian flip rule
``K_{ab} = K_{ba}^H`` via :func:`get_K`.

This is the real-valued (dtype-generic) port of the reference complex
implementation; for real models ``.mH`` is the ordinary transpose and the
formulas are identical. The single substantive generalization over the
reference is **multi-root** support: any parentless node is treated as an
independent root rather than requiring a unique root at index 0.
"""

from __future__ import annotations

import torch

from gaussian_bn.linalg import hermitianize


def get_K(
    K: dict[tuple[int, int], torch.Tensor],
    a: int,
    b: int,
) -> torch.Tensor:
    """Return ``K_{ab}``, applying the Hermitian flip when ``a < b``.

    ``K`` is assumed to store only canonical keys ``(j, k)`` with ``j >= k``.

    Args:
        K: Canonical K-block dictionary from :func:`compute_k_blocks`.
        a: Row node index.
        b: Column node index.

    Returns:
        The covariance block ``K_{ab}`` (``d_a x d_b``).
    """
    if a >= b:
        return K[(a, b)]
    return K[(b, a)].mH


def compute_k_blocks(
    num_nodes: int,
    parents: dict[int, list[int]],
    edge_mats: dict[tuple[int, int], torch.Tensor],
    root_covs: dict[int, torch.Tensor],
    noise_covs: dict[int, torch.Tensor],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    symmetrize_self_blocks: bool = True,
) -> dict[tuple[int, int], torch.Tensor]:
    """Compute all canonical K-blocks ``K_{jk}`` for ``0 <= k <= j < num_nodes``.

    Implements, in topological order:

    - cross block ``K_{jk} = sum_{i in Pa(j)} A_{ji} K_{ik}`` for ``k < j``
      (a zero block when ``j`` is a parentless root);
    - self block ``K_{jj} = sum_{i,i' in Pa(j)} A_{ji} K_{ii'} A_{ji'}^H +
      Sigma_j`` (which reduces to ``Sigma_r`` for a root ``r``).

    Args:
        num_nodes: Total number of nodes ``M`` (indices ``0..M-1``).
        parents: ``parents[j]`` = list of parent indices of node ``j`` (each
            ``< j``). Parentless nodes are roots; they may be omitted or map to
            an empty list.
        edge_mats: ``edge_mats[(j, i)] = A_{ji}`` for every edge ``i -> j``
            (shape ``d_j x d_i``).
        root_covs: ``root_covs[r] = Sigma_r`` for every parentless node ``r``.
        noise_covs: ``noise_covs[j] = Sigma_j`` for every non-root node ``j``.
        dtype: Optional dtype for the zero cross-blocks of roots; inferred from
            the first available covariance when ``None``.
        device: Optional device for those zero blocks; inferred when ``None``.
        symmetrize_self_blocks: If ``True``, apply ``(A + A^H) / 2`` to each self
            block ``K_{jj}`` to enforce Hermitian structure numerically.

    Returns:
        Dictionary ``K`` with keys ``(j, k)`` for ``0 <= k <= j < num_nodes``.

    Raises:
        ValueError: If a parent violates topological order (``i >= j``), if the
            root/non-root key sets do not partition ``range(num_nodes)``, or if
            no root is present.
    """
    roots = {j for j in range(num_nodes) if not parents.get(j)}
    if not roots:
        raise ValueError("DAG has no root (every node has a parent).")

    expected_roots = set(root_covs)
    expected_nonroots = set(noise_covs)
    if expected_roots != roots:
        raise ValueError(
            f"root_covs keys {sorted(expected_roots)} must match the parentless "
            f"nodes {sorted(roots)}."
        )
    if expected_nonroots != set(range(num_nodes)) - roots:
        raise ValueError(
            "noise_covs keys must be exactly the non-root nodes "
            f"{sorted(set(range(num_nodes)) - roots)}."
        )

    if dtype is None or device is None:
        ref = next(iter(root_covs.values()))
        dtype = ref.dtype if dtype is None else dtype
        device = ref.device if device is None else device

    def dim(j: int) -> int:
        if j in root_covs:
            return root_covs[j].shape[-1]
        return noise_covs[j].shape[-1]

    K: dict[tuple[int, int], torch.Tensor] = {}

    for j in range(num_nodes):
        pa = sorted(parents.get(j, []))
        for i in pa:
            if i >= j:
                raise ValueError(
                    f"Parent {i} of node {j} violates topological order (i < j)."
                )

        # (1) Cross blocks K_{jk} for k = 0, ..., j-1.
        for k in range(j):
            acc: torch.Tensor | None = None
            for i in pa:
                term = edge_mats[(j, i)] @ get_K(K, i, k)
                acc = term if acc is None else acc + term
            K[(j, k)] = acc if acc is not None else torch.zeros(
                (dim(j), dim(k)), dtype=dtype, device=device
            )

        # (2) Self block K_{jj}.
        if j in root_covs:
            acc = root_covs[j]
        else:
            acc = noise_covs[j]
            for i in pa:
                Aji = edge_mats[(j, i)]
                for ip in pa:
                    acc = acc + Aji @ get_K(K, i, ip) @ edge_mats[(j, ip)].mH
        K[(j, j)] = hermitianize(acc) if symmetrize_self_blocks else acc

    return K
