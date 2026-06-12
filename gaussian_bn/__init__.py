"""gaussian_bn: a K-recursion covariance backend for linear Gaussian Bayesian networks.

Real-valued (dtype-generic) PyTorch library that maps the local conditional
parameters of a linear Gaussian BN to all node-pair covariance blocks via the
K-recursion, and builds inference (marginalization, conditioning, mutual
information, conditional mutual information), parameter estimation (local-
regression MLE, marginal-likelihood / EM / gradient training), and Fisher-based
edge-identifiability analysis on top of that single differentiable engine.

Ported and hardened from the dtype-generic reference package ``gaussian_dag``.
"""

from __future__ import annotations

from gaussian_bn.design import (
    SensorPlacement,
    d_optimal_score,
    e_optimal_score,
    sensor_placement,
)
from gaussian_bn.identifiability import (
    IdentifiabilityReport,
    edge_fisher,
    fisher_metric,
    identifiability_report,
)
from gaussian_bn.intervention import (
    counterfactual,
    do_cmi,
    do_covariance,
    do_hard,
    do_soft,
)
from gaussian_bn.inference import (
    conditional_covariance,
    conditional_mean,
    conditional_mutual_information,
    marginal,
    mutual_information,
    sample,
)
from gaussian_bn.krecursion import compute_k_blocks, get_K
from gaussian_bn.linalg import (
    cholesky_psd,
    hermitianize,
    logdet_hpd,
    schur_complement,
    solve_psd,
)
from gaussian_bn.model import (
    GaussianDAG,
    ParamPack,
    full_covariance_closed_form,
    k_full,
    mean_all,
    pack,
    unpack,
)
from gaussian_bn.optimize import (
    natural_gradient_ascent,
    natural_gradient_step,
    pga_ascent,
)
from gaussian_bn.projections import project_frobenius_ball, project_total_power
from gaussian_bn.reliability import (
    CRBReport,
    crb,
    crb_report,
    parameter_fisher,
)
from gaussian_bn.training import (
    ObsPattern,
    em_fit,
    fit_gradient,
    fit_gradient_custom,
    fit_local_regression,
    gaussian_nll,
    local_mle_from_cov,
    marginal_likelihood,
    posterior_full_moments,
)

__all__ = [
    "CRBReport",
    "GaussianDAG",
    "IdentifiabilityReport",
    "ObsPattern",
    "ParamPack",
    "SensorPlacement",
    "cholesky_psd",
    "compute_k_blocks",
    "conditional_covariance",
    "conditional_mean",
    "conditional_mutual_information",
    "counterfactual",
    "crb",
    "crb_report",
    "d_optimal_score",
    "do_cmi",
    "do_covariance",
    "do_hard",
    "do_soft",
    "e_optimal_score",
    "edge_fisher",
    "em_fit",
    "fisher_metric",
    "fit_gradient",
    "fit_gradient_custom",
    "fit_local_regression",
    "full_covariance_closed_form",
    "gaussian_nll",
    "get_K",
    "hermitianize",
    "identifiability_report",
    "k_full",
    "local_mle_from_cov",
    "logdet_hpd",
    "marginal",
    "marginal_likelihood",
    "mean_all",
    "mutual_information",
    "natural_gradient_ascent",
    "natural_gradient_step",
    "pack",
    "parameter_fisher",
    "pga_ascent",
    "posterior_full_moments",
    "project_frobenius_ball",
    "project_total_power",
    "sample",
    "schur_complement",
    "sensor_placement",
    "solve_psd",
    "unpack",
]
