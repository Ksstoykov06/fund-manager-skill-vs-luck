"""fundskill: are fund managers skilled or lucky?"""
from .regression import fit_alphas
from .bootstrap import fama_french_bootstrap, BootstrapResult
from .fdr import fdr_table
from . import data, simulate

__all__ = ["fit_alphas", "fama_french_bootstrap", "BootstrapResult", "fdr_table", "data", "simulate"]
