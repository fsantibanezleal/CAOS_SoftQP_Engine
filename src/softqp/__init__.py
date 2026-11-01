"""Native soft-QP optimization, based on OSQP and OSQPSlack."""
__version__ = "0.2.0"
from .interface import OSQP, PenaltyType, SolverStatus, SolverError

from .peak import PeakGroup, PeakProblem, PreparedPeakSolver, solve_peak
