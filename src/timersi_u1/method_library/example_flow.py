"""Complete candidate example for an agent-verified cumulative-total target.

The hypothesis is semantic, not "the task name contains cumulative". Compare the
point improvement and quantile calibration independently; integrated quantiles
are approximate. Edit or reject this program when the evidence disagrees.
"""
from .exposure_flow import cumulative_flow


def predict(view, tools, memory):
    return cumulative_flow(view, tools, memory, nonnegative_increment=False)
