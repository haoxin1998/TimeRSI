"""Use shared historical folds when FEV scaling cannot be computed on early history."""
import numpy as np


def eligible_research_offsets(baseline):
    return [row['offset'] for row in baseline['folds']
            if row['metrics']['SQL'] is not None and np.isfinite(row['metrics']['SQL'])]


def historical_mean(rows, eligible_offsets=None):
    chosen=rows if eligible_offsets is None else [r for r in rows if r['offset'] in eligible_offsets]
    return float(np.mean([r['metrics']['SQL'] for r in chosen])) if chosen else float('nan')


def recovery_shortlist(scores, eligible_offsets):
    shortlist=sorted((n for n,v in scores.items() if np.isfinite(v)),key=lambda n:(scores[n],n))
    # Entries in scores have already produced actual validated R forecasts.
    # Their loss may be unavailable because the data's scaling history is too short.
    return sorted(scores) if not shortlist and not eligible_offsets else shortlist
