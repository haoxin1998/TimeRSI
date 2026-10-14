"""Research context comparison and explicit executable/analysis separation."""
from pathlib import Path
import shutil
import numpy as np


GROWING_GUIDE = '''
## Research an algorithm that grows with its history
Read CONTEXT-COMPARISON.json before implementing. Learn a forecasting algorithm
that refits to each supplied history. Distinguish a mechanism that failed with
short history from one that lacks support until longer history becomes available.
Compare the research and current forecast contexts: history length, calendar
coverage, entity count and known future inputs. A short pseudo-backtest from the
same season does not establish that trend or seasonal mechanisms are useless.
Consider how the same code will estimate its parameters as history grows, and how
pooling depends on the entity population. You choose the mechanisms and fitting
scope. Do not overfit historical data.
Use the remaining probe slots for meaningful algorithm alternatives or adaptation
mechanisms as well as refinement of a supported candidate. Complete the supplied
initial probe budget when useful alternatives remain; a flat constant sweep on
one fold alone does not establish that no algorithmic improvement is possible.
Synthetic histories can check behavior at longer lengths but are not evidence of
real forecast accuracy. Independent later selection is performed by the executor.

Write the executable program in program/candidate.py, with runtime helpers and
edited method_library/ files inside program/. Keep exploratory scripts and their
results in eda/ and research conclusions in notes/. For local Python experiments,
add program/ to sys.path before importing candidate and its runtime helpers.
The executor freezes program/ only; it archives eda/, notes/, HYPOTHESIS.md and
REFLECTION.md separately. Read RESEARCH-HISTORY.json for previous research notes,
local experiments and session traces from this run. Save notes/RESEARCH.md with
the useful hypotheses, rejected alternatives and conditions still untested, so
the next research session can continue the work.
'''

OLD_STYLE_GUIDE = '''You are the autonomous code researcher for a frozen TimesFM3
forecasting model. Read INPUT.json, API.md, CONTEXT-COMPARISON.json, parent.py,
research-data/INDEX.json, history.json and RESEARCH-HISTORY.json. Optimize official
historical SQL by developing a complete forecasting algorithm.

Do real exploratory analysis and multi-file Python experiments. Use the full R
inputs, historical truth, native point and nine quantile predictions, and per-entity
errors. Construct meaningful calendar/semantic variables, change inputs or context,
fit causal CPU specialists, compose modules and calibrate uncertainty. Each new
forecast has its own historical data: your program may refit parameters and enable
features as that history grows. The tools and editable method_library/ are optional
execution references; you may author any runnable numerical mechanism yourself.
Do not overfit historical data; seek mechanisms useful across time and entities.

Write program/candidate.py defining predict(view, tools, memory) -> Forecast.
Runtime helpers belong in program/; local experiments and results belong in eda/.
Save evidence, rejected alternatives and next questions in HYPOTHESIS.md,
REFLECTION.md and notes/RESEARCH.md. Use INPUT.json's python_executable. For local
experiments add program/ to sys.path. research_sdk.run_local evaluates a complete
CPU/unchanged-Raw program with exact metrics and saves local predictions.
After a real probe, pass archive/probe-NNN/R-native/fold-<offset>/ as
native_directory to run_local to replay its exact native variants while editing
CPU fitting, composition or uncertainty calibration. New native inputs still
require a real probe; replay is not a new evaluated submission.

You have a live executor: after a runnable change, run `<python_executable> probe.py`.
It freezes program/ and returns actual R metrics in THIS conversation. Inspect
archive/probe-NNN for code, predictions, per-entity errors and research notes.
Revise, restore a useful branch, remove a failed part, or implement a complementary
mechanism and probe again. `probe.py --parent raw` creates an independent branch.
Use the supplied probe_budget for informative algorithm experiments; submit useful
alternatives before ending. Failures consume slots but do not erase earlier programs.
All successful submitted programs and Raw remain eligible for independent S selection
after the session. S and current future target values remain with the executor.

Evaluate full point and nine-quantile forecasts in original units/order. Fit each
historical evaluation from that origin's own past, using only known future inputs.
Research truths can score experiments; they are not input features for forecasts.
Use this workspace and public history only; foundation calls go through the executor.
Leave complete runnable code and research notes for the next session. Continue useful
mechanisms in later epochs using actual matured_deployment_feedback and full archives.
Its MAE/quantile loss fields are unscaled; when available, compare the separately
reported official_SQL_by_target and relative_official_SQL_change before concluding
that a deployed revision improved the objective. Opposite directions are possible
when targets/entities have different scales. SQL cells use the original origin's
history and become visible only after that forecast matures; they are not current
S labels or benchmark totals.
RESEARCH-HISTORY.json indexes previous sessions' public_archive and research_data,
so you may inspect their exact R inputs, native snapshots, predictions and failures,
not only their final notes. Older R results are evidence at their own origins, not
an untested claim about today's longer history or a new successful submission.
Research forecast persistence and the centre as learning rules, not only fixed
constants optimized across the current R folds. Short-lived growth, decay, event
effects and cross-target relations need not persist for the entire horizon.
Where useful, fit these parameters from rolling outcomes INSIDE each supplied
past, including completed full-horizon backtests when available. Keep meaningful
growth and no-growth alternatives; neither unconditional extrapolation nor a
universal ban on growth is the goal. Internal calibration should replay the SAME
complete pipeline used for prediction, including coupling, transformations and
mixtures: residuals of a stripped-down projector do not validate the final centre
or quantiles. Diagnose systematic centre errors separately from interval width.
Test the resulting complete program with real probes. You choose whether this
mechanism is appropriate; no named task, fixed expert or required recipe is implied.
Do not run git add/commit/push; project versioning is managed outside this session.
'''

NATIVE_GROWTH_GUIDE = '''
## Foundation model roles as context grows
Research which parts of the native forecast are useful at different history lengths:
level, trend, remaining horizon shape and interval scale. A poor three-observation
native path does not establish that a longer-history path should be discarded.
Likewise a good short-history specialist need not remain the only reliable source.
Compare a structural specialist alone with complete programs that allow native
information to contribute differently as historical support increases. Estimate
coefficients/scales from each supplied past and let the same code adapt at later origins.

Test level, horizon variation and probability calibration as separate mechanisms.
For a positively supported structure, native_residual_fusion offers editable gains,
history-dependent reliability, threshold or smooth disagreement weighting, and
choices about shape/width at known event positions. These are reference choices;
you can write an alternative. Ordinary expert/native disagreement need not imply
that native information is harmful. Native point and q50 can also differ; choose
the meaningful signal for your probabilistic forecasting mechanism.
Use the larger budget for meaningful alternatives and context adaptation, rather
than only refining constants on the same short research fold. Synthetic checks
verify algorithm behavior, while real probes and subsequent feedback assess accuracy.
When short-history R cannot distinguish plausible longer-history behaviors, retain
complete runnable alternatives with different adaptation rules rather than treating
a near-tie on R as evidence that both algorithms are equivalent. All successful
programs remain eligible for independent S selection; no S labels are exposed to you.
Examine SQL's quantile loss as well as point error: better points or a narrower
interval alone do not establish better probabilistic forecasts.
'''

PLACEMENT_GUIDE = '''
## Research where an observed mechanism belongs
For a semantic or calendar signal supported by the data, compare meaningful ways
to use it: native input features, an independently fitted specialist, and forecast
or quantile corrections. These are alternative implementations of a hypothesis,
not a mandatory pipeline or a quota. Test the complete resulting algorithms.
A post-processing implementation that loses on R does not establish that the
signal would also fail as a native input feature; likewise a native input gain
does not establish that every external correction should be added to it.
Use real native probes when changing inputs. Use recorded native outputs for
cheap fitting/calibration experiments. Preserve distinct runnable alternatives
when their mechanisms differ, even if an earlier R score is slightly worse.
Keep algorithm growth and uncertainty coherent as each forecast history changes.
Record which implementations you actually tested and what remains a hypothesis.
You may reject the signal or develop another mechanism; choose from evidence.
'''


def summarize(view):
    lengths = [it['y'].shape[1] for it in view.items]
    dates = [np.asarray(it['timestamps']).astype('datetime64[D]') for it in view.items]
    past_dates = [d[:n] for d,n in zip(dates,lengths)]
    future_dates = [d[n:] for d,n in zip(dates,lengths)]
    return {'entities':len(view.items), 'targets':len(view.targets),
        'frequency':view.frequency, 'horizon':view.horizon,
        'history_length':{'min':min(lengths),'median':float(np.median(lengths)),'max':max(lengths)},
        'past_start':str(min(d[0] for d in past_dates)),
        'past_end':str(max(d[-1] for d in past_dates)),
        'past_months':sorted({int(str(t)[5:7]) for d in past_dates for t in d}),
        'future_start':str(min(d[0] for d in future_dates)),
        'future_end':str(max(d[-1] for d in future_dates)),
        'known_columns':sorted({n for it in view.items for n in it['known']})}


def context_comparison(research_views, current):
    return {'research':{str(k):summarize(v) for k,v in research_views.items()},
        'current':summarize(current),
        'scope':'Visible inputs only; no selection labels or current future targets'}


def executable_root(workspace):
    workspace=Path(workspace)
    return workspace/'program' if (workspace/'program/candidate.py').exists() else workspace


def archive_analysis(workspace, destination):
    workspace=Path(workspace);destination=Path(destination)
    for name in ('eda','notes'):
        if (workspace/name).is_dir():
            shutil.copytree(workspace/name,destination/name,dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('HYPOTHESIS.md','REFLECTION.md','CONTEXT-COMPARISON.json'):
        if (workspace/name).exists():
            destination.mkdir(parents=True,exist_ok=True)
            shutil.copy2(workspace/name,destination/name)
