# Editable forecasting mechanisms, not an answer catalogue

This directory is research material. Copy, modify, combine, or ignore it. Your
deliverable is a complete `predict(view, tools, memory)` program, not a method ID.
There is no required pipeline order. These references are not validated winners
for the current task and contain no previous task predictions or fitted champions.

## Start with an explanation, not a module name

Read the task and field descriptions, actual dates, values and Raw errors. State
what a variable measures and whether its future is supplied as known. For example,
an accumulated total and a daily flow are different targets; an operating fraction
and an on/off indicator need different interpretations. A dataset name alone is
not evidence for a numerical rule.

Useful experiment choices include a new mechanism, improving a promising branch,
removing an unsupported component, restoring an earlier program, and composing
complementary programs. These are choices, not a quota. Keep a useful runnable
program while investigating a harder alternative. Use the complete history and
probe archives to avoid rediscovering an already tested failure.

## Preserve the roles inside a strong program

**Structure and unpredictable variation are not interchangeable.** A specialist
may explain exposure, events and deterministic shape while a foundation model
adds level or trend information. Averaging both complete forecasts can dilute the
very mechanism that helped. `native_residual_fusion.py` exposes separate level,
trend, remaining-shape and interval contributions. You provide a justified normal
position mask and historical residual scale; you can change or reject the design.

**Representation changes need restoration.** `exposure_flow.py` contains rate /
exposure and cumulative / increment examples. Keep past and future covariates
aligned and return the original units. A cumulative quantile obtained by summing
increment quantiles is an approximation, not a guaranteed calibrated distribution.

**A point improvement need not improve SQL.** Check the point and all nine
quantiles. Fit uncertainty for the actual predictor you deploy. If you add pooling,
cross-target coupling, or a new transformation, internal backtests of an older
simpler predictor no longer measure the new predictor's errors. Re-run the actual
generation path at historical cutoffs, or explicitly test the approximation.

**Short history calls for a smaller hypothesis.** The rich `event_structure.py`
reference defaults to at least 30 eligible observations. Below that threshold it
does not produce a specialist. Lowering this number alone does not make a rich
calendar fit appropriate. An exposure-adjusted level with one supported effect
can be a different, simpler branch: `short_history_exposure.py` starts with a
rate intercept and adds effects as historical support grows; the complete editable
entry is `example_short_exposure.py`. You supply exposure and regressors explicitly.
Its small-sample uncertainty includes a stated prior, not a claim of measured
out-of-sample coverage. Do not assume that native covariates should be
removed just because history is short: test that change using real native calls.

**A component can help a mixture even if it loses alone.** Evaluate the complete
composed program. Conversely, a strong standalone specialist does not imply that
every additional blend or routing layer will improve it. Raw, the current parent,
and previous complete programs remain meaningful alternatives.

**Pooling changes the fitted model, not just execution speed.** If an expert fits
jointly across entities, a small research sample and the full deployment population
can produce different coefficients even for the same entity and cutoff. Compare
the intended local, pooled or partially pooled mechanism using its actual fitting
scope; do not treat one as a faster but otherwise identical version of another.

**Native reliability can grow with context.** A three-observation native path can
be poor while a longer-history path supplies useful variation and interval scale.
Compare policies that enable these contributions as history grows. In
`ResidualFusionConfig`, `disagreement_mode='smooth'` attenuates every disagreement;
`'threshold'` attenuates only disagreement above the chosen scale. These are
editable hypotheses. Ordinary disagreement need not be evidence that the native
signal should be suppressed. `shape_on_events` and `spread_on_events` let you
explicitly test whether native variation at known event positions is useful.

## Feedback and selection

Use supplied R folds for experiments and inspect variation across available
origins, entities, targets and regimes rather than only one aggregate. Do not
overfit historical data. A recent winner is evidence, not a permanent rule. The
executor performs later historical selection; only already observed outcomes may
inform a new forecast. Never seek current/future formal labels or other tasks'
private files. The program may adapt to current known inputs without those labels.

Each real probe freezes your full Python bundle and returns R feedback. Read the
paths in its response rather than guessing filenames. `API.md` explains how to
load official research arrays, score CPU predictions, and request actual native
evaluation. Native calls, scoring and archiving are execution tools; they do not
limit the algorithms you can write.
