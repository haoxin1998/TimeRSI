"""Claude Code proposal transport, with complete per-iteration workspaces."""
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from contextlib import nullcontext
from .core import write
from .research_design import GROWING_GUIDE, OLD_STYLE_GUIDE, NATIVE_GROWTH_GUIDE, PLACEMENT_GUIDE, executable_root


def long_context_model(model):
    base=model.removesuffix('[1m]')
    names={'fable':'claude-fable-5','opus':'claude-opus-5',
           'claude-fable-5':'claude-fable-5','claude-opus-5':'claude-opus-5'}
    return names[base]+'[1m]' if base in names else model

GUIDE = '''You are researching an adaptive time-series forecasting program around frozen TimesFM3.
The goal is to discover useful mechanisms from this task's visible data, not to pick
an answer by dataset name or replay predictions from a previous run. Read INPUT.json,
API.md, history.json and parent.py. Write candidate.py defining predict(view, tools, memory).

## Research freedom
Use Bash and INPUT.json python_executable for substantive exploratory data analysis,
causal CPU model fitting and small controlled experiments. Inspect the actual data,
form an explanation for forecast errors, implement it, test it and revise it using
feedback. You decide how much analysis is needed before the first candidate, which
mechanisms to explore and in what order. There is no required pipeline order or
quota of different mechanisms. A new branch, an ablation, restoring a good earlier
program, combining complementary programs, or improving one well-supported mechanism
are all valid. Spend the budget on informative experiments rather than repetitive
weight sweeps when a missing mechanism is the main issue.

You may create and modify multiple Python files/packages and use installed numpy,
pandas, sklearn and lightgbm. Fit additional causal CPU experts, create new features,
change target representations, implement conditional routing or calibration, and
combine forecasts in any order. The tools API is convenient execution infrastructure,
NOT the complete action space. TimesFM's weights stay frozen; CPU experts may fit
each supplied view's historical data. A complete statistical or learned CPU expert
may replace TimesFM entirely when the evidence supports it: neither a native call
nor a nonzero Raw weight is required inside your candidate. You may also choose
between complete experts conditionally, not only add a correction to Raw.
Do not execute GPU models directly: native
inference is owned by the executor. Submit native changes for real executor evaluation.
The executor runs your proposal after submission and returns R feedback to the
next iteration; model probes are not available directly inside this session.

## Editable method references and own-run memory
method_library/, when present, contains optional, editable reference code and notes,
not task answers or an obligatory recipe list. Read the implementations, adopt the
parts justified by visible semantics and errors, change them, compose them or write
your own. Do not assume that a reference method improves this task without testing it.
method_library/README.md, when supplied, explains mechanism roles, composition and
failure lessons. Use it to navigate relevant code, not as a mandatory experiment plan.
All executable Python helpers, including your edited method_library/, are frozen with
the candidate. Library imports use `from method_library.<module> import ...`.
history/ contains all earlier public iterations supplied from THIS run, including
complete code bundles, hypotheses and R feedback. Read any of them, restore their
code into the working directory, branch from them or reuse their helpers. Restoring
code is not the same as reading an old cached prediction: the executor reevaluates
the restored program on current historical windows. history.json is a compact index;
use bounded Read offset/limit for large detailed R-FEEDBACK.json files. No old-run
champion list, task-specific answer table or benchmark score lookup is provided.

## Data and evaluation
research-data/INDEX.json indexes full historical R inputs, truth, Raw point and
quantile forecasts in npz files (numpy.load, allow_pickle=False). These full arrays,
not only INPUT.json's small summary samples, support exploratory analysis. Use
timersi_u1.research_sdk.load_fold and score_forecast for exact CPU tests with the
provided interpreter. They use the executor's official metrics and scaling histories;
do not approximate SQL by an unscaled loss ratio or use mock views missing fields.
CachedRaw supports unchanged Raw inference plus shift/spread/blend, not new native
features or transforms. After a real probe, its archive/R-native/fold-<offset>/
contains exact native variants already evaluated on that R fold. Pass it as
native_directory to research_sdk.run_local for cheap full-program CPU experiments.
Only matching inputs replay; new native inputs still require a real probe.
Native asinh inverse quantiles can have extreme tails even
when points improve: evaluate probabilistic accuracy as well as point accuracy.
Every fit must use the supplied view's past. Only explicitly known future covariates
may be used. Original covariates already enter Raw, so test whether an added expert
duplicates or genuinely improves their effect. Preserve target units and output order.
Do not overfit historical data. Prefer explanations supported across time, entities
and regimes. Evaluate complete programs and mixtures, not just isolated components.
Selection windows and current/future official labels are not available for research.

INPUT.json current_forecast_inputs and research_contexts describe visible dates,
covariates and regimes. Compare their situations to formulate testable adaptations.
Do not invent missing domain facts or turn a dataset name into a fixed prediction rule.
INPUT.json matured_deployment_feedback shows what earlier programs from THIS run
actually did versus Raw. Its records end before the oldest current selection window.
MAE/quantile-loss fields are unscaled. When present, official_SQL_by_target and
relative_official_SQL_change use each original forecast origin's exact FEV scaling
history and only matured visible entities. Compare the SQL signal separately:
unscaled loss can improve while SQL worsens. Partial visible-cell means are not
full-task or benchmark scores. Repeated failures may call
for fixing the mechanism, restoring a prior branch or starting a new one. Current
selection labels and future official outcomes remain hidden.
Residual corrections must match their forecast source: Raw errors describe Raw,
not an unrelated expert blend. Correct Raw before blending, or test source='self'
using this exact code bundle's mature out-of-sample errors. Revised code initially
has no self-errors; a parent's residuals are not automatically the child's.

## Budget, outputs and workspace
INPUT.json session_wall_limit_seconds and agent_max_turns are actual session limits;
an interactive session also supplies its real probe_budget. Allocate analysis,
implementation and evaluation within those limits, leaving time for a runnable
submission. These are upper bounds, not a requirement to fill the budget with minor
variations. Save useful working drafts during research and explain the hypothesis,
evidence and rejected alternatives in HYPOTHESIS.md. The deliverable is executable
code, not merely a suggested strategy. The executor evaluates the full program and
its 25%/50% parent mixtures on historical R windows; you may implement other mixtures.
Use this workspace only. Do not fetch external data, install packages, inspect
credentials or read private executor/other task directories. Do not run git
add/commit/push: the harness archives programs and project versioning is managed
outside the code agent. Public history is read-only; edit copies in the workspace.
'''

API = '''# Executable API (adaptive research harness 133)
This API supplies shared native inference, historical evaluation and convenient
forecast operations. It does not limit the algorithms you may implement. Full
Python helpers, causal CPU model training, optional editable method_library/ code,
custom representations and data-dependent routing are supported. Each candidate is
a complete program, not a choice from a fixed menu. No task-name dispatch is needed.
Calling tools.timesfm is optional: a complete causal CPU/statistical expert may
return Forecast directly, including its own point and quantile estimates. The
executor still computes Raw independently as a comparison, not as a compulsory
component of your program.

## CPU research helper (run using INPUT.json python_executable)
```python
import json
from pathlib import Path
from timersi_u1.research_sdk import load_fold, score_forecast, CachedRaw
research_dir = Path('research-data')
index = json.loads((research_dir / 'INDEX.json').read_text())
offsets = [fold['offset'] for fold in index['folds']]
# Enumerate actual offsets: short history may expose only offset 2, not 3 or 4.
view, raw, evidence = load_fold(research_dir, offset=offsets[0])
truth = evidence['arrays']['truth']  # [N,H,D]; NOT evidence['truth']
dates = view.items[0]['timestamps']  # [L+H]; NOT evidence['timestamps_0']
print(score_forecast(raw, evidence))
cpu_tools = CachedRaw(view, raw)
adjusted = cpu_tools.shift(cpu_tools.timesfm(view), 0.1)
print(score_forecast(adjusted, evidence))
# For a complete CPU/unchanged-Raw program, preserve local experiment results:
# from timersi_u1.research_sdk import run_local
# forecast, metrics = run_local('research-data', offsets[0], 'program', 'eda/local-001')
```
load_fold returns (View, Forecast, evidence). evidence has exactly the top-level
keys 'spec' (that fold's INDEX metadata) and 'arrays' (the named NPZ arrays).
evidence['arrays'] includes truth, raw_point, raw_quantiles, metric_y_past and
metric_y_past_lengths. For typed inputs, each view.items[i] has the keys
'timestamps', 'y', 'known' and 'past'. INDEX.json maps item-specific storage keys
if you need the raw NPZ layout.
Truth is present only for the published R folds, not selection/current future.

CachedRaw is a CPU fixture, not a native model simulator. It supports unchanged
cpu_tools.timesfm(view) on that SAME original view object, plus shift/spread/blend.
It does not run a new context, transformed target, generated feature, altered view,
or native expert. For those changes, write a complete candidate.py and, in an
interactive research session, run `<python_executable> probe.py`: the outer executor
performs REAL model calls and returns R feedback. Pure CPU experts can be developed
and scored locally with score_forecast before submitting their complete program.

## Probe archives and long CPU analysis
The probe response provides 'archive' (for example archive/probe-000) and program
names in 'results'. Use those names rather than guessing a candidate filename.
For a returned program_name and an actual offset from INDEX.json, the saved file is:
  <reply['archive']>/R-predictions/fold-<offset>/<program_name>/predictions.npz
That NPZ contains 'point' and 'quantiles'; METRICS.json and ERROR-DETAILS.json are in
the same directory. It is NOT R-predictions/fold-<offset>/<program_name>.npz.
Complete source is under <reply['archive']>/code/ and feedback in R-FEEDBACK.json.

For cheap full-program experiments with a native variant that has already had a
real R probe, use that same fold's native archive:
```python
from timersi_u1.research_sdk import run_local
forecast, metrics = run_local('research-data', offset, 'program', 'eda/local-002',
    native_directory=f"{reply['archive']}/R-native/fold-{offset}")
```
Set reply to the JSON response from probe.py and offset to an actual R offset.
R-native/INDEX.json identifies the exact native input fingerprints; output arrays
are point/quantiles in original target units, before candidate blending/correction.
This is not candidate-final output masquerading as TimesFM, nor a model simulator.
Unchanged native calls can replay while you revise CPU experts, composition or
calibration. A changed native input missing from this archive requires a real probe.
run_local never launches a foundation model. Supply memory explicitly only when
replaying a program that consumes the same public, matured memory; its default is [].
Public archives remain read-only; new expert fits and local outputs use scratch/eda.

## Optional pure CPU-step cache
tools.cached_array(operation, inputs, compute) caches a numerical array under the
full current program identity and exact typed input/parameter values. For repeated
causal fitting, pass every relevant past array, known-future input, date/frequency
and hyperparameter in inputs, and use a zero-argument compute callback. For example:
  pred = tools.cached_array('ridge-pred', {'X': X, 'y': y, 'future_X': Xf, 'alpha': 10.},
      lambda: Ridge(alpha=10.).fit(X, y).predict(Xf))
Use only for a pure step; include a fixed seed when needed, and don't cache a step
that depends on omitted memory or mutable state. Changed program source, data or
parameters do not reuse that result. It does not import an old-run model or remove
historical fitting. Numerical EVENTS.jsonl records hits/misses in the executor.

CPU fitting or rolling backtests can take longer than Bash's default command
timeout even while the agent session has plenty of time remaining. For such work,
set an explicit Bash tool timeout in milliseconds (for example timeout=600000),
within the remaining session budget, or use its background workflow and inspect
completion. A command timeout is not evidence that the forecasting hypothesis failed.
This is execution guidance, not a limit on analysis depth or a required probe schedule.

## Forecast program API
predict(view, tools, memory) -> Forecast
view.items: list of dicts with y[D,L], timestamps[L+H], known{name:array[L+H]},
past{name:array[L]}, static, id. view.horizon, seasonality, targets, frequency.
item['observed'][D,L], when supplied, marks real historical target observations
before forward filling. Native inputs remain filled y; fit CPU labels and residual
scales on real observations rather than treating imputed targets as measurements.
view.facts includes task/column descriptions when provided by the dataset (may be null).
item['known_raw'] and item['past_raw'] preserve original categories/values before numeric
encoding; use these for semantics, e.g. holiday code '0' means no holiday, while its
hashed numeric value in known is not necessarily zero. Known raw values include
the known future; past_raw stops at the cutoff. Do not invent missing task facts.
Forecast has point[N,H,D] and quantiles[N,H,D,9]. numpy is available as np; Forecast is in globals.
tools.timesfm(view, features=None, context=15360, transform='identity', signed=False)
  transform: identity/asinh/flow. features(timestamps,known,context_length) returns
  dict of finite arrays length L+H. Fit feature scales on the past prefix only.
tools.native_flow(view, uncertainty='sum' or 'rss', signed=False, context=15360,
                  features=None, covariate_differences=())
  Real TimesFM increment forecasts restored to stock units. Use only when an
  accumulated-total representation is appropriate. Optional named numeric past
  columns get observed first-difference channels in addition to their original
  values. SUM/RSS are alternative interval approximations, not calibration guarantees.
  signed=True permits negative increment predictions; signed=False requests native
  conditional nonnegativity. Compare full point and quantile losses, not just centres.
tools.calendar is such a feature function. Original covariates are preserved.
tools.calendar_expert(view, weight=.5, spread_weight=.6, recent=730)
  Past-fit multiplicative calendar/known-covariate decomposition, native prediction
  of the de-seasonalised level, restoration and log-space expert/interval mixture.
  Fits per entity, uses original categorical values; genuine Open/is_open fields
  license closed-day zeros. Signed or insufficient-history entities keep native.
  Useful hypothesis for positive calendar-driven series, not a task-name rule.
tools.asinh_center(view, weight=1.)
  Apply asinh TimesFM's point adjustment to Raw point and all Raw quantiles.
  Does NOT use the inverse-asinh quantile tails. Compare with identity/asinh;
  it is not guaranteed to improve SQL or remain nonnegative.
tools.scene(view, forecast=None, max_items=4, max_targets=8) returns compact causal
  calendar, level/volatility and known-future covariate summaries. Counts describe the
  full view; items/targets are only the requested prefix. No future target values.
tools.expert(view, kind='ridge' or 'lgbm', lags=None, recent=2048, fit_scope='item')
  Recursive lag/calendar/known-covariate prediction. fit_scope='task' fits one shared
  model per target across entities; training stops at the earliest entity cutoff.
  fit_scope='item' fits each item's past. Same-input requests reuse this run's cache.
  Its quantiles use in-sample residual spread, NOT validated probabilistic calibration.
tools.seasonal(view) -> seasonal naive forecast with historical difference intervals.
tools.blend([forecast1,forecast2], weights=[0.75,0.25]) -> Forecast
tools.shift(forecast, delta) -> moves point AND all quantiles; delta broadcasts to [N,H,D].
tools.spread(forecast, factor=1.) -> changes interval width around q50, leaves point unchanged.
tools.residual(forecast, view, memory, weight=.25, source='raw', max_windows=3)
  Shifts point/all quantiles by recent mature median residuals, aligned by entity and target.
  source='self' uses this exact code bundle's previous out-of-sample forecast errors.
  It is a no-op until that program has mature history. source='raw' should normally
  correct the raw forecast BEFORE blending, not shift an unrelated mixture wholesale.
memory contains only records fully observable at each entity's current cutoff; may be empty.
Each record has source, ids, targets, ends, point[N,H,D], quantiles and truth.
New records also have starts, aligned with ids; older records may lack this field.
starts/ends are lists of ISO date strings. For numerical comparisons use
np.asarray(record['ends'], dtype='datetime64[ns]'); NumPy max on a string array
is not a supported datetime reduction. The converted array's max() is supported.
Deployment feedback supplies forecast_starts/forecast_ends: use these actual dates
to identify seasons/events, rather than interpreting an official window number as a date.
Raw residuals
are from genuine earlier predictions of THIS run, not fitted training errors or old runs.
The record's source is 'raw' or a concrete 'program:<content-hash>' identity,
NOT the literal string 'self'. source='self' is a selector understood by
tools.residual, not a label stored in memory. During predict(), tools.active_program
holds this exact bundle's identity. For custom multiplicative/log corrections,
filter records with rec['source'] == tools.active_program, then align ids/targets.
Filtering rec['source'] == 'self' silently selects no records. Different code has
a different identity and initially no own errors; do not replace them with Raw errors.
You may directly implement additional causal numerical transformations and Forecast outputs.
Every output must preserve entity/target order, original units and sorted finite quantiles.
Example:
def predict(view, tools, memory):
    raw = tools.timesfm(view)
    extra = tools.timesfm(view, features=tools.calendar)
    return tools.blend([raw, extra], [0.75, 0.25])
'''


def readable_json(value,depth=0):
    """Compact scalar arrays without turning the entire packet into one huge line."""
    pad='  '*depth;child='  '*(depth+1)
    if isinstance(value,dict):
        return '{\n'+',\n'.join(child+json.dumps(str(k),ensure_ascii=False)+': '+readable_json(v,depth+1)
            for k,v in value.items())+'\n'+pad+'}'
    if isinstance(value,list):
        if not value:return '[]'
        if all(not isinstance(v,(dict,list)) for v in value):
            entries=[json.dumps(v,ensure_ascii=False) for v in value];lines=[];line=''
            for entry in entries:
                if line and len(line)+len(entry)>1500:lines.append(line);line=''
                line+=(', ' if line else '')+entry
            if line:lines.append(line)
            return '['+(',\n'+child).join(lines)+']'
        return '[\n'+',\n'.join(child+readable_json(v,depth+1) for v in value)+'\n'+pad+']'
    return json.dumps(value,ensure_ascii=False)


def budget_draft_usable(exit_code,expired,results,candidate):
    """Evaluate actual written code after a turn cap; retain the abnormal receipt."""
    if expired or exit_code!=1 or not results or not candidate.exists():return False
    if results[-1].get('subtype')!='error_max_turns':return False
    try:compile(candidate.read_text(),str(candidate),'exec')
    except (SyntaxError,UnicodeError):return False
    return True


def publish_method_library(workspace, source=None):
    """Expose editable source references, never imported task-specific predictions."""
    source=Path(source) if source is not None else Path(__file__).parent/'method_library'
    if not source.is_dir():return []
    published=[]
    for original in sorted(source.rglob('*')):
        relative=original.relative_to(source)
        if not original.is_file() or original.suffix not in ('.py','.md'):continue
        if '__pycache__' in relative.parts:continue
        target=Path(workspace)/'method_library'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(original,target)
        published.append(str(Path('method_library')/relative))
    return published


def vertex_session_route(folder, model, effort, project, region='global'):
    """Isolate only an explicitly requested provider route, not research budgets."""
    if not project:return [],{}, {'isolated':False}
    folder=Path(folder).resolve()
    config=folder/'claude-config';config.mkdir(parents=True,exist_ok=True)
    # Runtime cache/debug is evidence, not reusable candidate source or credentials.
    (config/'.gitignore').write_text('*\n')
    routing={'CLAUDE_CODE_USE_VERTEX':'1','ANTHROPIC_VERTEX_PROJECT_ID':project,
        'CLOUD_ML_REGION':region,'ANTHROPIC_VERTEX_REGION':region,
        'GOOGLE_CLOUD_QUOTA_PROJECT':project,'ANTHROPIC_MODEL':model}
    settings=folder/'AGENT-VERTEX-SETTINGS.json'
    write(settings,{'env':routing,'model':model,
        'thinking':{'enabled':True,'effortLevel':effort}})
    options=['--setting-sources','','--settings',str(settings),'--debug','api']
    overrides={**routing,'CLAUDE_CONFIG_DIR':str(config)}
    record={'isolated':True,'configured_project':project,'configured_region':region,
        'configured_model':model,'config_directory':str(config),'settings_path':str(settings),
        'setting_sources':[],'budget_overrides':{},
        'expected_resource_path':f'/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{model}:streamRawPredict'}
    return options,overrides,record


def observed_vertex_routes(route):
    """Retain paths and source labels only, never debug headers or credentials."""
    if not route.get('isolated'):return []
    entries=[]
    for log in sorted((Path(route['config_directory'])/'debug').glob('*.txt')):
        for line in log.read_text(errors='replace').splitlines():
            if '[API REQUEST]' not in line:continue
            match=re.search(r'(/v1/projects/([^/\s]+)/locations/[^\s]+)',line)
            if not match:continue
            source=re.search(r'\bsource=([\w-]+)',line)
            entries.append({'resource_path':match.group(1),'project':match.group(2),
                'source':source.group(1) if source else None})
    return entries


def propose(folder, packet, history, parent_code, previous, model='fable', effort='medium', timeout=600,
            research_directory=None, guide=GUIDE, max_turns=48, workspace_service=None,
            vertex_project=None, vertex_region='global', output_artifact='candidate.py',
            publish_session_history=True):
    folder=Path(folder); workspace=folder/'workspace';workspace.mkdir(parents=True,exist_ok=True)
    model=long_context_model(model)
    packet={**packet,'python_executable':'/usr/local/bin/python'}
    if output_artifact=='decision.json':
        packet={**packet,'required_output_path':str((workspace/output_artifact).resolve())}
        guide+='\nWrite the final decision to INPUT.json required_output_path, using the absolute path. Bash may retain a changed working directory.\n'
    design=packet.get('research_design','control')
    growing=design in ('growing','old-style','native-growth','placement')
    if growing:
        (workspace/'program').mkdir(exist_ok=True)
        (workspace/'eda').mkdir(exist_ok=True)
        (workspace/'notes').mkdir(exist_ok=True)
        if design=='growing':guide+=GROWING_GUIDE
        else:guide=OLD_STYLE_GUIDE+(NATIVE_GROWTH_GUIDE if design=='native-growth' else '')
        if design=='placement':guide+=PLACEMENT_GUIDE
        write(workspace/'CONTEXT-COMPARISON.json',packet['context_comparison'])
    # A real observed agent `git add -A` escaped to the enclosing shared repository.
    # Contain ordinary Git discovery without changing the project's worktree/index.
    if not (workspace/'.git').exists():
        subprocess.run(['git','init','--quiet',str(workspace.resolve())],check=True)
    if research_directory is not None:
        shutil.copytree(research_directory,workspace/'research-data',dirs_exist_ok=True)
    references=publish_method_library(workspace/'program' if growing else workspace)
    write(workspace/'INPUT.json',{**packet,'session_wall_limit_seconds':timeout,
        'agent_max_turns':max_turns,'editable_method_references':references})
    # Preserve the typed conversion from write(), but avoid thousands of lines
    # containing single numeric values in a code-agent Read response.
    compact=json.loads((workspace/'INPUT.json').read_text())
    (workspace/'INPUT.json').write_text(readable_json(compact)+'\n')
    index=[]
    for number,row in enumerate(history):
        item={k:row.get(k) for k in ('epoch','iteration','parent','status','research_incumbent','incumbent_SQL')}
        if number<len(previous):item['details']=f'history/{previous[number].name}/R-FEEDBACK.json'
        item['candidates']={name:{'SQL':value['SQL'],
            'folds':[{'offset':fold['offset'],'metrics':fold['metrics']} for fold in value['folds']]}
            for name,value in row.get('results',{}).items()}
        if row.get('traceback'):item['error_tail']=row['traceback'][-1500:]
        index.append(item)
    write(workspace/'history.json',index)
    (workspace/'parent.py').write_text(parent_code)
    (workspace/'API.md').write_text(API)
    (folder/'REQUEST.md').write_text(guide)
    for old in previous:
        target=workspace/'history'/old.name
        target.mkdir(parents=True,exist_ok=True)
        for name in ('candidate.py','R-FEEDBACK.json','HYPOTHESIS.md','REFLECTION.md','agent.stdout.jsonl'):
            if (old/name).exists(): shutil.copy2(old/name,target/name)
        if (old/'code').exists():shutil.copytree(old/'code',target/'code',dirs_exist_ok=True)
        for name in ('eda','notes'):
            if (old/name).exists():shutil.copytree(old/name,target/name,dirs_exist_ok=True)
    sessions=[]
    prior_sessions=sorted(folder.parent.parent.glob('*/agent-session')) if publish_session_history else []
    for prior in prior_sessions:
        if prior.resolve()==folder.resolve():continue
        archive_target=workspace/'research-history'/prior.parent.name
        for name in ('eda','notes','archive','research-data'):
            path=prior/'workspace'/name
            if path.exists():shutil.copytree(path,archive_target/name,dirs_exist_ok=True)
        archive_target.mkdir(parents=True,exist_ok=True)
        for name in ('HYPOTHESIS.md','REFLECTION.md'):
            path=prior/'workspace'/name
            if path.exists():shutil.copy2(path,archive_target/name)
        if (prior/'agent.stdout.jsonl').exists():
            shutil.copy2(prior/'agent.stdout.jsonl',archive_target/'agent.stdout.jsonl')
        sessions.append({'epoch':prior.parent.name,'notes':str(archive_target.relative_to(workspace)),
            'public_archive':str((archive_target/'archive').relative_to(workspace)),
            'research_data':str((archive_target/'research-data').relative_to(workspace)),
            'session_trace':str((archive_target/'agent.stdout.jsonl').relative_to(workspace))})
    if publish_session_history or not (workspace/'RESEARCH-HISTORY.json').exists():
        write(workspace/'RESEARCH-HISTORY.json',sessions)
    executable=os.environ.get('CLAUDE_CODE_BINARY') or shutil.which('claude')
    if not executable:raise RuntimeError('Claude Code not found; set CLAUDE_CODE_BINARY or PATH')
    route_options,route_env,route=vertex_session_route(folder,model,effort,vertex_project,vertex_region)
    command=[executable,'--bare','--print','--model',model,'--effort',effort,
        '--output-format','stream-json','--verbose','--max-turns',str(max_turns),
        '--permission-mode','acceptEdits','--tools','Read,Write,Edit,Bash',
        '--allowedTools','Read','Write','Edit','Bash',*route_options,'--',guide]
    start=time.monotonic();expired=False
    service=workspace_service(workspace) if workspace_service else nullcontext()
    with service,(folder/'agent.stdout.jsonl').open('w') as out,(folder/'agent.stderr.log').open('w') as err:
        agent_env={**os.environ,**route_env};agent_env['CUDA_VISIBLE_DEVICES']=''
        agent_env.pop('TIMERSI_U1_GPU_POOL',None)
        agent_env['GIT_CEILING_DIRECTORIES']=str(workspace.resolve().parent)
        route['effective_budget_environment']={key:agent_env.get(key) for key in
            ('CLAUDE_CODE_MAX_OUTPUT_TOKENS','MAX_THINKING_TOKENS','CLAUDE_CODE_MAX_RETRIES')}
        write(folder/'AGENT-TRANSPORT.json',{'model':model,'effort':effort,'max_turns':max_turns,
            'timeout_seconds':timeout,'command_without_prompt':command[:-1],
            'vertex_route':route,'tool_policy':'Read,Write,Edit,Bash; acceptEdits'})
        from .isolation import agent_launch
        with agent_launch(command,workspace,model,effort,vertex_project,vertex_region) as isolated:
            proc=subprocess.Popen(isolated,cwd=workspace,stdout=out,stderr=err,start_new_session=True)
            write(folder/'AGENT-PID.json',{'pid':proc.pid,'started_unix':time.time()})
            try:code=proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                expired=True;os.killpg(proc.pid,signal.SIGTERM)
                try:code=proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
    models=set();results=[];calls=[];initialization={}
    for line in (folder/'agent.stdout.jsonl').read_text().splitlines():
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        if row.get('type')=='assistant':
            msg=row.get('message',{})
            if msg.get('model'):models.add(msg['model'])
            calls.extend(b.get('name') for b in msg.get('content',[]) if b.get('type')=='tool_use')
        if row.get('type')=='result':results.append(row)
        if row.get('type')=='system' and row.get('subtype')=='init':
            initialization={key:row.get(key) for key in
                ('model','tools','skills','plugins','mcp_servers','permissionMode')}
    route['observed_requests']=observed_vertex_routes(route)
    route['observed_projects']=sorted({entry['project'] for entry in route['observed_requests']})
    entry=(workspace if output_artifact=='decision.json' else executable_root(workspace))/output_artifact
    draft=output_artifact=='candidate.py' and budget_draft_usable(code,expired,results,entry)
    receipt={'seconds':time.monotonic()-start,'exit_code':code,'timeout':expired,
        'observed_models':sorted(models),'tool_calls':calls,'results':results,'effort':effort,
        'executable':executable,'usable_budget_limited_code':draft,
        'configured_model':model,'vertex_route':route,'initialization':initialization}
    receipt['requested_context_window']=1000000 if model.endswith('[1m]') else None
    receipt['reported_context_windows']=sorted({v['contextWindow'] for r in results
        for v in r.get('modelUsage',{}).values() if v.get('contextWindow')})
    write(folder/'AGENT-RESULT.json',receipt)
    if expired or (code and not draft) or not models or not entry.exists():
        raise RuntimeError('No successful genuine agent artifact; see AGENT-RESULT.json')
    shutil.copy2(entry,folder/output_artifact)
    for name in ('HYPOTHESIS.md','REFLECTION.md'):
        if (workspace/name).exists():shutil.copy2(workspace/name,folder/name)
    runtime=executable_root(workspace)
    for source in runtime.rglob('*.py'):
        relative=source.relative_to(runtime)
        if relative.parts[0] in ('history','archive','.probe-queue','research-history','eda','notes') or '__pycache__' in relative.parts:continue
        target=folder/'code'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    return (folder/output_artifact).read_text(),receipt
