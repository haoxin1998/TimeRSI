"""Editable sparse-event expert, without region or named-dataset dispatch.

The agent supplies a calendar_builder(item): EventCalendar, constructed from
provided known fields and/or an explicitly authored date calendar. This module
learns level × cyclic phase × event effects using only each origin's target past.
It supports hierarchical event shrinkage, a known event-factor model channel,
optional context repair, credit for event movement already in the native output,
a statistical specialist blend, and asymmetric quantile-shape calibration.
"""
from dataclasses import dataclass, field, replace
from typing import Mapping
import numpy as np
import pandas as pd
from timersi_u1.core import Forecast


@dataclass
class EventCalendar:
    # Empty event_type means an ordinary observation. names/groups/phase are
    # supplied semantic distinctions, e.g. named event, business day, hour/day.
    event_type: np.ndarray
    event_name: np.ndarray | None = None
    event_group: np.ndarray | None = None
    phase: np.ndarray | None = None

    def arrays(self, timestamps):
        n = len(timestamps)
        kinds = np.asarray(self.event_type).astype(str)
        names = kinds if self.event_name is None else np.asarray(self.event_name).astype(str)
        groups = np.full(n, "all") if self.event_group is None else np.asarray(self.event_group).astype(str)
        phase = (np.asarray(pd.DatetimeIndex(timestamps).dayofweek).astype(str)
                 if self.phase is None else np.asarray(self.phase).astype(str))
        if any(array.shape != (n,) for array in (kinds, names, groups, phase)):
            raise ValueError("Event calendar arrays must span context + horizon")
        return kinds, names, groups, phase


@dataclass(frozen=True)
class EventConfig:
    level_window: int = 29
    decomposition_iterations: int = 3
    specialist_level_window: int = 21
    group_prior_weight: float = 2.0
    class_prior_weight: float = 1.5
    name_prior_weight: float = 2.0
    # Optional explicit beliefs about event classes; empty by default.
    class_parent_scale: Mapping[str, float] = field(default_factory=dict)
    minimum_events: int = 3
    event_scale: float = 1.0
    inject_calendar_channel: bool = True
    clean_context: bool = False
    repair_log_lower: float = -1.2
    repair_log_upper: float = 0.5
    gap_credit: float = 1.0
    gap_window: int = 9
    specialist_weight: float = 0.3
    lower_spread: float = 1.0
    upper_spread: float = 1.0
    point_log_shift: float = 0.0


def events_from_known(item, column, normal_value=None, event_group=None, phase=None):
    """Bind an explicitly chosen known categorical event field, not its name.

    If normal_value is omitted, use its historical modal value as the proposed
    ordinary state. This assumption is itself testable and may be overridden.
    Unseen future category names shrink toward the generic learned event effect.
    """
    source = item.get("known_raw", {})
    values = np.asarray(source[column] if column in source else item["known"][column]).astype(str)
    length = item["y"].shape[1]
    if normal_value is None:
        levels, counts = np.unique(values[:length], return_counts=True)
        normal_value = levels[np.argmax(counts)]
    types = np.where(values == str(normal_value), "", "event")
    return EventCalendar(types, values, event_group, phase)


def decompose(history, phase, event_type, config, observed=None):
    """Robust ordinary-state level and cycle; event dates do not set that level.

    Centred rolling medians run only within history supplied at this origin;
    no forecasting-horizon target is present in this computation.
    """
    history = np.asarray(history, float)
    valid = np.isfinite(history) & (history > 0)
    if observed is not None:
        valid &= np.asarray(observed, bool)
    normal = (event_type == "") & valid
    if np.sum(normal) < 8:
        return None
    log_y = np.full(len(history), np.nan)
    log_y[valid] = np.log(history[valid])
    phase_values = {value: 0.0 for value in np.unique(phase)}
    for _ in range(config.decomposition_iterations):
        cycle = np.array([phase_values[value] for value in phase])
        adjusted = pd.Series(log_y - cycle).where(normal)
        level = adjusted.rolling(config.level_window, center=True,
                                 min_periods=max(3, config.level_window // 4)).median()
        level = level.interpolate(limit_direction="both").to_numpy()
        if not np.isfinite(level).all():
            level = np.full(len(history), np.nanmedian(log_y[normal]))
        residual = log_y - level
        for value in phase_values:
            values = residual[normal & (phase == value)]
            phase_values[value] = float(np.median(values)) if len(values) else 0.0
        centre = float(np.mean(list(phase_values.values())))
        phase_values = {key: value - centre for key, value in phase_values.items()}
    cycle = np.array([phase_values[value] for value in phase])
    # Re-estimate the level once after the final cycle update.
    adjusted = pd.Series(log_y - cycle).where(normal)
    level = adjusted.rolling(config.level_window, center=True,
                             min_periods=max(3, config.level_window // 4)).median()
    level = level.interpolate(limit_direction="both").to_numpy()
    if not np.isfinite(level).all():
        level = np.full(len(history), np.nanmedian(log_y[normal]))
    return {"level": level, "phase": phase_values,
            "residual": log_y - level - cycle, "valid": valid}


def fit_event_effects(residual, event_type, event_name, event_group, config):
    """global event → group → event class → named event shrinkage.

    Every empirical term is estimated from historical event residuals. Sparse or
    unseen names inherit the broader class/group, rather than a task-specific
    lookup. Optional class_parent_scale expresses explicit, editable knowledge.
    """
    event = (event_type != "") & np.isfinite(residual)
    total = int(event.sum())
    global_effect = float(np.median(residual[event])) if total else 0.0
    groups = {}; classes = {}; names = {}; support = {}
    for group in np.unique(event_group[event]):
        group_mask = event & (event_group == group)
        n = int(group_mask.sum())
        group_effect = ((n * float(np.median(residual[group_mask])) + config.group_prior_weight * global_effect)
                        / (n + config.group_prior_weight))
        groups[group] = group_effect
        for kind in np.unique(event_type[group_mask]):
            class_mask = group_mask & (event_type == kind)
            n_class = int(class_mask.sum())
            parent = group_effect * config.class_parent_scale.get(kind, 1.0)
            class_effect = ((n_class * float(np.median(residual[class_mask])) + config.class_prior_weight * parent)
                            / (n_class + config.class_prior_weight))
            classes[(group, kind)] = class_effect
            for name in np.unique(event_name[class_mask]):
                name_mask = class_mask & (event_name == name)
                count = int(name_mask.sum())
                names[(group, kind, name)] = ((count * float(np.median(residual[name_mask]))
                                              + config.name_prior_weight * class_effect)
                                             / (count + config.name_prior_weight))
                support[(group, kind, name)] = count
    return {"global": global_effect, "groups": groups, "classes": classes,
            "names": names, "support": support, "total_events": total}


def event_adjustment(effects, event_type, event_name, event_group, config):
    fade = min(1.0, effects["total_events"] / max(config.minimum_events, 1))
    out = np.zeros(len(event_type))
    for j, (kind, name, group) in enumerate(zip(event_type, event_name, event_group)):
        if not kind:
            continue
        fallback = effects["groups"].get(group, effects["global"]) * config.class_parent_scale.get(kind, 1.0)
        effect = effects["names"].get((group, kind, name), effects["classes"].get((group, kind), fallback))
        out[j] = config.event_scale * fade * effect
    return out


def fit_event_item(item, calendar, config=EventConfig()):
    length = item["y"].shape[1]
    types, names, groups, phase = calendar.arrays(item["timestamps"])
    fits = []
    for d, target in enumerate(item["y"]):
        observed = item.get("observed")
        dec = decompose(target, phase[:length], types[:length], config,
                        None if observed is None else observed[d])
        if dec is None:
            fits.append(None)
            continue
        effects = fit_event_effects(dec["residual"], types[:length], names[:length], groups[:length], config)
        adjustment = event_adjustment(effects, types, names, groups, config)
        phase_future = np.array([dec["phase"].get(value, 0.0) for value in phase[length:]])
        current_level = float(np.median(dec["level"][-config.specialist_level_window:]))
        specialist_log = current_level + phase_future + adjustment[length:]
        cleaned = np.asarray(target, float).copy()
        if config.clean_context:
            event = (types[:length] != "") & dec["valid"] & np.isfinite(dec["residual"])
            cleaned[event] *= np.exp(-np.clip(dec["residual"][event], config.repair_log_lower, config.repair_log_upper))
        fits.append({"adjustment": adjustment, "phase_future": phase_future,
                     "specialist_log": specialist_log, "cleaned": cleaned,
                     "effects": effects, "decomposition": dec})
    return fits


def uncredited_adjustment(point, phase_effect, event_log_effect, credit=1.0, window=9):
    """Avoid adding an event movement already present in the native trajectory.

    This is a testable shape heuristic, not a guarantee: a long event spanning
    most of the horizon may be absorbed by the rolling baseline. No labels are
    used. Agent may change the baseline, turn credit off, or replace this method.
    """
    log_base = np.log(np.maximum(point, 1e-9)) - phase_effect
    smooth = pd.Series(log_base).rolling(window, center=True, min_periods=1).median().to_numpy()
    movement = log_base - smooth
    present = abs(event_log_effect) > 1e-9
    fraction = np.zeros_like(movement)
    fraction[present] = np.clip(movement[present] / event_log_effect[present], 0, 1)
    return event_log_effect * (1 - credit * fraction)


def predict_events(view, tools, memory, calendar_builder, config=EventConfig()):
    """Full editable event program, including a real native model call.

    calendar_builder is supplied by candidate code. Never retrieve it through a
    task-name registry: write it from observed field/date semantics and test it.
    """
    if not (0 <= config.specialist_weight <= 1 and 0 <= config.gap_credit <= 1):
        raise ValueError("Blend/credit weights must lie in [0, 1]")
    fits = [fit_event_item(item, calendar_builder(item), config) for item in view.items]
    native_items = []
    for item, item_fits in zip(view.items, fits):
        known = dict(item["known"])
        usable = [fit for fit in item_fits if fit is not None]
        if config.inject_calendar_channel and usable:
            known["agent_event_factor"] = np.mean([np.exp(fit["adjustment"]) for fit in usable], axis=0)
        target = np.stack([fit["cleaned"] if fit is not None else item["y"][d]
                           for d, fit in enumerate(item_fits)])
        native_items.append({**item, "y": target, "known": known})
    native = tools.timesfm(replace(view, items=native_items))
    point = native.point.copy(); quantiles = native.quantiles.copy()
    for i, item_fits in enumerate(fits):
        length = view.items[i]["y"].shape[1]
        for d, fit in enumerate(item_fits):
            if fit is None:
                continue
            log_event = uncredited_adjustment(native.point[i, :, d], fit["phase_future"],
                                               fit["adjustment"][length:], config.gap_credit, config.gap_window)
            log_native = np.log(np.maximum(native.point[i, :, d], 1e-9)) + log_event
            log_point = ((1 - config.specialist_weight) * log_native
                         + config.specialist_weight * fit["specialist_log"] + config.point_log_shift)
            final_point = np.exp(log_point)
            median = np.maximum(native.quantiles[i, :, d, 4], 1e-9)
            offset = np.log(np.maximum(native.quantiles[i, :, d], 1e-9) / median[:, None])
            shape = np.where(offset < 0, config.lower_spread * offset, config.upper_spread * offset)
            shaped = median[:, None] * np.exp(shape)
            final_quantiles = shaped * (final_point / np.maximum(native.point[i, :, d], 1e-9))[:, None]
            point[i, :, d] = final_point; quantiles[i, :, d] = np.sort(final_quantiles, axis=-1)
    return Forecast(point, quantiles).validate()
