"""Complete sparse-event candidate, with editable semantic calendar construction.

Set EVENT_COLUMN only after inspecting a supplied known event field, or replace
calendar() with a justified region/date calendar written in this candidate. No
region is assumed by this example. An empty calendar tests just the statistical
cycle/level expert; it does not invent event labels.
"""
import numpy as np
from .event_specialist import (
    EventCalendar, EventConfig, events_from_known, predict_events,
)

EVENT_COLUMN = None
CONFIG = EventConfig()


def calendar(item):
    if EVENT_COLUMN is not None:
        return events_from_known(item, EVENT_COLUMN)
    return EventCalendar(np.full(len(item["timestamps"]), ""))


def predict(view, tools, memory):
    return predict_events(view, tools, memory, calendar, CONFIG)
