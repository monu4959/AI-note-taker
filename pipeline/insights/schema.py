"""Insight schemas (from the design doc). Every item cites the utterance IDs it came from."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 2   # v2: per-topic narrative (key points as prose) and Q&A


class Evidence(BaseModel):
    utterance_ids: list[str] = Field(description='Supporting utterance IDs, e.g. ["u0142", "u0144"]')


class Item(BaseModel):
    """A key point, decision or open question."""
    text: str
    evidence: Evidence


class ActionItem(BaseModel):
    task: str
    owner: str | None = Field(description="Speaker label or a name said in the meeting, only if "
                                          "someone committed or was assigned; never guessed")
    due: str | None = Field(description='As said, e.g. "by Friday"; null if not said')
    evidence: Evidence


class QA(BaseModel):
    """A question asked in the meeting and what was said in reply."""
    question: str = Field(description="The question, rephrased to stand on its own")
    asked_by: str | None = Field(description="Speaker label or name of who asked; null if unclear")
    answer: str | None = Field(description="What was said in reply, as a short summary; null if "
                                           "nobody answered")
    answered_by: str | None = Field(description="Speaker label or name of who answered; null if "
                                                "unanswered or unclear")
    status: Literal["answered", "partial", "unanswered"] = Field(
        description="answered: fully; partial: only in part or deferred; unanswered: no reply")
    evidence: Evidence = Field(description="The question's utterance ID first, then the answer's")


class Metric(BaseModel):
    name: str = Field(description='What is measured, e.g. "Q3 churn"')
    value: str = Field(description='As said, e.g. "4.2"')
    unit: str | None = Field(description='e.g. "%", "euros"; null if not said')
    context: str | None = Field(description='e.g. "down from 5.1"')
    evidence: Evidence


# --- call 1: segmentation ---
class TopicSpan(BaseModel):
    title: str
    start_id: str
    end_id: str


class Segmentation(BaseModel):
    topics: list[TopicSpan]


# --- call 2: per-topic extraction ---
class TopicExtraction(BaseModel):
    key_points: list[Item] = Field(description="2-5 sentences, in order, that together tell what was "
                                               "discussed in this topic; each cites its own lines")
    qa: list[QA]
    decisions: list[Item]
    action_items: list[ActionItem]
    metrics: list[Metric]
    open_questions: list[Item]


# --- call 3: summary ---
class Summary(BaseModel):
    summary: str = Field(description="3-5 sentence overview")


# --- single-pass alternative (one call does everything) ---
class Topic(BaseModel):
    title: str
    start_id: str
    end_id: str
    key_points: list[Item]


class MeetingInsights(BaseModel):
    summary: str
    topics: list[Topic]
    decisions: list[Item]
    action_items: list[ActionItem]
    metrics: list[Metric]
    open_questions: list[Item]
    qa: list[QA]


# --- call 4: semantic verification ---
class Verdict(BaseModel):
    item_id: str
    support: Literal["supported", "partial", "unsupported"]
    reason: str


class Verification(BaseModel):
    verdicts: list[Verdict]
