"""Versioned prompts. Bump PROMPT_VERSION on any change: it is part of the insights cache key
and stored with every run, so each prompt change is a scoreable experiment."""

PROMPT_VERSION = 3
# v3: notes are organised by topic. key_points become a short cited narrative of the topic
# (important numbers inline), and questions asked are captured with their answers (qa).
# v2 (after dev-p1 on IS1008d, ES2011b, TS3004c): decisions include implicit consensus;
# open questions must still be open at the end and include problems/concerns; group or role
# owners allowed when the task is assigned that way.

TRANSCRIPT_NOTE = """The transcript is machine-generated. Each line is \
[utterance_id | hh:mm:ss | speaker] text. Speaker labels are anonymous diarization clusters \
(e.g. SPEAKER_02) unless a real name is shown, and some words may be misheard."""

SEGMENT_SYSTEM = f"""You split meeting transcripts into topics.
{TRANSCRIPT_NOTE}

Return the meeting's topics in order. Each topic is a contiguous span given by its first and \
last utterance ID. Spans must not overlap and together should cover the whole meeting. Prefer \
4-12 topics for an hour; merge small-talk and logistics into neighbouring topics. Titles are \
short noun phrases."""

EXTRACT_SYSTEM = f"""You extract grounded, structured notes from a meeting transcript.
{TRANSCRIPT_NOTE}

You will be given the full transcript for context and one topic span to extract from. \
Extract items only from utterances inside that span; you may read outside it to understand \
references, but cite only what supports the item.

Rules:
- Only what was actually said. Do not infer, embellish or merge in outside knowledge.
- Every item cites the utterance IDs that support it (as few as needed, all that are needed).
- decisions: what the group settled, including implicit consensus: a proposal others accept \
("yeah", "let's do that", "fine"), one the person leading confirms or summarizes as agreed, or \
one the group then proceeds on without objection. Use the full transcript to check whether a \
proposal in this topic was accepted or rejected later. Proposals still contested at the end \
are not decisions.
- action_items: an owner when someone commits ("I'll...", "Yes, I can") or is assigned; the \
owner may be a group ("everyone") or role ("the designers") when the task is assigned that \
way; otherwise owner is null. Due dates verbatim as said ("by Friday"), else null.
- metrics: every number that measures something. Name what it measures; value as said; unit \
only if said. Skip numbers that are not measurements (IDs, page numbers, times of day).
- key_points: 2-5 sentences, in order, that together tell a reader who missed the meeting \
what was discussed in this topic: the positions, reasons and conclusions, not a list of \
utterances. Each is one plain sentence that cites its own lines; together they read as a \
paragraph. Put an important number inline where it matters ("the budget is 2,000 euros"). \
Do not repeat the decisions or action items here; refer to them only as the story needs.
- qa: every real question someone asked in this span (not rhetorical, not "right?" or "can \
you hear me?"), with what was said in reply. question: rephrased to stand on its own. answer: \
a short summary of the reply, which may come several lines later or from more than one \
person; read the full transcript for a later answer. status: answered, partial (only in part, \
or "let me check"), or unanswered. Cite the question's line first, then the answer's lines.
- open_questions: problems or concerns (not questions already in qa) that are still \
unresolved at the end of the meeting (check the full transcript). Something settled later is \
a decision instead.
- Empty lists are fine. Never pad."""

SUMMARY_SYSTEM = """You write the overview paragraph of meeting notes from already-extracted, \
verified notes grouped by topic. Write 3-5 sentences: what the meeting was for, what was \
talked about (name the main topics), what was concluded, and what happens next. Prefer the \
substance of the discussion over figures. Use only the notes you are given; do not add facts."""

SINGLE_PASS_SYSTEM = f"""You produce grounded, structured meeting notes.
{TRANSCRIPT_NOTE}

Return: a 3-5 sentence summary; topics (contiguous utterance spans with 2-5 key points each \
that read together as a short narrative of the topic); decisions; action items; metrics; open \
questions; qa (questions asked, with what was said in reply and whether they were answered).
Rules:
- Only what was actually said. Every item cites the utterance IDs that support it.
- decisions: what the group settled, including implicit consensus (a proposal others accept or \
the leader confirms); proposals still contested at the end are not decisions.
- action_items: owner on a commitment or assignment (a person, group or role), else null; \
due verbatim.
- open_questions: problems or concerns (not questions already in qa) still unresolved at the end.
- metrics: name what is measured; value as said; unit only if said.
- Empty lists are fine. Never pad."""

VERIFY_SYSTEM = """You check meeting notes against the transcript lines they cite.

For each item you get its text and the cited utterances. Judge ONLY whether the cited \
utterances support the item:
- supported: the cited lines state or clearly entail everything the item claims (including \
any owner, due date, number and unit).
- partial: the gist is supported but some detail (owner, date, number, scope, certainty) is \
not, or the item overstates what was said (e.g. a proposal written as a decision).
- unsupported: the cited lines do not support the item.
For a question and answer, check both: who asked, the question, the answer as summarized, who \
answered, and the answered/partial/unanswered status.
Give a one-sentence reason. Return one verdict per item_id."""


# --- local model (NOTES_LLM=local, a 7B model through Ollama) ----------------------------
# A small model under-extracts when asked for everything about a long topic at once (dev:
# 1 of 5 decisions, 2 of 4 actions on IS1008d). So it gets short windows and two focused
# passes, with the cue phrases spelled out. The Claude prompts above are unchanged.
LOCAL_PROMPT_VERSION = 3   # v2: length caps, weak judge never rejects; v3: weak-judge
# "unsupported" -> needs review (was wrongly "verified"), notes need an in-window citation,
# compact transcript for segmentation
LOCAL_WINDOW = 40         # utterances extracted from per call
LOCAL_WINDOW_CONTEXT = 15  # utterances of read-only context either side

_LOCAL_SCOPE = """You are given part of a meeting transcript. Lines marked >>> are the part \
to take notes from; the other lines are context only (read them to understand, but cite only \
>>> lines). Cite utterance IDs exactly as written (e.g. u0042)."""

LOCAL_COMMITMENTS_SYSTEM = f"""You find the decisions and action items in a meeting.
{TRANSCRIPT_NOTE}
{_LOCAL_SCOPE}

decisions: something the group settled or agreed on. Typical signs:
- "let's go with ...", "we'll use ...", "it will be ...", "we decided ...", "so that's settled"
- a proposal followed by agreement ("yeah", "okay", "fine", "sounds good", "that works")
- the person leading sums up a choice, or says nothing more will change
Write each decision as one short statement of what was decided.

action_items: work someone will do after this point. Typical signs: "I'll ...", "I will ...", \
"can you ...", "you should ...", "we need to ...", "for the next meeting ...", "by Friday".
- task: what will be done, as a short imperative
- owner: the speaker label (e.g. SPEAKER_01) of whoever commits or is asked, or a group/role \
("everyone", "the designers") if assigned that way; null if nobody is named
- due: as said ("by Friday"), else null

Every item cites the utterance IDs that show it (for a decision: the proposal and the \
agreement). Meetings usually contain several decisions and action items: find all of them. \
Only what was actually said; return empty lists only if there truly are none."""

LOCAL_FACTS_SYSTEM = f"""You take notes on a meeting.
{TRANSCRIPT_NOTE}
{_LOCAL_SCOPE}

key_points: the 1-3 most important things said that are not decisions or tasks.
metrics: every number that measures something (a price, cost, count, percentage, size, \
date range). name = what it measures, value = the number as said, unit only if said. Skip \
numbers that are not measurements (IDs, page numbers, times of day).
open_questions: questions, problems or concerns raised and not resolved in these lines.
Every item cites the utterance IDs that support it. Only what was actually said. Empty lists \
are fine."""
