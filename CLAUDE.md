# AI notetaker

Meeting intelligence pipeline (live captions + post-meeting transcript, diarization,
speaker ID and cited insights). Design: `Meeting-Intelligence-Pipeline-Design.pdf`.

## Agent skills

### Issue tracker

Issues live as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix), recorded as a `Status:` line in each issue file. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the project root. See `docs/agents/domain.md`.
