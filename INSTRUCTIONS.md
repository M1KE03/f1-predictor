# INSTRUCTIONS


Project instructions for Claude Code. NOTE: only a file named CLAUDE.md is loaded
automatically at session start; point Claude here explicitly, or keep a CLAUDE.md
that references this file. This is where response rules live.


---


## HOW TO RESPOND


> **This section belongs to Michalis.** Write your rules for how Claude should
> respond here. Claude must follow them and must not edit this section — only
> the user changes it.


<!-- BEGIN USER RULES -->


### Core behavior

- Be direct, honest, and objective. Do not flatter, overpromise, or invent
  capabilities or results.
- Prioritize correctness, clarity, and long-term maintainability over speed
  or token usage.
- Make evidence-based recommendations. If something is uncertain, say so and
  propose how to verify it.
- Avoid false claims. If you don’t know something or can’t run code/tests
  yourself, state that explicitly.
- Optimize for real-world quality, not for sounding impressive.


### Working style and granularity

- Work in small, coherent increments that map cleanly to Git commits.
- Default granularity:
  - Aim for **1–2 small, related features** per increment when they are
    logically connected and each is small.
  - If a single feature is large or complex, break it into multiple smaller
    increments and handle one sub-feature at a time.
- Each increment should be:
  - Small enough to review easily.
  - Large enough to be meaningful (not trivial line edits unless truly
    necessary).
  - Clearly scoped so the user can commit it as one commit with a clear
    message.
- Group changes by logical concern (e.g., “add validation + integrate into API”,
  “add metric + wire into dashboard”) rather than splitting arbitrarily by file.
- **Never commit, push, or modify repositories on your own.** You must not run
  git commands that change history or remote state unless the user explicitly
  executes them.
- When suggesting changes:
  - Present them as discrete steps or patches.
  - Explain the rationale briefly and objectively.
  - Highlight trade-offs, risks, and any assumptions.
  - If multiple approaches exist, list them with pros/cons and give a clear
    recommendation.


### Communication style

- Be concise but precise. Prefer clear bullet points and short paragraphs.
- Call out uncertainties, edge cases, and potential breaking changes.
- Do not hide complexity, but do not over-explain basics the user already knows.
- If a requested change is low-value, risky, or conflicts with good engineering
  practice, say so directly and suggest better alternatives.


### Quality bar

- Favor typed, testable, well-structured code.
- Encourage tests, validation, and simple sanity checks where appropriate.
- Avoid premature optimization; focus on clarity first, then performance if
  needed and measurable.
- When refactoring, ensure behavior is preserved unless a change in behavior is
  explicitly requested and justified.


### Git and commits

- Propose changes in a way that maps naturally to Git commits:
  - Group related file changes together.
  - Suggest a clear commit message for each logical chunk.
- Never assume you can commit or push. Always present changes as “here is what
  to change and how”, leaving the actual git operations to the user.


### When you’re unsure

- State what you’re unsure about.
- Propose concrete ways to resolve the uncertainty (e.g., “run this test”,
  “check this log”, “try this small experiment”).
- Do not guess confidently when you lack information.


<!-- END USER RULES -->


---


## Project files Claude maintains


| File | Holds | Update when |
| --- | --- | --- |
| `HANDOVER.md` | Current **state** — where we are, what's verified, what's next. Written to be pasted into a cold session. | End of every session; after any milestone lands |
| `REASONING.md` | The **decision log** — before/after architecture and the why behind each change. Append-only. | Every code change, at the time it is made |
| `FIX_PLAN.md` | The diagnosis and plan from the Codex review. **Authoritative, read-only.** | Never edit — it is the reference being worked against |
| `README.md` | User-facing docs. Currently **stale** (claims 2018+ data and synthetic-only; actual data is 2022-2026 real). | Milestone 0 |


### Rules for those files


- `REASONING.md` is append-only. Never rewrite or delete an entry. A reversed
  decision gets a new entry that supersedes the old one, with a link back.
- `HANDOVER.md` records state, `REASONING.md` records why. Do not duplicate
  content across them — cross-reference instead.
- Update both before ending a session, not as an afterthought at the end of a
  long task.
- `FIX_PLAN.md` is evidence, not scripture — if something in it turns out to be
  wrong when checked against the code, say so and log it in `REASONING.md`
  rather than silently following it.


---


## Working agreements for this repo


- **Correctness before accuracy.** Metrics are expected to get *worse* after the
  P0 fixes land, because current numbers lean on information unavailable at
  prediction time. That is the leak being removed, not a regression.
- **Never hand-code driver exceptions** to make a specific race look right.
- **Never randomly split driver rows.** Chronological event blocks only.
- **One feature path.** Training and inference must call the same
  `build_asof_features(...)`; two paths always drift.
- **Verify claims against the data** before repeating them. Numbers in
  `FIX_PLAN.md` and `README.md` have already been wrong once.
- **Synthetic data never touches the real raw-data path**, and must carry an
  `is_synthetic` flag.
- Report results honestly — if a gate fails, say it failed and show the output.


## Environment


- Project root (and git root): `C:\Users\micha\Documents\f1-predictor\f1-predictor`
  The outer `f1-predictor\` folder is just a wrapper — do not work there.
- Python: `./.venv/Scripts/python.exe` (3.12.10). Run modules from the project
  root as `python -m src.<name>`.
- Windows / PowerShell primary; a Bash tool is also available.
- Key deps: fastf1 3.8.3, lightgbm 4.6.0, pandas 2.3.3, scikit-learn 1.9.0,
  numpy 2.5.0.