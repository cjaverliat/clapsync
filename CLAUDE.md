# CLAUDE.md

Guidelines to reduce common LLM coding mistakes. Bias toward caution over speed; use judgment on trivial tasks.

## Environment: pixi

This project uses [pixi](https://pixi.sh) — run everything through it. No bare `python`, `pip`, or `conda`.

- Run Python: `pixi run python X.py`
- Deps (keeps `pixi.lock` synced): `pixi add <pkg>`, `pixi add --pypi <pkg>`, `pixi remove <pkg>`
- Defined tasks: `pixi run <task>` (see `[tasks]` in `pixi.toml`)
- Interactive shell: `pixi shell`

## Think before coding

- State assumptions; if uncertain, ask.
- Surface multiple interpretations instead of picking one silently.
- Propose a simpler approach when one exists; push back when warranted.
- If something is unclear, stop and name it.

## Surgical changes

Touch only what the request needs.

- Don't refactor, reformat, or "improve" working code nearby.
- Match existing style.
- Note unrelated dead code; don't delete it.
- Remove only orphans your own changes created.

Test: every changed line traces to the request.

## Goal-driven execution

Turn tasks into verifiable goals, then loop until met.

- "Fix the bug" → write a failing test that reproduces it, then make it pass.
- "Refactor X" → tests green before and after.
- Multi-step work: state a brief plan with a verify check per step.

## SOLID (when designing classes/modules)

SOLID is a refactoring vocabulary, not an up-front mandate. **When it conflicts with KISS/YAGNI, simplicity wins.** Apply a principle in response to actual pain (a class that's hard to change or test), not preemptively.

Don't add an abstraction until a **second real implementation exists now** — name it, or don't create the abstraction. Default to the concrete version; introduce structure when duplication or a real variation appears.

- **SRP** — one reason to change per class; separate concerns.
- **OCP** — extend via new implementations of a contract; don't modify existing code.
- **LSP** — subtypes fully substitutable for their base; no weakened behavior or surprise exceptions.
- **ISP** — small, focused interfaces; clients depend only on what they use.
- **DIP** — depend on abstractions; inject implementations rather than hardcoding them.

Python specifics:
- Prefer a plain function over a class until there's real state.
- Rely on duck typing; don't declare a `Protocol`/ABC for a single implementation.
- Avoid SOLID boilerplate when avoidable — much of it solves Java problems Python doesn't have.

Banned tells (signs of over-abstraction): an interface/`Protocol`/ABC with one implementer; a factory that builds one type; a `Manager`/`Handler`/`Service` that only forwards calls; dependency injection where a default argument works; a config system with one config. Self-check: if new types/files outweigh actual behavior added, simplify.

## DRY

Extract real repetition into shared functions/classes; centralize common operations. Don't deduplicate speculatively.

## KISS

Simplest solution that works: clear names, built-ins over custom machinery, minimal dependencies, small units. If 200 lines could be 50, rewrite.

## YAGNI

Build for current requirements, not hypothetical ones. No speculative abstractions, config, or flexibility; no premature optimization. Wait for a real use case.

---

**Working if:** cleaner diffs, fewer overcomplication rewrites, and questions before implementation. Good code is simple, clear, and purposeful.