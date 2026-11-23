# SOP: how to use this folder

Read this first. It tells you how to load context, what to write back, and how to keep the folder small.

## Purpose
`context/` holds what a new session cannot cheaply rediscover: gotchas, failed approaches, open problems and working
preferences. It is not a copy of the code, `docs/`, or the published pages. If the code or a page already says it, link there.

## Loading (every session)
1. Read this file, then `README.md` (the index).
2. Read `shared/workflow.md` (how to work with the user).
3. Work out which project the task belongs to (folder names under `projects/` match the page slug).
4. Read that project's `README.md`, then only the project files the task needs.
5. Read only the `shared/` files the task touches (simulation work: `mujoco.md`; Panda motion: `robots/franka_panda.md`;
   cloud or model runs: `infra.md`; the web page or deploy: `publishing.md`).
6. Never load another project's folder. Projects are segregated on purpose.

## Trust order
Code and the running system win over these notes. If a note names a file, function or number, check it still exists
before relying on it. If a note is wrong, fix or delete it in the same session.

## What to store
- A finding that cost real time to discover (more than about 15 minutes), with the cause, not only the symptom.
- A failed approach and why it failed, so it is not retried.
- An open problem with the best current lead and where any unfinished work lives (for example a git stash).
- A working preference the user stated, with the reason.
- A status line per task: measured result, how many seeds, date.

## What not to store
- Anything already in code, `docs/`, or the page: constants, specs, commands that already have a home. Link instead.
- Secrets, tokens, or the `keys/` folder contents. Never reference its contents.
- The user's name, email or personal details.
- Timelines or session narratives (the git log has them).
- Unmeasured claims. Mark anything unverified as `(unverified)`.

## Writing rules
- Date every fact or status block (`YYYY-MM-DD`). Put `Last verified:` at the top of each file.
- One topic per file. Keep each file under about 150 lines; split it when it grows.
- Lead with the rule or finding, then the reason, then how to apply it.
- Prefer plain bullets. No marketing language.

## Placing a finding
- Would it still be true on a different project? It goes in `shared/`.
- Is it about one benchmark? It goes in that project's folder.
- Move a project finding to `shared/` only when a second project needs it.
- Files in one project never link to another project's files; they may link to `shared/`.

## Updating (do this before ending a session)
1. New finding, failed approach or open problem: add it to the right file.
2. Status changed: update that task's dated line in the project `README.md`.
3. A note turned out wrong or stale: fix or delete it.
4. New file added: add one line to `README.md`.
5. Re-read the diff of `context/` once; remove anything that repeats the code or the page.

## Adding a new project
1. Copy `projects/_template/` to `projects/<page_slug>/`.
2. Fill in its `README.md` (what it is, where the code and page are, dated status).
3. Add one line for it to the project list in `README.md`.
4. Do not edit other projects' folders. Promote shared findings per the rule above.

## Before finishing any task
Ask: did I learn something the next session would waste time on? If yes, it is not done until it is written here.
