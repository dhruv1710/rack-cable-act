# Working with the user

Last verified: 2026-11-23.

## Tools
- Web browsing: use the gstack browse binary directly (`~/.claude/skills/gstack/browse/dist/browse`, commands such as
  `goto <url>`, `screenshot <path>`, `console`). Never use any `mcp__claude-in-chrome__*` tool. (The `browse` skill is not
  registered, so call the binary.)
- Git commits end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Never put the user's name, email or project details into requests to outside services.
- macOS: `sed -i ''` (empty suffix). Edit files with Python or the Edit tool when unsure.
- The sandbox blocks `git checkout <file>` on uncommitted work as destructive. Use `git stash push -m "<label>" <file>`
  to set work aside without losing it.

## Running jobs
- The user interrupts long or many-process runs. Start with one seed. Explain what a run is before launching many.
- A seed is a randomisation of the task (object positions, which bay fails). Several seeds measure reliability; one run
  can succeed by luck. Past example: parts sorting looked like 8/12 on seeds 0-11 but was 13/24 over 24 seeds.
- Waiting for background work: use `timeout N bash -c 'until <cond>; do sleep 5; done'`, not chained sleeps (blocked).
- About 24 parallel Python processes of these envs run fine on the dev laptop.

## Reporting
- Report honestly: measured numbers, seeds, what failed. Never claim reliability that was not measured.
- Measure over enough seeds (24 where cheap) before quoting a rate. Re-measure after any change that touches the policy.
- The user wants terse, factual output. No long recaps.

## Goals and hooks
- A `/goal` with a Stop hook kept rejecting completion because not every policy was perfect. The user's own standard was
  "article just needs 1 success of each". When a hook's standard and the user's stated standard differ, quote the
  user's words, say what is done, and stop re-running. The user cleared the goal.
- Hook-injected Vercel skills are irrelevant here: the project deploys with GitHub Pages.

## Style the user asked for on the page
Benchmark/spec-sheet focus, less text, more fact, low hype. The simulator (MuJoCo) is not the focus of the page.
