# Publishing: pages, videos, deploy

Last verified: 2026-11-23. Site: https://dhruv1710.github.io/rack-cable-act/ (repo `dhruv1710/rack-cable-act`).

## Style (copy of the owner's site)
Newsreader serif body (18px/1.65) and JetBrains Mono for headings, code and table numbers; a 160px sticky left sidebar
with a mono name link and section anchors; a content column up to 960px page width; `figure`/`figcaption`; plain tables
with thin row rules; an H1 with a blinking `_` cursor; a footer with the copyright line already on the page. Mobile: sidebar
becomes inline links under 760px. Reuse the CSS already in `public/envs/index.html`; do not restyle.
Content rule: spec-sheet facts, short captions, failures listed plainly.

## Hand edits to keep (do not overwrite)
In `public/envs/index.html`: the title text, the three summary bullets (including the IsaacSIM line), and the sidebar
"Also check out" block.

## Video pipeline
- Record: `.venv/bin/python -c "import envs.record_scripted as r; r.run('<task>', 1, seeds=[<seed>])"` writes
  `out/scripted/<task>_seed<N>_ok.mp4` (use `record_all=True` to keep failures). Policies are registered in
  `POLICIES` in `envs/record_scripted.py`.
- Compress: `ffmpeg -y -i in.mp4 -vcodec libx264 -crf 30 -preset slow -pix_fmt yuv420p -movflags +faststart out.mp4`.
- Put it at `public/envs/video/scripted_<task>.mp4`. Baseline videos are `public/envs/video/<task>.mp4`.

## Deploy
- `.github/workflows/pages.yml`: a `build` job uploads `public/`, a separate `deploy` job deploys. Triggers: push to `main`
  touching `public/**` or the workflow, and `workflow_dispatch`.
- **A commit that only touches `envs/` or `docs/` does not deploy.** Only `public/**` changes do.
- Gotcha: with upload and deploy in one job, `deploy-pages` could not see the artifact ("Found 0 artifacts") and failed
  every time. Separate jobs fixed it. Re-running the failed run did not help.
- Verify: `gh run list --limit 1`, then `curl -s <url> | grep -c "<new text>"`.
- Check layout locally: `python3 -m http.server 8765` from `public/`, then `browse goto http://localhost:8765/envs/` and
  `browse screenshot <path>`; stop the server afterwards.
