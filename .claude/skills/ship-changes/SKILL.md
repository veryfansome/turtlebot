---
name: ship-changes
description: Commit, push and open a PR for finished work in this repo, and after the owner merges it bring local main and dev up to the merge ("sync main and rebranch"). Use when the user asks to commit, push, open a PR, or sync after a merge. Commits and pushes happen only when the user asks.
---

# Shipping a change

Work happens on `dev`; PRs go from `dev` to `main`; the owner merges them on GitHub. A commit's subject is `area: plain sentence` (`learn: ...`, `cloud: ...`), its body says why in plain words, and it ends with the Co-Authored-By line the session gives.

## Committing

1. The verify-changes skill has run on this diff, or the user explicitly skipped it.
2. `git status` (untracked files included). Stage **by name** (`git add path/to/file ...`), never `git add .`, `-A` or a whole folder: a 17 MB pickle once got into a pushed commit and had to be squashed out of history.
3. Read `git diff --cached --stat`. Nothing under `data/`, `cloud/results/` or `cloud/waves/*/chunks/`, no campaign files from `learn/evolve/` (archive, rounds, impls beyond the baselines); no `.pt`, `.pkl`, `.npz`, `.mcap`; no secrets or home-network addresses (the repo is public). Ask about anything over about 1 MB.
4. One commit per topic when the diff covers several.

## Pushing and the PR

5. `git push origin dev`.
6. `gh pr create --base main --head dev`, body: `## Summary` (bullets per topic), what was tested and what wasn't, what the user must do next (e.g. a VM sync). End it with the generated-with line the session gives.
7. Don't merge it: the owner does.

## After the owner merges: sync main and rebranch

8. Check nothing is training (`ps aux | grep '[p]et_learn'`): the merge below can change files in `learn/`. If one is, wait for it to finish.
9. `git fetch origin main:main` (fast-forwards local `main` without checking it out).
10. On `dev`: `git merge --ff-only main`, then `git push origin dev`. If the fast-forward fails, stop and show `git log --oneline main..dev`.
11. Report `git rev-parse --short dev main origin/dev origin/main`: all four should match.

## Rules

- Never `git checkout`, `switch`, `stash`, `reset`, `rebase`, `merge` or `pull` while a training runs (`ps aux | grep '[p]et_learn'`): it loads code from disk.
- Never force-push, rewrite pushed history or delete branches without the user's say-so for that action.
- A commit that changes `learn/data_source.json` or `learn/eval/data_manifest.json` changes what RunPod waves can measure: call it out in the PR.
- The robot runs only pushed code: after a merge it should run, use the vm-sync skill.
