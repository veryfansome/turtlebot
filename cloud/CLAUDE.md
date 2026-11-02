# cloud/: notes for Claude

Training waves on RunPod, scored the same way as the evolve search. Read [README.md](README.md); run every wave through the cloud-wave skill. This is where money is spent.

- Never launch a pod (`lane.sh run`, `runpod.sh launch`) without the owner's go for that wave, given after the cost estimate `wave.py` prints. A go covers one wave. Never launch one to test a change: use the offline tests (`(cd learn && uv run pytest ../cloud/tests -q)`) and a local `wave_pod.py` rehearsal with `PET_STEPS` and `PET_PYTHON` set.
- `bash cloud/runpod.sh reap` only lists pods: run it before a launch and report anything UNTRACKED to the owner.
- Run the scripts with `bash`, never zsh (zsh doesn't word-split their ssh options).
- Pods get `git archive HEAD` (commit what a wave needs first) and pull the searched sessions from the Hugging Face dataset (`hub pull --manifest eval/data_manifest.json`, run from `learn/`, at the revision in `learn/data_source.json`).
- Wave names are never reused. Commit a wave's `wave.json` and `genomes/`; its `chunks/` (copies of candidate impls) is git-ignored and goes to the campaign repo; `cloud/results/` is git-ignored.
- Every lane must end with "terminate: CONFIRMED stopped". If it doesn't, tell the owner at once, with the pod id. Never set `LANE_KEEP_POD=1` unless the owner asked for it.
- Scores compare only within one env (GPU type and concurrency, e.g. `runpod-geforce-rtx-4090-c3`), never with the Mac's.
- Never print `~/.runpod.env` or the tokens in it, or put them on a command line.
- Not run yet: the speed probe (about $0.50), then the evolve search's noise-floor and gen-0 waves, then a batch x lr grid.
