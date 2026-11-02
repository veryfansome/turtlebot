---
name: evolve-runpod-round
description: Run the evolve search over the world model's training in this repo (learn/evolve, registry mode) with scoring on RunPod - finish its setup (noise floor, gen 0) or run a round whose candidates are scored as a cloud wave, ingested, and pushed to the private campaign repo. Use instead of plain /evolve:round scoring whenever the user asks to evolve, run a round, or finish the evolve setup here. Every pod launch needs the user's go for that wave.
---

# Evolve rounds here, scored on RunPod

[learn/evolve/EVOLVE.md](../../../learn/evolve/EVOLVE.md) is the contract: read it first. The engine runs from the repo root as `evolve --root learn ...`. Scores count only in the RunPod env, never the Mac's.

## Before anything

1. `(cd learn && uv run python evolve/campaign.py pull)`: the archive, impls, round notes and `insights.md` live in `veryfansome/turtlebot-evolve`; a fresh clone has none of them.
2. `evolve --root learn doctor`, and the cloud-wave skill's data-pin check.

## Finishing setup (once; not done as of 2026-11-01)

3. With the user's go, the speed probe (cloud-wave skill: `wave.py probe <name>`, about $0.50). It sizes everything after it.
4. Noise floor and gen 0: a wave of the baseline over 5 seeds (`noise_runs`), e.g. `(cd learn && uv run python ../cloud/wave.py make noise-1 --genome baseline=<the baseline genome> --seeds 0,1,2,3,4)` (check `wave.py make --help` first), run through the cloud-wave skill and ingested. How the measured spread becomes `fitness.noise_floor`, and how the ingested baseline becomes gen 0, isn't settled: read `evolve doctor --help` and `evolve ingest --help` first, and agree it with the user. Editing evolve.json is a protected change (`EVOLVE_ALLOW_PROTECTED=1`, then `doctor`); set `fitness.selection_env` to the pod's env tag. Don't use `evolve seed` or `doctor --measure-noise` locally: those measure the Mac.

## A round

5. Follow /evolve:round through sampling, jails, inventors and `evolve guard`, with a fresh `--seed` tag.
6. **Don't run `evolve score` locally.** Collect the guarded genomes and a lineage JSON (`{id: {parent, inventor, operator, axis_changed, rationale, generation}}`), make sure any code the wave needs is committed (ask the user: ship-changes skill), then `(cd learn && uv run python ../cloud/wave.py make <round>-cands --genome ID=PATH ... --meta LINEAGE.json)`.
7. Run it through the cloud-wave skill (reap, estimate, the user's go, `bash cloud/lane.sh run`, summary). Differences under the noise floor are noise.
8. `(cd learn && uv run python ../cloud/wave.py ingest <round>-cands)`, `evolve --root learn board` to see the effect, then `(cd learn && uv run python evolve/campaign.py push)`.
9. Report each child against its parent, with numbers. The search never crowns: making a candidate the default is a separate change, through train-model and verify-changes.

## Rules

- Never edit an archived impl, the archive, `learn/eval/` or `learn/pet_learn/` during a round.
- The final split (sessions recorded after setup) never picks parents; the pod side doesn't stage it yet.
- Push the campaign after every round, even one with nothing new.
