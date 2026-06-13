# stackelberg_codepo

Refactored project for Stackelberg incentive-aware preference optimization in collaborative code generation.

The primary end-to-end smoke path is now native to this package. The codebase is intentionally modular, with the validated demo behavior migrated into these project boundaries:

- `execution`: code extraction and task validation.
- `preference`: utility, weighting, leader/follower pair construction, and planner-output cleaning.
- `training`: training command wrappers and artifact checks.
- `alternating`: one-iteration and future K-iteration orchestration.
- `cli`: a single main entry point.

## Tiny Verification

Run from the project root:

```bash
/opt/conda/bin/python scripts/main.py tiny-smoke --config configs/tiny.json
```

Expected outputs:

```text
outputs/tiny_smoke/leader_preferences.jsonl
outputs/tiny_smoke/leader_clean_preferences.jsonl
outputs/tiny_smoke/follower_preferences.jsonl
outputs/tiny_smoke/manifest.json
```

This verification uses a toy HumanEval-like task and local Python execution. It proves that the refactored modules can:

1. validate candidate code;
2. compute quality/cost/utility;
3. build leader preference pairs;
4. clean planner overreach;
5. build follower preference pairs;
6. write a manifest from a single CLI entry.

## Migration Status

The primary one-iteration algorithm path is now native to this package. The command
`full-algorithm-smoke` calls `stackelberg_codepo` modules for all validated stages:

- `alternating.leader_sampling`: multi-round leader/coder trajectory sampling with incentive budget, stopping, role-boundary penalty, and leader utility.
- `preference.leader_round_conversion` and `preference.leader_cleaning_wpo`: leader WPO construction and planner-overreach cleaning.
- `alternating.follower_sampling`: state-consistent follower candidate resampling from trajectory states.
- `training.weighted_dpo`: weighted DPO LoRA training.
- `evaluation.role_eval`: joint planner/coder adapter evaluation.

External demo scripts are no longer used by `full-algorithm-smoke`; the validated behavior has been migrated into this package.

## Real Trajectory Smoke

After `tiny-smoke`, run the first real-data migration check:

```bash
/opt/conda/bin/python scripts/main.py from-trajectories --config configs/real_trajectories_smoke.json
```

This reads the native full-algorithm smoke trajectory artifact and reconstructs:

```text
outputs/real_trajectories_smoke/trajectories_scored.jsonl
outputs/real_trajectories_smoke/leader_preferences.jsonl
outputs/real_trajectories_smoke/leader_clean_preferences.jsonl
outputs/real_trajectories_smoke/leader_wpo.jsonl
outputs/real_trajectories_smoke/follower_candidates.jsonl
outputs/real_trajectories_smoke/follower_preferences.jsonl
outputs/real_trajectories_smoke/follower_wpo.jsonl
outputs/real_trajectories_smoke/manifest.json
```

This verifies that the refactored project can consume native real trajectory artifacts instead of toy fixtures.

To also verify that the refactored project can hand real WPO data to the current weighted DPO smoke trainer:

```bash
/opt/conda/bin/python scripts/main.py real-paper-smoke --config configs/real_trajectories_smoke.json
```

This command trains 1-step leader/follower LoRA adapters under:

```text
outputs/real_trajectories_smoke/adapters/leader
outputs/real_trajectories_smoke/adapters/follower
```

## Full Algorithm Smoke

The project exposes a full algorithm smoke command. It runs one native alternating-optimization iteration and produces sample trajectories, leader/follower WPO data, leader/follower LoRA adapters, evaluation results, and a new-project report:

```bash
/opt/conda/bin/python scripts/main.py full-algorithm-smoke --config configs/full_algorithm_smoke.json
```

Main output:

```text
outputs/full_algorithm_smoke/manifest.json
outputs/full_algorithm_smoke/new_project_report.json
outputs/full_algorithm_smoke/adapters/leader
outputs/full_algorithm_smoke/adapters/follower
outputs/full_algorithm_smoke/eval
```

This command is the refactored project's current end-to-end algorithm entry. It runs the migrated native `stackelberg_codepo` modules directly.

Validated smoke result:

- Trajectories: 9
- Trajectory preferences: 2
- Clean leader WPO: 2
- Follower WPO: 4
- Leader adapter: `outputs/full_algorithm_smoke/adapters/leader`
- Follower adapter: `outputs/full_algorithm_smoke/adapters/follower`
- Eval: 2/3 pass on test split, average assert pass rate 0.6667
- Report: `outputs/full_algorithm_smoke/new_project_report.json`

This is the point where the refactored project has the same small-sample algorithm loop that the demo previously proved: sampling, preference construction, weighted DPO smoke training, and role evaluation.
