# RoboLab DROID on HUD with Dreamscale policies

RoboLab (Isaac Lab) DROID tasks on HUD. Each episode's simulator runs in its own
Modal L40S sandbox (HUD `ModalRuntime`); the agent runs on your machine and
calls a Dreamscale policy through the public `dreamscale` SDK. Models:
`cosmos3-nano-policy-droid` and `flux-3-action-droid`, plus `hold` for a
no-inference smoke.

## Prerequisites

- Python 3.11 and [uv](https://docs.astral.sh/uv/).
- A HUD API key: `HUD_API_KEY` or `hud set`.
- A Modal account and token: `modal token set` (or `MODAL_TOKEN_ID` /
  `MODAL_TOKEN_SECRET`). The simulator sandboxes run and bill on your account.
- A Dreamscale login for real models: `uv run --project environments/robolab
  dreamscale login`, or `DREAMSCALE_API_KEY` in the environment. The two DROID
  models may need access enabled on your account; contact Dreamscale if a
  session is refused.

Commands below run from the repo root. This directory is its own uv project
(`uv sync --project environments/robolab` installs it).

## Build the simulator image once

```sh
uv run --project environments/robolab python environments/robolab/build_image.py
export ROBOLAB_MODAL_IMAGE=modal://im-...   # the id it prints
```

The image is Isaac Sim 5.0 plus RoboLab at the pinned revision. Modal caches
layers, so rebuilding after a change to `env.py` / `sim.py` is quick. Instead of
an image id you can pass `--build-image` to any run.

## Smoke test

```sh
uv run --project environments/robolab python environments/robolab/run_waves.py \
  --model hold --tasks BananaInBowlTask --max-steps 64 --output runs/hold-smoke
```

No inference: the arm holds its pose. It checks Modal, the image and HUD end to
end and prints the HUD job URL.

## Run a wave

```sh
uv run --project environments/robolab python environments/robolab/run_waves.py \
  --model cosmos3-nano-policy-droid --tasks all --episodes-per-task 5 \
  --concurrency 10 --output runs/cosmos3-w1
```

Swap in `--model flux-3-action-droid` for the other policy. Useful flags:
`--tasks BananaInBowlTask,PickDrillTask`, `--episode-offset K` (continue a wave),
`--keep-warm 300` (SDK `keep_warm`, 0–3600 s), `--max-steps N`,
`--no-local-video`, `--dry-run` (print the episodes and exit).

## Outputs

In `--output`:

- `summary.json`: per-episode success, RoboLab score, steps and HUD trace URL;
  per-task and overall success rates; latency and startup timings; Modal
  sandbox-seconds with a rough cost estimate; the HUD job URL.
- `wave.json`: the plan and job URL, written before the first sandbox starts.
- `episodes/<task>__ep<k>__<trace>/`: `timing.jsonl` (one row per SDK request
  with the SDK's timing fields), `episode.json`, and `policy_views.mp4`
  (left | wrist | right at 15 fps).

HUD keeps the canonical trace with per-camera video and every executed chunk.
`aggregate.py` combines several waves of one model into a single result, and
`make_mosaic.py` renders a wave as a grid video.

## What the harness does

- **Scene and cameras** follow RoboLab's own Cosmos runner:
  `auto_register_droid_envs(task, cameras=WRIST_LEFT_RIGHT_HEAD)`,
  `create_env(seed=scene_seed=0)`. The env publishes `exterior_1_left`,
  `exterior_2_left` and `wrist_left` at 640x360 (RoboLab's `resize_with_pad`).
- **State** is the 7 Franka joint positions (rad) plus RoboLab's gripper closed
  fraction in [0, 1]. Actions are 7 absolute joint targets plus the gripper,
  thresholded at 0.5.
- **Tasks**: `tasks.json` holds ten tasks across RoboLab's difficulty labels.
  Each episode runs to RoboLab's own time limit (`episode_s` x 15 steps).
- **Grade**: reward is RoboLab's success (1/0); `info` adds the subtask score,
  termination step and steps executed.
- **Agent**: one SDK session per episode (`rtc="off"`, `calibration="off"`,
  `control_hz=15`), opened before connecting to the sim. Each reply is a 32x8
  chunk executed open-loop before the next request. The sim only steps when an
  action arrives, so simulated time and scores do not depend on latency.

## Notes

- Every observation (3 x 640x360 RGB, ~2 MB) travels from the sandbox to your
  machine each step, so wall-clock is dominated by transfer: a 750-step episode
  takes about 10 minutes plus a few minutes of sandbox boot. Keep concurrency
  within your downlink (~10 episodes needs 200+ Mbit/s).
- Modal costs roughly $2.7 per sandbox-hour (L40S + 8 CPU + 48 GiB); one pass
  over all ten tasks is about 2.5 sandbox-hours. The summary's estimate uses list
  prices; your Modal dashboard is the bill.
- Isaac Sim 5.0 needs the 580 NVIDIA driver, so sandboxes are pinned to Modal's
  OCI cloud by default (`--sandbox-cloud ''` lets Modal choose).
- Each sandbox is terminated at the end of its episode; the runner re-checks
  every sandbox it started and terminates any survivor.

Tests (no GPU, no network): `cd environments/robolab && uv run pytest && uv run ruff check .`
