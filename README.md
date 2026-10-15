# Dreamscale × HUD

Run [Dreamscale](https://dreamscalelabs.com) robot policies and record every rollout on
[HUD](https://hud.ai): instruction, joints, action chunks, camera video and a grade.

## Record real YAM rollouts

This is how the blog's MolmoAct2-BimanualYAM rollouts were recorded. You need a bimanual
YAM rig and a Linux computer wired to it.

1. **Install the CLI and sign in.**

   ```bash
   uv tool install dreamscale
   dreamscale login
   ```

2. **Set up the rig once** (cameras and arms, in a browser page it prints):

   ```bash
   dreamscale robots yam setup
   dreamscale doctor yam
   ```

3. **Connect HUD once** (installs HUD support and saves your HUD API key):

   ```bash
   dreamscale integrations hud setup
   ```

4. **Run with `--hud`:**

   ```bash
   dreamscale run yam --model molmoact2-bimanual-yam --hud --hud-job "MolmoAct2-BimanualYAM"
   ```

   Type an instruction, watch the preview, press Enter to execute. When the task stops,
   the arms hold while you grade it: `y`, `n` or a 0–1 score, plus an optional note
   (`n dropped the towel`). Repeat for more tasks; Ctrl+C ends the session.

5. **See it on HUD.** The session is one job on [hud.ai](https://hud.ai); each task is a
   trace with one video per camera, and its grade is the reward.

6. **Download the videos** (optional). From a clone of this repo:

   ```bash
   uv sync
   uv run python -m dreamscale_hud.trace_videos https://hud.ai/jobs/JOB_ID --output videos
   ```

   You get one MP4 per camera, a side-by-side mosaic per trace, and `traces.json` with
   each task's reward and grade.

`dreamscale run so101 --hud` works the same way on an SO-101 arm.

## Simulated benchmarks

Here HUD runs the simulator and this repo's agent calls Dreamscale for actions.
Install with Python 3.12 and [uv](https://docs.astral.sh/uv/), then sign in to both:

```bash
uv sync
uv run dreamscale login
uv run hud login
```

### LIBERO (MolmoAct2-LIBERO)

```bash
docker build -f Dockerfile.hud -t dreamscale-hud-libero:local .
uv run dreamscale-hud --task-ids 0 --init-state-ids 0
```

That runs one episode in a local CPU simulator and records it on HUD. Omit the two
options for six episodes; `--suite`, `--all-suites` and `--taskset tasksets/five-suites.json`
select other LIBERO tasks. Results, timings and per-camera MP4s go to `artifacts/<time>/`.

To run the simulator on HUD instead of Docker, deploy it to your HUD workspace once:

```bash
uv run hud deploy . --no-env --runtime hud
uv run dreamscale-hud --runtime hud
```

Build the image for your machine's own CPU architecture: LIBERO's physics is wrong under
x86 emulation on Apple Silicon, and the simulator refuses to start when it detects that.

### RoboLab, RoboTwin, RoboDojo

Isaac Lab and Sapien simulators that run in Modal GPU sandboxes through HUD. RoboLab
runs Dreamscale's DROID policies; see [environments/robolab](environments/robolab/README.md).
RoboTwin and RoboDojo ship a hold-pose agent:

```bash
uv run python environments/robotwin/run_modal.py
uv run python environments/robodojo/run_modal.py
```

Each needs your Modal token and HUD key in `environments/<env>/.env` (see `.env.example`).

## Use the agent in Python

```python
from hud import HUDRuntime, Taskset
from dreamscale_hud import DreamscaleRobotAgent
from dreamscale_hud.cli import tasks

async with DreamscaleRobotAgent() as agent:
    job = await Taskset("libero", tasks()).run(agent, runtime=HUDRuntime(), max_concurrent=1)
```

## Development

```bash
uv run ruff check .
uv run pytest -q
```

Dreamscale usage and HUD/Modal simulation are billed by their providers. Integration
code is MIT licensed; dependencies, checkpoints and simulator assets keep their own
licenses ([THIRD_PARTY.md](THIRD_PARTY.md)).
