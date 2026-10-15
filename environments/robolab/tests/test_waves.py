"""Wave runner: argument parsing, key check and summary aggregation."""

from __future__ import annotations

import pytest

import run_waves
import summary


def test_parse_and_expand_tasks():
    args = run_waves.parse_args(
        [
            "--model", "hold", "--tasks", "BananaInBowlTask,PickDrillTask",
            "--episodes-per-task", "3", "--episode-offset", "2", "--concurrency", "4",
            "--output", "/tmp/x", "--image", "modal://im-test",
        ]
    )  # fmt: skip
    tasks = run_waves.build_tasks(args, "hold")
    assert [(t.args["task_name"], t.args["episode"]) for t in tasks] == [
        ("BananaInBowlTask", 2),
        ("BananaInBowlTask", 3),
        ("BananaInBowlTask", 4),
        ("PickDrillTask", 2),
        ("PickDrillTask", 3),
        ("PickDrillTask", 4),
    ]
    assert all(t.env == "dreamscale-robolab" and t.id == "episode" for t in tasks)
    assert len({t.slug for t in tasks}) == len(tasks)
    assert tasks[0].columns["difficulty"] == "simple"


def test_parse_requires_image_and_known_tasks():
    with pytest.raises(SystemExit):
        run_waves.parse_args(["--model", "hold", "--output", "/tmp/x", "--image", ""])
    with pytest.raises(SystemExit):
        run_waves.parse_args(
            ["--model", "hold", "--output", "/tmp/x", "--image", "m", "--tasks", "Nope"]
        )
    with pytest.raises(SystemExit):
        run_waves.parse_args(["--model", "pi0", "--output", "/tmp/x", "--image", "m"])


def test_parse_without_image_explains_how_to_build_one(monkeypatch, capsys):
    monkeypatch.setattr(run_waves, "IMAGE", "")
    with pytest.raises(SystemExit):
        run_waves.parse_args(["--model", "hold", "--output", "/tmp/x"])
    assert "build_image.py" in capsys.readouterr().err
    args = run_waves.parse_args(["--model", "hold", "--output", "/tmp/x", "--build-image"])
    assert args.build_image and not args.image


def test_require_dreamscale_key(monkeypatch, tmp_path):
    from dreamscale.config import Config, save_config

    monkeypatch.delenv("DREAMSCALE_API_KEY", raising=False)
    monkeypatch.setenv("DREAMSCALE_HOME", str(tmp_path))
    with pytest.raises(SystemExit, match="dreamscale login"):
        run_waves.require_dreamscale_key()
    monkeypatch.setenv("DREAMSCALE_API_KEY", "k")
    run_waves.require_dreamscale_key()
    monkeypatch.delenv("DREAMSCALE_API_KEY")
    save_config(Config(api_key="k"), save_api_key=True)
    run_waves.require_dreamscale_key()


def _record(task, episode, *, success, score, rtt, server, boot, lifetime, error=None):
    run = {
        "trace_id": f"trace-{task}-{episode}",
        "reward": 1.0 if success else 0.0,
        "grade_info": {}
        if error
        else {"success": success, "robolab_score": score, "steps": 300, "max_episode_length": 750},
        "grade_is_error": bool(error),
        "trace_status": "error" if error else "completed",
        "error": error,
    }
    agent = {
        "agent_called_unix": 1000.0 + boot + 20,
        "session_connect_s": 2.0,
        "first_observation_s": 2.5,
        "loop_wall_s": 90.0,
        "requests": 10,
        "sdk_rtt_ms": {"n": 1, "p50": rtt},
        "session": {"session_id": "s", "transport": "quic"},
        "timing_file": "episodes/x/timing.jsonl",
    }
    sandbox = {
        "requested_unix": 1000.0,
        "ready_unix": 1000.0 + boot,
        "released_unix": 1000.0 + lifetime,
        "instance_id": "sb-1",
    }
    return summary.episode_record(
        task_name=task, episode=episode, run=run, agent=agent, sandbox=sandbox
    )


def test_aggregate_success_scores_latency_and_cost():
    records = [
        _record("A", 0, success=True, score=1.0, rtt=200, server=90, boot=100, lifetime=400),
        _record("A", 1, success=False, score=0.5, rtt=300, server=95, boot=120, lifetime=500),
        _record("B", 0, success=False, score=None, rtt=400, server=99, boot=140, lifetime=600),
        _record("B", 1, success=None, score=None, rtt=0, server=0, boot=160, lifetime=700,
                error="sandbox failed"),
    ]  # fmt: skip
    rows = [
        {"event": "inference", "chunk_index": i % 3, "sdk_rtt_ms": v, "server_inference_ms": s,
         "timing": {"data_plane_rtt_ms": 20.0}}
        for i, (v, s) in enumerate([(100, 50), (200, 60), (300, 70), (400, 80), (1000, 90)])
    ]  # fmt: skip
    out = summary.aggregate(records, inference_rows=rows, meta={"model": "m", "job": {}})
    assert out["overall"] == {
        "episodes": 4,
        "graded": 3,
        "errors": 1,
        "successes": 1,
        "success_rate": pytest.approx(1 / 3),
        "mean_score": pytest.approx((1.0 + 0.5) / 2),
    }
    assert out["per_task"]["A"]["success_rate"] == 0.5
    assert out["per_task"]["B"]["graded"] == 1
    assert out["latency_ms"]["sdk_rtt"]["p50"] == 300
    assert out["latency_ms"]["sdk_rtt"]["p95"] == pytest.approx(880)
    assert out["latency_ms"]["server_inference"]["p50"] == 70
    assert out["latency_ms"]["first_request_sdk_rtt"]["n"] == 2
    assert out["startup_s"]["sandbox_boot_s"]["p50"] == pytest.approx(130)
    assert out["startup_s"]["sandbox_ready_to_agent_s"]["p50"] == pytest.approx(20)
    assert out["modal"]["sandbox_seconds"] == pytest.approx(2200)
    assert out["modal"]["estimated_usd"] == pytest.approx(
        2200 / 3600 * summary.modal_usd_per_sandbox_hour()
    )
    first = out["episodes"][0]
    assert first["trace_url"] == "https://hud.ai/trace/trace-A-0"
    assert first["success"] is True and first["steps"] == 300
    assert out["episodes"][3]["graded"] is False and out["episodes"][3]["success"] is None


def test_read_inference_rows(tmp_path):
    path = tmp_path / "timing.jsonl"
    path.write_text('{"event": "session_ready"}\n{"event": "inference", "sdk_rtt_ms": 5}\n\n')
    assert summary.read_inference_rows(path) == [{"event": "inference", "sdk_rtt_ms": 5}]
    assert summary.read_inference_rows(tmp_path / "missing.jsonl") == []
