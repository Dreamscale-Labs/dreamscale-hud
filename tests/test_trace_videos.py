import av
import numpy as np
import pytest

from dreamscale_hud.trace_videos import _id, camera_segments, download, write_mosaic


def test_reference_ids_and_segment_gaps():
    job = "1d6367b9-b5ef-4fa0-b1b4-2c2afacf9c4a"
    assert _id(f"https://hud.ai/jobs/{job}") == job
    assert _id(job.replace("-", "")) == job
    with pytest.raises(ValueError):
        _id("https://hud.ai/jobs/")
    events = [
        {"kind": "robot_video_segment", "camera": "top", "index": i, "url": f"u{i}"} for i in (1, 0)
    ] + [{"kind": "robot_observation"}]
    assert camera_segments(events) == {"top": ["u0", "u1"]}
    events.append({"kind": "robot_video_segment", "camera": "left", "index": 1, "url": "x"})
    with pytest.raises(ValueError, match="left"):
        camera_segments(events)


def _video(path, height, width, frames):
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for i in range(frames):
            image = np.full((height, width, 3), i * 10, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)


def test_mosaic_shares_height_and_holds_shorter_streams(tmp_path):
    _video(tmp_path / "top.mp4", 480, 640, 6)
    _video(tmp_path / "left.mp4", 240, 320, 4)
    write_mosaic([tmp_path / "top.mp4", tmp_path / "left.mp4"], tmp_path / "m.mp4", fps=30)
    with av.open(str(tmp_path / "m.mp4")) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(video=0))
        assert (stream.width, stream.height, frames) == (320 + 320, 240, 6)


def test_download_job_writes_videos_mosaic_and_summary(tmp_path, monkeypatch):
    _video(tmp_path / "segment.mp4", 64, 96, 3)
    data = (tmp_path / "segment.mp4").read_bytes()

    class Client:
        def get(self, path, params=None):
            if path.endswith("/traces"):
                return {
                    "items": [
                        {
                            "id": "286820b0-6216-49c6-96bb-fc2bdbf4f6a2",
                            "task_slug": "fold-the-towel",
                            "reward": 0.5,
                            "metadata": {"operator_grade": {"reward": 0.5, "note": "half"}},
                        }
                    ],
                    "total": 1,
                }
            return {
                "events": [
                    {
                        "seq": i,
                        "kind": "robot_video_segment",
                        "camera": cam,
                        "index": 0,
                        "fps": 30,
                        "url": cam,
                    }
                    for i, cam in enumerate(("top", "left"))
                ],
                "has_more": False,
            }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return data

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout: Response())
    (summary,) = download(
        "https://hud.ai/jobs/1d6367b9-b5ef-4fa0-b1b4-2c2afacf9c4a",
        tmp_path / "out",
        client=Client(),
    )
    folder = tmp_path / "out" / "fold-the-towel-286820b0"
    assert summary["grade"]["note"] == "half" and summary["cameras"] == ["left", "top"]
    assert {p.name for p in folder.iterdir()} == {"left.mp4", "top.mp4", "mosaic.mp4"}
    assert (tmp_path / "out" / "traces.json").exists()
