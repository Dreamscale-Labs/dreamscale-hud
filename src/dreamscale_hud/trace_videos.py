"""Download a HUD job's (or one trace's) camera videos as MP4s, plus a side-by-side mosaic.

python -m dreamscale_hud.trace_videos https://hud.ai/jobs/<job-id> --output videos

Each trace becomes ``<task>-<trace8>/<camera>.mp4`` and ``mosaic.mp4`` (cameras left to
right at a shared height), with ``traces.json`` listing each trace's task, reward and
grade. Works for any HUD robot trace, including `dreamscale run yam --hud`.
"""

import argparse
import json
import re
import urllib.request
import uuid
from pathlib import Path

import av
import numpy as np
from hud.utils.platform import PlatformClient

_UUID = re.compile(r"[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}", re.I)


def _id(text):
    match = _UUID.search(text)
    if match is None:
        raise ValueError(f"No HUD job or trace ID in {text!r}")
    return str(uuid.UUID(match[0]))


def job_traces(client, job_id):
    traces, offset = [], 0
    while True:
        page = client.get(f"/jobs/{job_id}/traces", params={"limit": 100, "offset": offset})
        traces += page["items"]
        offset += len(page["items"])
        if not page["items"] or offset >= page.get("total", offset):
            return traces


def trace_events(client, trace_id):
    page = client.get(f"/trace/{trace_id}/events")
    events = list(page["events"])
    while page.get("has_more") and events:
        page = client.get(f"/trace/{trace_id}/events", params={"since_seq": events[-1]["seq"]})
        fresh = [e for e in page["events"] if e["seq"] > events[-1]["seq"]]
        if not fresh:
            break
        events += fresh
    return events


def camera_segments(events):
    """{camera: [segment URL by index]}, refusing gaps that would corrupt playback."""
    cameras = {}
    for event in events:
        if event.get("kind") == "robot_video_segment":
            cameras.setdefault(event["camera"], {})[event["index"]] = event["url"]
    for camera, segments in cameras.items():
        if sorted(segments) != list(range(len(segments))):
            raise ValueError(f"Incomplete HUD video segments for camera {camera}")
    return {
        camera: [segments[i] for i in range(len(segments))] for camera, segments in cameras.items()
    }


def _frames(path):
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def write_mosaic(videos, target, *, fps):
    """Concatenate cameras horizontally at the smallest height; shorter streams hold."""
    streams = [_frames(path) for path in videos]
    height = min(frames[0].shape[0] for frames in streams if frames)
    length = max(len(frames) for frames in streams)
    with av.open(str(target), "w") as output:
        stream = None
        for index in range(length):
            row = []
            for frames in streams:
                frame = frames[min(index, len(frames) - 1)]
                step = frame.shape[0] / height
                rows = (np.arange(height) * step).astype(int)
                cols = (np.arange(int(frame.shape[1] / step)) * step).astype(int)
                row.append(frame[rows][:, cols])
            image = np.concatenate(row, axis=1)
            image = image[: image.shape[0] // 2 * 2, : image.shape[1] // 2 * 2]
            if stream is None:
                stream = output.add_stream("libx264", rate=fps)
                stream.width, stream.height = image.shape[1], image.shape[0]
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "18"}
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                output.mux(packet)
        if stream is not None:
            for packet in stream.encode(None):
                output.mux(packet)


def download(reference, destination: Path, *, client=None):
    client = client or PlatformClient.from_settings()
    reference_id = _id(reference)
    if "/trace" in reference:
        traces = [{"id": reference_id}]
    else:
        traces = job_traces(client, reference_id)
    destination.mkdir(parents=True, exist_ok=True)
    summary = []
    for trace in traces:
        events = trace_events(client, trace["id"])
        segments = camera_segments(events)
        fps = next((e["fps"] for e in events if e.get("kind") == "robot_video_segment"), 30)
        folder = destination / f"{trace.get('task_slug') or 'trace'}-{trace['id'][:8]}"
        folder.mkdir(exist_ok=True)
        videos = []
        for camera, urls in sorted(segments.items()):
            target = folder / f"{camera}.mp4"
            with target.open("wb") as output:
                for url in urls:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        output.write(response.read())
            videos.append(target)
        if len(videos) > 1:
            write_mosaic(videos, folder / "mosaic.mp4", fps=fps)
        grade = (trace.get("metadata") or {}).get("operator_grade")
        summary.append(
            {
                "trace_id": trace["id"],
                "task": trace.get("task_slug"),
                "reward": trace.get("reward"),
                "grade": grade,
                "folder": folder.name,
                "cameras": [video.stem for video in videos],
            }
        )
        print(f"{folder}: {', '.join(v.name for v in videos) or 'no video'}")
    (destination / "traces.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", help="HUD job or trace URL or ID")
    parser.add_argument("--output", type=Path, default=Path("hud-videos"))
    args = parser.parse_args()
    download(args.reference, args.output)


if __name__ == "__main__":
    main()
