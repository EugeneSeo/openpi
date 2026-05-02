#!/usr/bin/env python3
"""Convert the local SO101 LeRobot v3 shard layout to OpenPI's episode layout."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SRC = Path("/cluster/project/cvg/students/dohkim/datasets/local/so101_teleop_test_filtered")
DEFAULT_DST = Path("/cluster/project/cvg/students/dohkim/datasets/local/so101_teleop_test_filtered_openpi")
VIDEO_KEYS = ("observation.images.front", "observation.images.wrist")


def _read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, indent=4)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_tasks(src: Path) -> tuple[dict[int, str], list[dict]]:
    tasks_path = src / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        rows = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    else:
        df = pq.read_table(src / "meta" / "tasks.parquet").to_pandas().reset_index()
        rows = [
            {"task_index": int(row["task_index"]), "task": str(row["task"])}
            for _, row in df.iterrows()
        ]
    return {int(row["task_index"]): str(row["task"]) for row in rows}, rows


def _run_ffmpeg_slice(
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    *,
    start_frame: int,
    num_frames: int,
    fps: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = start_frame / fps
    duration = num_frames / fps
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_time:.9f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration:.9f}",
        "-an",
        "-c:v",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def convert(src: Path, dst: Path, *, overwrite: bool = False) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(dst)

    info = _read_json(src / "meta" / "info.json")
    fps = int(info["fps"])
    tasks_by_index, task_rows = _load_tasks(src)
    stats = _read_json(src / "meta" / "stats.json")

    dst.mkdir(parents=True)
    (dst / "data" / "chunk-000").mkdir(parents=True)
    (dst / "meta").mkdir(parents=True)

    data_files = sorted((src / "data").glob("*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {src / 'data'}")

    episodes_rows: list[dict] = []
    episodes_stats_rows: list[dict] = []
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    for data_file in data_files:
        table = pq.read_table(data_file)
        df = table.to_pandas()
        file_stem = data_file.stem
        shard_start = 0

        for ep_index, ep_df in df.groupby("episode_index", sort=True):
            ep_index = int(ep_index)
            ep_df = ep_df.sort_values("frame_index")
            length = len(ep_df)
            task_indices = sorted({int(x) for x in ep_df["task_index"].tolist()})
            episode_tasks = [tasks_by_index[i] for i in task_indices]

            ep_table = pa.Table.from_pandas(ep_df, preserve_index=False)
            out_data = dst / "data" / "chunk-000" / f"episode_{ep_index:06d}.parquet"
            pq.write_table(ep_table, out_data)

            for video_key in VIDEO_KEYS:
                in_video = src / "videos" / video_key / data_file.parent.name / f"{file_stem}.mp4"
                out_video = dst / "videos" / "chunk-000" / video_key / f"episode_{ep_index:06d}.mp4"
                _run_ffmpeg_slice(
                    ffmpeg,
                    in_video,
                    out_video,
                    start_frame=shard_start,
                    num_frames=length,
                    fps=fps,
                )

            episodes_rows.append(
                {
                    "episode_index": ep_index,
                    "tasks": episode_tasks,
                    "length": length,
                }
            )
            episodes_stats_rows.append({"episode_index": ep_index, "stats": stats})
            shard_start += length

    out_info = dict(info)
    out_info["codebase_version"] = "v2.1"
    out_info["total_episodes"] = len(episodes_rows)
    out_info["total_frames"] = sum(row["length"] for row in episodes_rows)
    out_info["total_tasks"] = len(task_rows)
    out_info["chunks_size"] = 1000
    out_info["total_chunks"] = 1
    out_info["total_videos"] = len(episodes_rows) * len(VIDEO_KEYS)
    out_info["splits"] = {"train": f"0:{len(episodes_rows)}"}
    out_info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    out_info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    _write_json(dst / "meta" / "info.json", out_info)
    _write_json(dst / "meta" / "stats.json", stats)
    _write_jsonl(dst / "meta" / "tasks.jsonl", sorted(task_rows, key=lambda row: row["task_index"]))
    _write_jsonl(dst / "meta" / "episodes.jsonl", sorted(episodes_rows, key=lambda row: row["episode_index"]))
    _write_jsonl(
        dst / "meta" / "episodes_stats.jsonl",
        sorted(episodes_stats_rows, key=lambda row: row["episode_index"]),
    )
    if (src / "README.md").exists():
        shutil.copy2(src / "README.md", dst / "README.md")

    print(f"Converted {len(episodes_rows)} episodes to {dst}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert(args.src, args.dst, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
