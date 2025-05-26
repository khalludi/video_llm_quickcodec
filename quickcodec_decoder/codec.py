import os
import subprocess
import multiprocessing
import numpy as np
import av
from pathlib import Path
from typing import List, Tuple

def extract_keyframes(video_path: str) -> List[float]:
    """Use ffprobe to extract all keyframe timestamps in seconds."""
    cmd = [
        "ffprobe", "-select_streams", "v", "-show_frames",
        "-show_entries", "frame=pkt_pts_time,key_frame",
        "-of", "csv", video_path
    ]
    output = subprocess.check_output(cmd).decode("utf-8")
    timestamps = []
    for line in output.splitlines():
        parts = line.split(',')
        if len(parts) == 3 and parts[1] == '1':  # key_frame == 1
            timestamps.append(float(parts[2]))
    return timestamps

def split_intervals(timestamps: List[float], num_segments: int) -> List[Tuple[float, float]]:
    """Split keyframe timestamps into N intervals."""
    segments = []
    segment_size = len(timestamps) // num_segments
    for i in range(num_segments):
        start = timestamps[i * segment_size]
        end = timestamps[(i + 1) * segment_size] if i < num_segments - 1 else timestamps[-1]
        segments.append((start, end))
    return segments

def decode_segment(video_path: str, start: float, end: float, fps: int, out_dir: Path, segment_id: int):
    """Decode a segment between [start, end] at specified FPS."""
    out_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = out_dir / f"segment{segment_id}_%04d.jpg"
    cmd = [
        "ffmpeg", "-ss", str(start), "-to", str(end), "-i", video_path,
        "-vf", f"fps={fps},scale=224:224",
        "-q:v", "2", str(output_pattern)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_parallel_decode(video_path: str, fps: int = 1, num_workers: int = 4, output_dir: str = "frames"):
    keyframes = extract_keyframes(video_path)
    intervals = split_intervals(keyframes, num_workers)
    out_dir = Path(output_dir)
    args = [
        (video_path, start, end, fps, out_dir, idx)
        for idx, (start, end) in enumerate(intervals)
    ]
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(decode_segment, args)