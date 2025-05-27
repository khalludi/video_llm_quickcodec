import os
import argparse
import subprocess
from pathlib import Path

def clip_video(input_path: str, output_path: str, duration: int = 300):
    """Trim the video to the first N seconds."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-t", str(duration),
        "-c:v", "libx264", "-c:a", "aac",
        output_path
    ]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Path to save the clipped video")
    parser.add_argument("--duration", type=int, default=300, help="Clip duration in seconds (default: 300)")
    args = parser.parse_args()
    clip_video(args.input, args.output, args.duration)

if __name__ == "__main__":
    main()