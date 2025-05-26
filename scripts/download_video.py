import os
import argparse
import subprocess
from pathlib import Path

def download_youtube_video(url: str, output_dir: str = "videos"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "--merge-output-format", "mp4",
        "-o", f"{output_dir}/%(title)s.%(ext)s",
        url
    ]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="YouTube video URL")
    parser.add_argument("--output", type=str, default="videos", help="Directory to save the video")
    args = parser.parse_args()
    download_youtube_video(args.url, args.output)

if __name__ == "__main__":
    main()