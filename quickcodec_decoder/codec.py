import os
import subprocess
import multiprocessing
import numpy as np
import av
from pathlib import Path
from typing import List, Tuple
import whisper
import glob
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def extract_keyframes(video_path: str) -> List[float]:
    """Use ffprobe to extract all keyframe timestamps in seconds."""
    cmd = [
        "ffprobe", "-select_streams", "v", "-show_frames",
        "-show_entries", "frame=pkt_pts_time,key_frame",
        "-of", "csv", video_path
    ]
    output = subprocess.check_output(cmd).decode("utf-8")
    print("[DEBUG] Raw ffprobe output (first 500 chars):")
    print(output[:500])
    timestamps = []
    for line in output.splitlines():
        parts = line.split(',')
        if len(parts) == 3 and parts[1] == '1':  # key_frame == 1
            try:
                timestamps.append(float(parts[2]))
            except ValueError:
                print(f"[WARN] Could not parse timestamp from line: {line}")
    print(f"[DEBUG] Found {len(timestamps)} keyframe timestamps.")
    return timestamps

def split_intervals(timestamps: List[float], num_segments: int, fallback_duration: float = None) -> List[Tuple[float, float]]:
    """Split keyframe timestamps or fallback to fixed duration intervals."""
    if timestamps:
        if len(timestamps) < num_segments:
            num_segments = len(timestamps)
        print(f"[DEBUG] Splitting {len(timestamps)} timestamps into {num_segments} segments.")
        segments = []
        segment_size = len(timestamps) // num_segments
        for i in range(num_segments):
            start_idx = i * segment_size
            start = timestamps[start_idx]
            end_idx = (i + 1) * segment_size if i < num_segments - 1 else len(timestamps) - 1
            end = timestamps[end_idx]
            print(f"[DEBUG] Segment {i}: start={start:.2f}s, end={end:.2f}s")
            segments.append((start, end))
        return segments
    else:
        if fallback_duration is None:
            fallback_duration = 60.0 * num_segments  # default to 1 minute per segment
        print(f"[WARN] No keyframes found. Falling back to fixed {fallback_duration / num_segments:.2f}s segments.")
        segment_length = fallback_duration / num_segments
        return [(i * segment_length, (i + 1) * segment_length) for i in range(num_segments)]

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

def extract_audio(video_path: str, output_audio_path: str = None):
    """Extract the audio track from the video."""
    if output_audio_path is None:
        base = Path(video_path).stem
        output_audio_path = f"{base}_audio.wav"
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        output_audio_path
    ]
    subprocess.run(cmd, check=True)
    print(f"[INFO] Extracted audio to {output_audio_path}")
    return output_audio_path

def transcribe_audio(audio_path: str, model_size: str = "base") -> List[dict]:
    """Transcribe audio using OpenAI Whisper and return segments."""
    print(f"[INFO] Transcribing audio with Whisper ({model_size})...")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, verbose=True)
    print("[INFO] Transcription complete.")
    return result["segments"]

def generate_frame_captions(frame_dir: Path) -> dict:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    captions = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for path in sorted(glob.glob(str(frame_dir / "*.jpg"))):
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions[path] = caption
    return captions

def run_parallel_decode(video_path: str, fps: int = 1, num_workers: int = 4, output_dir: str = "frames"):
    keyframes = extract_keyframes(video_path)
    duration_cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    try:
        duration = float(subprocess.check_output(duration_cmd).decode("utf-8").strip())
    except Exception as e:
        print(f"[ERROR] Failed to get video duration: {e}")
        return
    intervals = split_intervals(keyframes, num_workers, fallback_duration=duration)
    out_dir = Path(output_dir)
    args = [
        (video_path, start, end, fps, out_dir, idx)
        for idx, (start, end) in enumerate(intervals)
    ]
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(decode_segment, args)

    audio_path = extract_audio(video_path, str(out_dir / "audio.wav"))
    segments = transcribe_audio(audio_path)
    with open(out_dir / "transcript.txt", "w") as f:
        for seg in segments:
            f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}\n")
    print(f"[INFO] Transcription saved to {out_dir / 'transcript.txt'}")

    captions = generate_frame_captions(out_dir)
    with open(out_dir / "captions.txt", "w") as f:
        for path, caption in captions.items():
            f.write(f"{path}: {caption}\n")
    print(f"[INFO] Captions saved to {out_dir / 'captions.txt'}")
