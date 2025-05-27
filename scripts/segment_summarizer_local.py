import os
import re
import argparse
import requests
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

# Load and parse frame captions grouped by segment
def load_captions(captions_path):
    segment_captions = defaultdict(list)
    for line in Path(captions_path).read_text().splitlines():
        match = re.search(r"segment(\d+)_\d+\.jpg", line)
        if match:
            seg_id = int(match.group(1))
            caption = line.split(": ", 1)[-1]
            segment_captions[seg_id].append(caption)
    return segment_captions

# Load and parse transcript segments grouped into time chunks
def load_transcript(transcript_path, chunk_size=60.0):
    segment_transcripts = defaultdict(str)
    segment_timestamps = defaultdict(lambda: [float('inf'), float('-inf')])
    for line in Path(transcript_path).read_text().splitlines():
        match = re.match(r"\[(\d+\.\d+) - (\d+\.\d+)\] (.*)", line)
        if match:
            start = float(match.group(1))
            end = float(match.group(2))
            text = match.group(3)
            segment_id = int(start // chunk_size)
            segment_timestamps[segment_id][0] = min(segment_timestamps[segment_id][0], start)
            segment_timestamps[segment_id][1] = max(segment_timestamps[segment_id][1], end)
            segment_transcripts[segment_id] += text + " "
    return segment_transcripts, segment_timestamps

# Summarize with HuggingFace model (Seq2Seq or CausalLM)
def summarize_with_hf_model(transcript, captions, model, tokenizer, device, is_causal):
    prompt = f"Transcript: {transcript}\n\nCaptions: {', '.join(captions)}\n\nSummarize this lecture segment."
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    if is_causal:
        outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    else:
        outputs = model.generate(**inputs, max_length=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Summarize with Ollama local API
def summarize_with_ollama(transcript, captions, model):
    prompt = f"Transcript: {transcript}\n\nCaptions: {', '.join(captions)}\n\nSummarize this lecture segment."
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })
    return response.json().get("response", "[ERROR] No response from Ollama")

def main(captions_path, transcript_path, output_path, engine, model_name):
    captions = load_captions(captions_path)
    transcript, timestamps = load_transcript(transcript_path)

    is_causal = False
    model, tokenizer, device = None, None, None
    if engine == "hf":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        except:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            is_causal = True

    with open(output_path, "w") as f:
        for seg_id in sorted(transcript):
            if engine == "hf":
                summary = summarize_with_hf_model(transcript[seg_id], captions.get(seg_id, []), model, tokenizer, device, is_causal)
            elif engine == "ollama":
                summary = summarize_with_ollama(transcript[seg_id], captions.get(seg_id, []), model_name)
            else:
                summary = "[ERROR] Unknown engine"
            start, end = timestamps.get(seg_id, (seg_id * 60, (seg_id + 1) * 60))
            f.write(f"Segment {seg_id} ({start:.2f}–{end:.2f} seconds):\n{summary}\n\n")
    print(f"[INFO] Summaries saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=str, required=True)
    parser.add_argument("--transcript", type=str, required=True)
    parser.add_argument("--output", type=str, default="summaries.txt")
    parser.add_argument("--engine", choices=["hf", "ollama"], default="hf", help="Choose backend engine")
    parser.add_argument("--model", type=str, default="google/flan-t5-base", help="Model name (HuggingFace or Ollama)")
    args = parser.parse_args()
    main(args.captions, args.transcript, args.output, args.engine, args.model)
