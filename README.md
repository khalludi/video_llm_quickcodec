# 📚 Video Lecture Indexer (Multimodal + LLM)

Automatically break down educational videos into meaningful, timestamped summaries using both audio (speech) and visual (frame captions) cues. Entirely local — no cloud APIs required.

---

## ✨ Features

* 🎞️ Frame extraction using FFmpeg (with or without keyframe alignment)
* 🎙️ Audio transcription using Whisper
* 🖼️ Frame captioning with BLIP (or plug in your own vision-language model)
* 💬 Segment summarization using local LLMs (Gemma via Ollama, or HuggingFace)
* 📁 Outputs clean, timestamped summaries of each lecture segment

---

## 🛠 Installation

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/video-lecture-indexer.git
cd video-lecture-indexer
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg

```bash
brew install ffmpeg     # macOS
sudo apt install ffmpeg # Ubuntu/Linux
```

### 4. (Optional) Install Ollama for local LLMs

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma:2b
```

---

## 🚀 How to Use

### 1. Download and optionally trim a video

```bash
python scripts/download_video.py --url "https://youtube.com/..."
python scripts/clip_video.py --input videos/raw.mp4 --output videos/clip.mp4 --duration 300
```

### 2. Extract frames + audio

```bash
python scripts/run_decoder.py --video videos/clip.mp4 --fps 1 --workers 4 --output frames
```

### 3. Generate segment summaries

```bash
python scripts/segment_summarizer_combined.py \
  --captions frames/captions.txt \
  --transcript frames/transcript.txt \
  --output summaries.txt \
  --engine ollama \
  --model gemma
```

---

## 🧠 Example Output

```
Segment 0 (0.00–60.00 seconds):
The lecturer introduces the topic of generative models and explains their recent breakthroughs.

Segment 1 (60.00–120.00 seconds):
A diagram is shown while the speaker walks through attention mechanisms in transformers.
```

---

## 📦 Project Structure

```
video-lecture-indexer/
├── quickcodec_decoder/     # Frame decoding logic
├── scripts/                # Pipeline utilities
├── videos/                 # Raw and trimmed videos
├── frames/                 # Extracted frames, captions, transcript
├── summaries.txt           # Final output
├── requirements.txt
└── README.md
```

---

## 🤖 Tech Stack

* [Whisper](https://github.com/openai/whisper) for speech-to-text
* [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) for frame captioning
* [Gemma](https://ai.google.dev/gemma) via [Ollama](https://ollama.com) or HuggingFace LLMs
* [FFmpeg](https://ffmpeg.org/) for video/audio processing

---

## 📄 License

MIT — built for learning, studying, and AI-powered lecture review.
