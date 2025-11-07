# Podcast Intelligence Studio

This project stitches together audio ingestion, transcription, diarization, topic segmentation, and summarization into a single local workflow. You can upload audio or paste Spotify/Youtube links, run the full pipeline, and browse the outputs in a FastAPI-powered dashboard.

## Requirements

- macOS / Linux with Python **3.11**
- [Homebrew](https://brew.sh/) (for macOS)
- FFmpeg for audio I/O

```bash
brew install ffmpeg
```

## Python environment

It’s best to keep everything inside a virtual environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the repository root (or export these variables in your shell):

```
PYANNOTE_AUTH_TOKEN=hf_xxx                     # Hugging Face token with pyannote access
PODCASTINDEX_API_KEY=your_podcastindex_key
PODCASTINDEX_API_SECRET=your_podcastindex_secret
```

These values are automatically loaded by both the CLI (`test.py`) and the FastAPI backend. If any of them are missing, the associated feature (e.g., link ingestion) will be skipped with a helpful error.

## Running the FastAPI dashboard

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000` to open the dashboard. Features:

- Upload MP3s or paste Spotify / YouTube links
- Tweak transcription, diarization, segmentation, and summarization parameters
- Run any stage independently (e.g., summarize only)
- Review transcripts, diarization, topics, and summaries from previous runs

## CLI usage

The older CLI entry point is still available for scripting and automation.

Full pipeline example:

```bash
python test.py --mode all \
  --audio my_episode.mp3 \
  --model medium \
  --segment-method changepoint \
  --summary-model philschmid/bart-large-cnn-samsum
```

Summarize-only (reusing stored topics):

```bash
python test.py --mode summarize \
  --audio output/my_episode_transcript.txt \
  --topic-output output/my_episode_topic_segments.json \
  --summary-model facebook/bart-large-cnn
```

Key CLI flags are documented in `python test.py --help`.

## Output structure

All artifacts live under `output/` with the audio stem as a prefix, e.g.:

```
output/<stem>_transcript.txt
output/<stem>_transcript_segments.json
output/<stem>_original_transcript.txt
output/<stem>_diarization_segments.json
output/<stem>_topic_segments.json
output/<stem>_topic_summaries.json
```

When you ingest Spotify/YouTube links, the downloaded MP3s are cached in `output/link_downloads/` with metadata stored for reuse.

## Troubleshooting

- **Pyannote token errors**: ensure `PYANNOTE_AUTH_TOKEN` is valid and the `pyannote.audio` dependencies are installed.
- **Link downloads fail**: verify FFmpeg is installed and the `PODCASTINDEX_*` keys exist (Spotify → RSS). YouTube downloads require `yt-dlp` in the environment.
- **Summaries look noisy**: adjust `summary_chunk_words`, `summary_chunk_overlap`, and `min_topic_duration` from the dashboard, then rerun “summarize” mode without re-uploading audio.

## License

MIT License (see `LICENSE` if provided) – feel free to adapt for your own podcast tooling.
