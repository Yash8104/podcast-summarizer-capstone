from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set

from audio_processing import preprocess_audio
from diarization import apply_diarization_to_transcript, diarize_audio
from segmentation import find_topic_boundaries, merge_segments, merge_small_topics
from storage import (
    load_topics,
    load_transcript_segments,
    save_diarization_segments,
    save_original_transcript,
    save_original_transcript_segments,
    save_topic_summaries,
    save_topics,
    save_transcript,
    save_transcript_segments,
)
from summarization import load_summarizer, summarize_topics
from transcription import transcribe_audio

DEFAULT_AUDIO_PATH = Path("joe.mp3")
DEFAULT_TRANSCRIPT_TXT = Path("faster_transcription_output.txt")
DEFAULT_TRANSCRIPT_SEGMENTS = Path("transcript_segments.json")
DEFAULT_TOPIC_SEGMENTS = Path("topic_segments.json")
DEFAULT_DIARIZATION_SEGMENTS = Path("diarization_segments.json")
DEFAULT_TOPIC_SUMMARIES = Path("topic_summaries.json")


@dataclass
class PipelineConfig:
    audio_path: Path
    mode: str = "all"
    model_size: str = "medium"
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    diarization_token: Optional[str] = ""
    diarization_revision: Optional[str] = None
    diarization_device: Optional[str] = None
    segment_method: str = "changepoint"
    window_size: int = 6
    score_quantile: float = 0.8
    min_boundary_gap: int = 5
    smooth_size: int = 5
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_device: Optional[str] = None
    changepoint_penalty: float = 6.0
    changepoint_min_size: int = 4
    min_topic_duration: float = 120.0
    min_topic_turns: int = 8
    summary_model: str = "philschmid/bart-large-cnn-samsum"
    summary_device: Optional[str] = None
    summary_chunk_words: int = 165
    summary_chunk_overlap: int = 45
    summary_min_length: int = 40
    summary_max_length: int = 130
    summary_context_before: int = 1
    summary_context_after: int = 1
    summary_min_words: int = 50
    translate_languages: Optional[Set[str]] = field(default_factory=lambda: {"hi", "mr"})
    translate_target: str = "en"
    transcription_language: Optional[str] = None
    disable_preprocess: bool = False
    preprocess_target_sr: int = 16000
    preprocess_trim_db: Optional[float] = 30.0
    preprocess_highpass: Optional[float] = 60.0
    preprocess_lowpass: Optional[float] = 6000.0
    output_dir: Path = Path("output")
    audio_stem: Optional[str] = None


def _config_to_dict(config: PipelineConfig) -> Dict[str, Any]:
    data = asdict(config)
    data["audio_path"] = str(config.audio_path)
    data["output_dir"] = str(config.output_dir)
    data["translate_languages"] = sorted(list(config.translate_languages)) if config.translate_languages else []
    return data


def _register_file(files: Dict[str, Dict[str, Any]], key: str, path: Path) -> None:
    if path.exists():
        files[key] = {
            "path": str(path),
            "filename": path.name,
            "url": f"/output/{path.name}",
        }


def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = Path(config.audio_path)
    audio_exists = audio_path.exists()

    run_transcription_stage = config.mode in {"all", "transcribe", "transcribe+diarize"}
    run_diarization_stage = config.mode in {"all", "diarize", "transcribe+diarize"}
    run_segmentation_stage = config.mode in {"all", "segment", "segment+summarize"}
    run_summarization_stage = config.mode in {"all", "summarize", "segment+summarize"}

    requires_audio_file = run_transcription_stage or run_diarization_stage
    if requires_audio_file and not audio_exists:
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio_stem = config.audio_stem or audio_path.stem

    transcript_txt_path = output_dir / f"{audio_stem}_transcript.txt"
    transcript_segments_path = output_dir / f"{audio_stem}_transcript_segments.json"
    original_transcript_txt_path = output_dir / f"{audio_stem}_original_transcript.txt"
    original_transcript_segments_path = output_dir / f"{audio_stem}_original_transcript_segments.json"
    topic_segments_path = output_dir / f"{audio_stem}_topic_segments.json"
    diarization_output_path = output_dir / f"{audio_stem}_diarization_segments.json"
    summary_output_path = output_dir / f"{audio_stem}_topic_summaries.json"

    translate_languages = config.translate_languages or set()

    preprocess_temp_path: Optional[Path] = None
    processed_audio_path = audio_path

    if requires_audio_file and not config.disable_preprocess:
        trim_db = config.preprocess_trim_db if (config.preprocess_trim_db is not None and config.preprocess_trim_db >= 0) else None
        highpass = config.preprocess_highpass if (config.preprocess_highpass is not None and config.preprocess_highpass > 0) else None
        lowpass = config.preprocess_lowpass if (config.preprocess_lowpass is not None and config.preprocess_lowpass > 0) else None
        processed_path, created_temp = preprocess_audio(
            str(audio_path),
            target_sr=config.preprocess_target_sr,
            trim_db=trim_db,
            highpass_hz=highpass,
            lowpass_hz=lowpass,
        )
        processed_audio_path = processed_path
        if created_temp:
            preprocess_temp_path = processed_audio_path
            print(f"Preprocessed audio written to {processed_audio_path}")

    transcript_segments: Optional[list[dict]] = None
    topic_segments: Optional[list[dict]] = None

    try:
        if run_transcription_stage:
            transcript_segments = run_transcription(
                processed_audio_path,
                transcript_txt_path,
                transcript_segments_path,
                original_transcript_txt_path,
                original_transcript_segments_path,
                model_size=config.model_size,
                translate_languages=translate_languages,
                translate_target_language=config.translate_target,
                language_override=config.transcription_language,
            )

        if run_diarization_stage:
            if transcript_segments is None:
                transcript_segments = load_transcript_segments(str(transcript_segments_path))
                print(f"Loaded {len(transcript_segments)} transcript segments for diarization.")
            transcript_segments = run_diarization(
                processed_audio_path,
                transcript_segments,
                transcript_txt_path,
                transcript_segments_path,
                original_transcript_txt_path,
                original_transcript_segments_path,
                diarization_output_path,
                model_id=config.diarization_model,
                auth_token=config.diarization_token,
                revision=config.diarization_revision,
                device=config.diarization_device,
            )

        if run_segmentation_stage:
            if transcript_segments is None:
                transcript_segments = load_segments_for_segmentation(
                    transcript_segments_path,
                    transcript_txt_path,
                )
            topic_segments = run_segmentation(
                transcript_segments,
                topic_segments_path,
                method_preference=config.segment_method,
                window_size=config.window_size,
                score_quantile=config.score_quantile,
                min_gap=config.min_boundary_gap,
                smooth_size=config.smooth_size,
                embedding_model=config.embedding_model,
                embedding_device=config.embedding_device,
                changepoint_penalty=config.changepoint_penalty,
                changepoint_min_size=config.changepoint_min_size,
                min_topic_duration=config.min_topic_duration,
                min_topic_turns=config.min_topic_turns,
            )

        if run_summarization_stage:
            if topic_segments is None:
                topic_segments = load_topics(str(topic_segments_path))
                print(f"Loaded {len(topic_segments)} topic segments from {topic_segments_path}.")

            run_summarization(
                topic_segments,
                summary_output_path,
                model_name=config.summary_model,
                device=config.summary_device,
                chunk_word_limit=config.summary_chunk_words,
                chunk_overlap=config.summary_chunk_overlap,
                summary_min_length=config.summary_min_length,
                summary_max_length=config.summary_max_length,
                summary_context_before=config.summary_context_before,
                summary_context_after=config.summary_context_after,
                summary_min_words=config.summary_min_words,
            )
    finally:
        if preprocess_temp_path and preprocess_temp_path.exists():
            preprocess_temp_path.unlink(missing_ok=True)

    files: Dict[str, Dict[str, Any]] = {}
    _register_file(files, "transcript", transcript_txt_path)
    _register_file(files, "transcript_segments", transcript_segments_path)
    _register_file(files, "original_transcript", original_transcript_txt_path)
    _register_file(files, "original_transcript_segments", original_transcript_segments_path)
    _register_file(files, "topic_segments", topic_segments_path)
    _register_file(files, "topic_summaries", summary_output_path)
    _register_file(files, "diarization_segments", diarization_output_path)

    data: Dict[str, Any] = {}
    if transcript_txt_path.exists():
        data["transcript_text"] = transcript_txt_path.read_text(encoding="utf-8")
    if transcript_segments_path.exists():
        with open(transcript_segments_path, "r", encoding="utf-8") as f:
            data["transcript_segments"] = json.load(f)
    if topic_segments_path.exists():
        with open(topic_segments_path, "r", encoding="utf-8") as f:
            data["topic_segments"] = json.load(f)
    if summary_output_path.exists():
        with open(summary_output_path, "r", encoding="utf-8") as f:
            data["topic_summaries"] = json.load(f)
    if diarization_output_path.exists():
        with open(diarization_output_path, "r", encoding="utf-8") as f:
            data["diarization_segments"] = json.load(f)

    result = {
        "audio_stem": audio_stem,
        "mode": config.mode,
        "files": files,
        "data": data,
        "config": _config_to_dict(config),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    return result


def run_transcription(
    audio_path: Path,
    transcript_txt_path: Path,
    transcript_segments_path: Path,
    original_transcript_txt_path: Path,
    original_transcript_segments_path: Path,
    *,
    model_size: str,
    translate_languages: set[str] | None,
    translate_target_language: str,
    language_override: str | None,
) -> list[dict]:
    print(f"Transcribing {audio_path} with Whisper model '{model_size}'...")
    segments, info = transcribe_audio(
        str(audio_path),
        model_size=model_size,
        auto_translate_languages=translate_languages,
        translate_target_language=translate_target_language,
        language_override=language_override,
    )
    language = info.get("language")
    language_prob = info.get("language_probability")
    if language:
        if language_prob is not None:
            print(f"Detected language: {language} (p={language_prob:.2f})")
        else:
            print(f"Detected language: {language}")
    if info.get("translated"):
        source = info.get("translated_from")
        target = info.get("translated_to") or "en"
        print(f"Auto-translated transcript from {source} to {target}.")

    save_transcript(segments, str(transcript_txt_path))
    save_transcript_segments(segments, str(transcript_segments_path))
    save_original_transcript(segments, str(original_transcript_txt_path))
    save_original_transcript_segments(segments, str(original_transcript_segments_path))

    for segment in segments:
        segment.pop("original_text", None)
    print(
        f"Saved transcript text to {transcript_txt_path} and "
        f"segment metadata to {transcript_segments_path}."
    )
    return segments


def _segments_from_text(transcript_txt_path: Path) -> list[dict]:
    if not transcript_txt_path.exists():
        return []

    fallback_segments: list[dict] = []
    with open(transcript_txt_path, "r") as file:
        for line in file:
            text = line.strip()
            if text:
                fallback_segments.append({"text": text, "start": None, "end": None})
    return fallback_segments


def run_diarization(
    audio_path: Path,
    transcript_segments: Sequence[dict],
    transcript_txt_path: Path,
    transcript_segments_path: Path,
    original_transcript_txt_path: Path,
    original_transcript_segments_path: Path,
    diarization_output_path: Path,
    *,
    model_id: str,
    auth_token: str | None,
    revision: str | None,
    device: str | None,
) -> list[dict]:
    token = auth_token or os.environ.get("PYANNOTE_AUTH_TOKEN")
    if token is None:
        print("No diarization token provided; attempting to load the pipeline without authentication.")

    print(
        f"Running speaker diarization with model '{model_id}' "
        f"on device '{device or 'auto'}'..."
    )
    diarization_segments = diarize_audio(
        str(audio_path),
        model_id=model_id,
        auth_token=token,
        revision=revision,
        device=device,
    )
    print(f"Detected {len(diarization_segments)} diarization segments.")

    enhanced_segments = apply_diarization_to_transcript(transcript_segments, diarization_segments)
    save_transcript(enhanced_segments, str(transcript_txt_path))
    save_transcript_segments(enhanced_segments, str(transcript_segments_path))
    save_original_transcript(enhanced_segments, str(original_transcript_txt_path))
    save_original_transcript_segments(enhanced_segments, str(original_transcript_segments_path))
    save_diarization_segments(diarization_segments, str(diarization_output_path))
    print(
        f"Saved diarization-aware transcript to {transcript_txt_path} and metadata to "
        f"{transcript_segments_path}. Raw diarization segments stored in {diarization_output_path}."
    )
    return list(enhanced_segments)


def load_segments_for_segmentation(
    transcript_segments_path: Path,
    transcript_txt_path: Path,
) -> list[dict]:
    try:
        segments = load_transcript_segments(str(transcript_segments_path))
        print(f"Loaded {len(segments)} segments from {transcript_segments_path}.")
        return segments
    except FileNotFoundError:
        print(
            f"Segment metadata file {transcript_segments_path} not found. "
            "Attempting to fall back to plain-text transcript."
        )
    except ValueError as error:
        print(
            f"Could not parse {transcript_segments_path} ({error}). "
            "Attempting to fall back to plain-text transcript."
        )

    fallback_segments = _segments_from_text(transcript_txt_path)
    if not fallback_segments:
        raise FileNotFoundError(
            "No transcript segments available. Run transcription first or supply "
            f"a transcript JSON at {transcript_segments_path}."
        )

    print(
        f"Loaded {len(fallback_segments)} transcript lines from {transcript_txt_path}. "
        "Segment timings will be omitted in the output."
    )
    return fallback_segments




def run_segmentation(
    transcript_segments: Sequence[dict],
    topic_segments_path: Path,
    *,
    method_preference: str = "embeddings",
    window_size: int = 6,
    score_quantile: float = 0.8,
    min_gap: int = 5,
    smooth_size: int = 5,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    embedding_device: str | None = None,
    changepoint_penalty: float = 6.0,
    changepoint_min_size: int = 4,
    min_topic_duration: float = 120.0,
    min_topic_turns: int = 8,
) -> list[dict]:
    config_parts = [
        f"method={method_preference}",
        f"window={window_size}",
        f"quantile={score_quantile}",
        f"min_gap={min_gap}",
        f"smooth={smooth_size}",
        f"embedding_model={embedding_model}",
        f"embedding_device={embedding_device}",
        f"min_topic_duration={min_topic_duration}",
        f"min_topic_turns={min_topic_turns}",
    ]
    if method_preference == "changepoint":
        config_parts.append(f"penalty={changepoint_penalty}")
        config_parts.append(f"min_size={changepoint_min_size}")
    print("Segmenting transcript (" + ", ".join(str(part) for part in config_parts) + ")")

    boundaries = find_topic_boundaries(
        transcript_segments,
        method=method_preference,
        window_size=window_size,
        score_quantile=score_quantile,
        min_gap=min_gap,
        smooth_size=smooth_size,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
        changepoint_penalty=changepoint_penalty,
        changepoint_min_size=changepoint_min_size,
    )
    topic_segments = merge_segments(transcript_segments, boundaries)
    topic_segments = merge_small_topics(
        topic_segments,
        min_duration=min_topic_duration,
        min_turns=min_topic_turns,
    )
    save_topics(topic_segments, str(topic_segments_path))

    print(f"\nDetected {len(topic_segments)} topic segments. Saved to {topic_segments_path}.")
    for idx, topic in enumerate(topic_segments, start=1):
        start = topic.get("start")
        end = topic.get("end")
        span = (
            f"[{start:.2f}s -> {end:.2f}s]"
            if start is not None and end is not None
            else "[no timing]"
        )
        preview = topic["text"][:120]
        suffix = "..." if len(topic["text"]) > 120 else ""
        print(f"Segment {idx}: {span} {preview}{suffix}")

    return topic_segments


def run_summarization(
    topic_segments: Sequence[dict],
    summary_output_path: Path,
    *,
    model_name: str,
    device: str | None,
    chunk_word_limit: int,
    chunk_overlap: int,
    summary_min_length: int,
    summary_max_length: int,
    summary_context_before: int,
    summary_context_after: int,
    summary_min_words: int,
) -> list[dict]:
    print(
        "Summarizing topic segments "
        f"(model={model_name}, device={device}, chunk_words={chunk_word_limit}, "
        f"overlap={chunk_overlap}, min_len={summary_min_length}, max_len={summary_max_length}, "
        f"context_before={summary_context_before}, context_after={summary_context_after}, "
        f"min_words={summary_min_words})"
    )

    if chunk_word_limit <= 0:
        print("Chunk word limit is non-positive; summarizer will process each segment as a single chunk.")
    if chunk_overlap < 0:
        print(f"Chunk overlap {chunk_overlap} is negative. Setting overlap to 0.")
        chunk_overlap = 0
    if chunk_overlap >= chunk_word_limit:
        print(
            f"Chunk overlap {chunk_overlap} is greater than or equal to chunk limit {chunk_word_limit}. "
            f"Clamping overlap to {max(0, chunk_word_limit // 4)}."
        )
        chunk_overlap = max(0, chunk_word_limit // 4)

    summarizer = load_summarizer(model_name, device)
    topic_summaries = summarize_topics(
        topic_segments,
        summarizer,
        chunk_word_limit=chunk_word_limit,
        chunk_overlap=chunk_overlap,
        summary_min_length=summary_min_length,
        summary_max_length=summary_max_length,
        context_before=summary_context_before,
        context_after=summary_context_after,
        min_words_for_summary=summary_min_words,
    )
    save_topic_summaries(topic_summaries, str(summary_output_path))

    print(f"\nGenerated {len(topic_summaries)} summaries. Saved to {summary_output_path}.")
    for idx, topic in enumerate(topic_summaries, start=1):
        start = topic.get("start")
        end = topic.get("end")
        span = f"[{start:.2f}s -> {end:.2f}s]" if start is not None and end is not None else "[no timing]"
        primary_speaker = topic.get("primary_speaker")
        if primary_speaker:
            span = f"{span} ({primary_speaker})"
        preview = topic["summary"][:160]
        suffix = "..." if len(topic["summary"]) > 160 else ""
        print(f"Summary {idx}: {span} {preview}{suffix}")
        speaker_summaries = topic.get("speaker_summaries") or {}
        for speaker, speaker_summary in speaker_summaries.items():
            if not speaker_summary:
                continue
            speaker_preview = speaker_summary[:140]
            speaker_suffix = "..." if len(speaker_summary) > 140 else ""
            print(f"  - {speaker}: {speaker_preview}{speaker_suffix}")

    return topic_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Podcast transcription, diarization, segmentation, and summarization helper.")
    parser.add_argument(
        "--mode",
        choices=(
            "all",
            "transcribe",
            "segment",
            "summarize",
            "segment+summarize",
            "diarize",
            "transcribe+diarize",
        ),
        default="all",
        help=(
            "Select which pipeline stages to run. "
            "'segment+summarize' runs segmentation followed by summarization."
        ),
    )
    parser.add_argument(
        "--audio",
        default=str(DEFAULT_AUDIO_PATH),
        help="Path to the audio file for transcription.",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model size for transcription (tiny, base, small, etc.).",
    )
    parser.add_argument(
        "--transcript-text",
        default=str(DEFAULT_TRANSCRIPT_TXT),
        help="Where to write/read the plain-text transcript.",
    )
    parser.add_argument(
        "--transcript-segments",
        default=str(DEFAULT_TRANSCRIPT_SEGMENTS),
        help="Where to write/read the JSON transcript segments.",
    )
    parser.add_argument(
        "--topic-output",
        default=str(DEFAULT_TOPIC_SEGMENTS),
        help="Where to write the JSON topic segments.",
    )
    parser.add_argument(
        "--diarization-output",
        default=str(DEFAULT_DIARIZATION_SEGMENTS),
        help="Where to write the JSON diarization segments.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(DEFAULT_TOPIC_SUMMARIES),
        help="Where to write the JSON topic summaries.",
    )
    parser.add_argument(
        "--translate-languages",
        default="hi,mr",
        help="Comma-separated language codes to auto-translate into English (leave empty to disable).",
    )
    parser.add_argument(
        "--translate-target",
        default="en",
        help="Target language code for translation (default: en).",
    )
    parser.add_argument(
        "--transcription-language",
        default=None,
        help="Force Whisper source language (e.g., hi). Leave empty for auto-detect.",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Run speaker diarization on the transcript segments.",
    )
    parser.add_argument(
        "--segment-method",
        choices=("embeddings", "lexical", "changepoint"),
        default="embeddings",
        help="Segmentation strategy to use.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=6,
        help="Number of neighbouring segments to consider on each side when scoring boundaries.",
    )
    parser.add_argument(
        "--score-quantile",
        type=float,
        default=0.8,
        help="Quantile (0-1) on the change scores to pick boundary thresholds. Higher gives fewer segments.",
    )
    parser.add_argument(
        "--min-boundary-gap",
        type=int,
        default=5,
        help="Minimum number of segments to keep between detected boundaries.",
    )
    parser.add_argument(
        "--smooth-size",
        type=int,
        default=5,
        help="Width of the smoothing window applied to change scores before thresholding.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Sentence-transformers model to use for embedding-based segmentation.",
    )
    parser.add_argument(
        "--embedding-device",
        default=None,
        help="Torch device string to run embeddings on (e.g. 'cpu', 'mps', 'cuda'). Defaults to library auto choice.",
    )
    parser.add_argument(
        "--diarization-model",
        default="pyannote/speaker-diarization-community-1",
        help="Pyannote diarization pipeline to use for speaker attribution.",
    )
    parser.add_argument(
        "--diarization-token",
        default="",
        help="Hugging Face access token with permissions for the diarization model.",
    )
    parser.add_argument(
        "--diarization-revision",
        default=None,
        help="Optional model revision/tag for diarization checkpoints.",
    )
    parser.add_argument(
        "--diarization-device",
        default=None,
        help="Device to run diarization on (e.g. 'cpu', 'cuda', 'mps').",
    )
    parser.add_argument(
        "--changepoint-penalty",
        type=float,
        default=6.0,
        help="Penalty strength for changepoint detection (higher => fewer segments).",
    )
    parser.add_argument(
        "--changepoint-min-size",
        type=int,
        default=4,
        help="Minimum number of transcript segments between changepoints.",
    )
    parser.add_argument(
        "--min-topic-duration",
        type=float,
        default=120.0,
        help="Minimum duration (in seconds) for a topic segment; shorter segments are merged with neighbours.",
    )
    parser.add_argument(
        "--min-topic-turns",
        type=int,
        default=8,
        help="Minimum number of transcript turns per topic; smaller groups are merged with neighbours.",
    )
    parser.add_argument(
        "--summary-model",
        default="philschmid/bart-large-cnn-samsum",
        help="Summarization model checkpoint to use.",
    )
    parser.add_argument(
        "--summary-device",
        default=None,
        help="Device string for the summarization pipeline (e.g. 'cpu', 'mps', 'cuda').",
    )
    parser.add_argument(
        "--summary-chunk-words",
        type=int,
        default=165,
        help="Approximate number of words per chunk fed into the summarizer.",
    )
    parser.add_argument(
        "--summary-chunk-overlap",
        type=int,
        default=45,
        help="Word overlap between successive chunks when summarizing long segments.",
    )
    parser.add_argument(
        "--summary-min-length",
        type=int,
        default=40,
        help="Minimum token length for generated summaries.",
    )
    parser.add_argument(
        "--summary-max-length",
        type=int,
        default=130,
        help="Maximum token length for generated summaries.",
    )
    parser.add_argument(
        "--summary-context-before",
        type=int,
        default=1,
        help="Number of preceding topic segments to include as context when summarizing.",
    )
    parser.add_argument(
        "--summary-context-after",
        type=int,
        default=1,
        help="Number of following topic segments to include as context when summarizing.",
    )
    parser.add_argument(
        "--summary-min-words",
        type=int,
        default=50,
        help="Only run the abstractive summarizer when the input has at least this many words; otherwise fall back to the original text.",
    )
    parser.add_argument(
        "--disable-preprocess",
        action="store_true",
        help="Skip audio preprocessing (noise reduction, trimming).",
    )
    parser.add_argument(
        "--preprocess-target-sr",
        type=int,
        default=16000,
        help="Sample rate to resample audio during preprocessing.",
    )
    parser.add_argument(
        "--preprocess-trim-db",
        type=float,
        default=30.0,
        help="Remove leading/trailing sections below this dB threshold (set negative to disable).",
    )
    parser.add_argument(
        "--preprocess-highpass",
        type=float,
        default=60.0,
        help="High-pass filter cutoff in Hz (set 0 to disable).",
    )
    parser.add_argument(
        "--preprocess-lowpass",
        type=float,
        default=6000.0,
        help="Low-pass filter cutoff in Hz (set 0 to disable).",
    )
    return parser.parse_args()


def _legacy_main() -> None:
    args = parse_args()

    audio_path = Path(args.audio)
    audio_stem = audio_path.stem
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    def resolve_output_path(arg_value: str, default_path: Path, filename: str) -> Path:
        candidate = Path(arg_value)
        if candidate == default_path:
            return output_dir / filename
        return candidate

    transcript_txt_path = resolve_output_path(
        args.transcript_text,
        DEFAULT_TRANSCRIPT_TXT,
        f"{audio_stem}_transcript.txt",
    )
    transcript_segments_path = resolve_output_path(
        args.transcript_segments,
        DEFAULT_TRANSCRIPT_SEGMENTS,
        f"{audio_stem}_transcript_segments.json",
    )
    topic_segments_path = resolve_output_path(
        args.topic_output,
        DEFAULT_TOPIC_SEGMENTS,
        f"{audio_stem}_topic_segments.json",
    )
    diarization_output_path = resolve_output_path(
        args.diarization_output,
        DEFAULT_DIARIZATION_SEGMENTS,
        f"{audio_stem}_diarization_segments.json",
    )
    summary_output_path = resolve_output_path(
        args.summary_output,
        DEFAULT_TOPIC_SUMMARIES,
        f"{audio_stem}_topic_summaries.json",
    )
    original_transcript_txt_path = output_dir / f"{audio_stem}_original_transcript.txt"
    original_transcript_segments_path = output_dir / f"{audio_stem}_original_transcript_segments.json"

    translate_languages = {
        lang.strip().lower()
        for lang in args.translate_languages.split(",")
        if lang.strip()
    }

    preprocess_temp_path: Path | None = None
    processed_audio_path = audio_path
    if not args.disable_preprocess:
        trim_db = args.preprocess_trim_db if args.preprocess_trim_db >= 0 else None
        highpass = args.preprocess_highpass if args.preprocess_highpass > 0 else None
        lowpass = args.preprocess_lowpass if args.preprocess_lowpass > 0 else None
        processed_path, created_temp = preprocess_audio(
            str(audio_path),
            target_sr=args.preprocess_target_sr,
            trim_db=trim_db,
            highpass_hz=highpass,
            lowpass_hz=lowpass,
        )
        processed_audio_path = processed_path
        if created_temp:
            preprocess_temp_path = processed_audio_path
            print(f"Preprocessed audio written to {processed_audio_path}")
    else:
        processed_audio_path = audio_path

    run_transcription_stage = args.mode in {"all", "transcribe", "transcribe+diarize"}
    run_diarization_stage = args.diarize or args.mode in {"all", "diarize", "transcribe+diarize"}
    run_segmentation_stage = args.mode in {"all", "segment", "segment+summarize"}
    run_summarization_stage = args.mode in {"all", "summarize", "segment+summarize"}

    transcript_segments: list[dict] | None = None
    topic_segments: list[dict] | None = None

    try:
        if run_transcription_stage:
            transcript_segments = run_transcription(
                processed_audio_path,
                transcript_txt_path,
                transcript_segments_path,
                original_transcript_txt_path,
                original_transcript_segments_path,
                model_size=args.model,
                translate_languages=translate_languages,
                translate_target_language=args.translate_target,
                language_override=args.transcription_language,
            )

        if run_diarization_stage:
            if transcript_segments is None:
                transcript_segments = load_transcript_segments(str(transcript_segments_path))
                print(f"Loaded {len(transcript_segments)} transcript segments for diarization.")
            transcript_segments = run_diarization(
                processed_audio_path,
                transcript_segments,
                transcript_txt_path,
                transcript_segments_path,
                original_transcript_txt_path,
                original_transcript_segments_path,
                diarization_output_path,
                model_id=args.diarization_model,
                auth_token=args.diarization_token,
                revision=args.diarization_revision,
                device=args.diarization_device,
            )

        if run_segmentation_stage:
            if transcript_segments is None:
                transcript_segments = load_segments_for_segmentation(
                    transcript_segments_path,
                    transcript_txt_path,
                )
            topic_segments = run_segmentation(
                transcript_segments,
                topic_segments_path,
                method_preference=args.segment_method,
                window_size=args.window_size,
                score_quantile=args.score_quantile,
                min_gap=args.min_boundary_gap,
                smooth_size=args.smooth_size,
                embedding_model=args.embedding_model,
                embedding_device=args.embedding_device,
                changepoint_penalty=args.changepoint_penalty,
                changepoint_min_size=args.changepoint_min_size,
                min_topic_duration=args.min_topic_duration,
                min_topic_turns=args.min_topic_turns,
            )

        if run_summarization_stage:
            if topic_segments is None:
                try:
                    topic_segments = load_topics(str(topic_segments_path))
                    print(f"Loaded {len(topic_segments)} topic segments from {topic_segments_path}.")
                except FileNotFoundError as error:
                    raise FileNotFoundError(
                        f"No topic segments found at {topic_segments_path}. "
                        "Run segmentation first or provide a valid topic segments JSON."
                    ) from error
                except ValueError as error:
                    raise ValueError(
                        f"Failed to parse topic segments from {topic_segments_path}: {error}"
                    ) from error

            run_summarization(
                topic_segments,
                summary_output_path,
                model_name=args.summary_model,
                device=args.summary_device,
                chunk_word_limit=args.summary_chunk_words,
                chunk_overlap=args.summary_chunk_overlap,
                summary_min_length=args.summary_min_length,
                summary_max_length=args.summary_max_length,
                summary_context_before=args.summary_context_before,
                summary_context_after=args.summary_context_after,
                summary_min_words=args.summary_min_words,
            )
    finally:
        if preprocess_temp_path and preprocess_temp_path.exists():
            preprocess_temp_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()

    translate_languages_raw = {
        lang.strip().lower()
        for lang in args.translate_languages.split(",")
        if lang.strip()
    }
    translate_languages = translate_languages_raw if translate_languages_raw else None

    config_kwargs: Dict[str, Any] = {
        "audio_path": Path(args.audio),
        "mode": args.mode,
        "model_size": args.model,
        "diarization_model": args.diarization_model,
        "diarization_token": args.diarization_token or None,
        "diarization_revision": args.diarization_revision or None,
        "diarization_device": args.diarization_device or None,
        "segment_method": args.segment_method,
        "window_size": args.window_size,
        "score_quantile": args.score_quantile,
        "min_boundary_gap": args.min_boundary_gap,
        "smooth_size": args.smooth_size,
        "embedding_model": args.embedding_model,
        "embedding_device": args.embedding_device or None,
        "changepoint_penalty": args.changepoint_penalty,
        "changepoint_min_size": args.changepoint_min_size,
        "min_topic_duration": args.min_topic_duration,
        "min_topic_turns": args.min_topic_turns,
        "summary_model": args.summary_model,
        "summary_device": args.summary_device or None,
        "summary_chunk_words": args.summary_chunk_words,
        "summary_chunk_overlap": args.summary_chunk_overlap,
        "summary_min_length": args.summary_min_length,
        "summary_max_length": args.summary_max_length,
        "summary_context_before": args.summary_context_before,
        "summary_context_after": args.summary_context_after,
        "summary_min_words": args.summary_min_words,
        "translate_target": args.translate_target,
        "transcription_language": args.transcription_language or None,
        "disable_preprocess": args.disable_preprocess,
        "preprocess_target_sr": args.preprocess_target_sr,
        "preprocess_trim_db": args.preprocess_trim_db if args.preprocess_trim_db >= 0 else None,
        "preprocess_highpass": args.preprocess_highpass if args.preprocess_highpass > 0 else None,
        "preprocess_lowpass": args.preprocess_lowpass if args.preprocess_lowpass > 0 else None,
    }
    if translate_languages is not None:
        config_kwargs["translate_languages"] = translate_languages

    config = PipelineConfig(**config_kwargs)
    result = run_pipeline(config)

    print("Processing complete. Outputs saved in the 'output' directory.")
    for key, meta in result["files"].items():
        print(f"  {key}: {meta['path']}")


if __name__ == "__main__":
    main()
