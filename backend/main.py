from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from test import PipelineConfig, _config_to_dict, run_pipeline
from storage import (
    save_diarization_segments,
    save_original_transcript,
    save_original_transcript_segments,
    save_topic_summaries,
    save_topics,
    save_transcript,
    save_transcript_segments,
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
METADATA_SUFFIX = "_metadata.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

FILENAME_CLEANER = re.compile(r"[^A-Za-z0-9_.-]")

app = FastAPI(title="Podcast Pipeline", version="1.0.0")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _default_config_dict() -> Dict[str, Any]:
    dummy = PipelineConfig(audio_path=Path("example.mp3"))
    defaults = _config_to_dict(dummy)
    defaults.pop("audio_path", None)
    defaults["audio_path"] = None
    return defaults


def _parse_translate_languages(raw: Any) -> Optional[set[str]]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple, set)):
        languages = {str(item).strip().lower() for item in raw if str(item).strip()}
        return languages or None
    text = str(raw).strip()
    if not text:
        return None
    languages = {part.strip().lower() for part in text.split(",") if part.strip()}
    return languages or None


INT_FIELDS = {
    "window_size",
    "min_boundary_gap",
    "smooth_size",
    "changepoint_min_size",
    "min_topic_turns",
    "summary_chunk_words",
    "summary_chunk_overlap",
    "summary_min_length",
    "summary_max_length",
    "summary_context_before",
    "summary_context_after",
    "summary_min_words",
    "preprocess_target_sr",
}

FLOAT_FIELDS = {
    "score_quantile",
    "changepoint_penalty",
    "min_topic_duration",
    "preprocess_trim_db",
    "preprocess_highpass",
    "preprocess_lowpass",
}

BOOL_FIELDS = {"disable_preprocess"}

LISTABLE_FIELDS = {"translate_languages"}

TextLike = str


def _coerce_value(field: str, value: Any) -> Any:
    if value is None or value == "":
        return None
    if field in BOOL_FIELDS:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        return text in {"true", "1", "yes", "on"}
    if field in INT_FIELDS:
        try:
            return int(value)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=f"Invalid integer for '{field}': {value}") from error
    if field in FLOAT_FIELDS:
        try:
            return float(value)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=f"Invalid float for '{field}': {value}") from error
    if field in LISTABLE_FIELDS:
        languages = _parse_translate_languages(value)
        return languages
    return value


def _load_metadata_files() -> List[Path]:
    return sorted(
        OUTPUT_DIR.glob(f"*{METADATA_SUFFIX}"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )


def _public_url(path: Path) -> Optional[str]:
    try:
        relative = path.resolve().relative_to(OUTPUT_DIR)
    except ValueError:
        return None
    return f"/output/{relative.as_posix()}"


def _save_metadata(result: Dict[str, Any]) -> Path:
    audio_stem = result.get("audio_stem") or "audio"
    metadata = {
        "audio_stem": audio_stem,
        "created_at": result.get("created_at"),
        "mode": result.get("mode"),
        "files": result.get("files", {}),
        "config": result.get("config", {}),
        "source_audio": result.get("source_audio"),
    }
    path = OUTPUT_DIR / f"{audio_stem}{METADATA_SUFFIX}"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def _collect_run(audio_stem: str) -> Dict[str, Any]:
    metadata_path = OUTPUT_DIR / f"{audio_stem}{METADATA_SUFFIX}"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail=f"No run metadata for '{audio_stem}'.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    files_info = metadata.get("files", {})
    data: Dict[str, Any] = {}
    if "transcript" in files_info:
        transcript_path = Path(files_info["transcript"]["path"])
        if transcript_path.exists():
            data["transcript_text"] = transcript_path.read_text(encoding="utf-8")
    for key in ("transcript_segments", "topic_segments", "topic_summaries", "diarization_segments"):
        info = files_info.get(key)
        if not info:
            continue
        json_path = Path(info["path"])
        if json_path.exists():
            try:
                data[key] = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

    metadata["data"] = data
    return metadata


def _resolve_audio_path(path_text: str) -> Path:
    candidate = Path(path_text).expanduser()
    if not candidate.is_absolute():
        candidate = (BASE_DIR / candidate).resolve()
    if not str(candidate).startswith(str(BASE_DIR)):
        raise HTTPException(status_code=400, detail="Audio path must be within the project directory.")
    if not candidate.exists():
        raise HTTPException(status_code=400, detail=f"Audio file not found: {candidate}")
    return candidate


def _clean_filename(name: str, fallback: str = "audio") -> str:
    name = name.strip().replace(" ", "_")
    cleaned = FILENAME_CLEANER.sub("_", name)
    cleaned = cleaned.strip("._") or fallback
    return cleaned


def _store_uploaded_file(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix
    if not suffix:
        suffix = ".mp3"
    original_stem = Path(upload.filename or "").stem or "audio"
    cleaned_stem = _clean_filename(original_stem)

    destination = UPLOAD_DIR / f"{cleaned_stem}{suffix}"
    counter = 1
    while destination.exists():
        destination = UPLOAD_DIR / f"{cleaned_stem}_{counter}{suffix}"
        counter += 1

    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return destination


def _build_config(audio_path: Path, overrides: Dict[str, Any]) -> PipelineConfig:
    kwargs: Dict[str, Any] = {"audio_path": audio_path}
    for field, value in overrides.items():
        if field == "audio_path":
            continue
        coerced = _coerce_value(field, value)
        if coerced is None:
            continue
        kwargs[field] = coerced
    if "translate_languages" in overrides and kwargs.get("translate_languages") is None:
        kwargs["translate_languages"] = None
    return PipelineConfig(**kwargs)


def _map_speaker_name(name: str | None, replacements: Dict[str, str]) -> str | None:
    if not name:
        return name
    return replacements.get(name.lower(), name)


def _replace_text_casefold(text: str, replacements: Dict[str, str]) -> str:
    if not text or not replacements:
        return text
    pattern = re.compile("|".join(re.escape(key) for key in replacements.keys()), flags=re.IGNORECASE)

    def _substitute(match: re.Match[str]) -> str:
        mapped = replacements.get(match.group(0).lower())
        return mapped if mapped is not None else match.group(0)

    return pattern.sub(_substitute, text)


def _rename_text_line(line: str, replacements: Dict[str, str]) -> str:
    if ":" not in line:
        return line
    label, rest = line.split(":", 1)
    label = label.strip()
    replacement = replacements.get(label.lower())
    if not replacement:
        return line
    return f"{replacement}: {rest.lstrip()}"


def _apply_speaker_map_to_topic(topic: Dict[str, Any], replacements: Dict[str, str]) -> Dict[str, Any]:
    updated = dict(topic)
    turns = updated.get("turns") or []
    new_turns: List[Dict[str, Any]] = []
    for turn in turns:
        copy = dict(turn)
        speaker = copy.get("speaker")
        if speaker:
            copy["speaker"] = _map_speaker_name(speaker, replacements)
        new_turns.append(copy)
    updated["turns"] = new_turns

    primary = updated.get("primary_speaker")
    if primary:
        updated["primary_speaker"] = _map_speaker_name(primary, replacements)

    speaker_counts = updated.get("speaker_counts") or {}
    if isinstance(speaker_counts, dict):
        new_counts: Dict[str, int] = {}
        for raw_speaker, count in speaker_counts.items():
            mapped = _map_speaker_name(raw_speaker, replacements) or raw_speaker
            new_counts[mapped] = new_counts.get(mapped, 0) + int(count)
        updated["speaker_counts"] = new_counts

    if new_turns:
        segments = []
        for turn in new_turns:
            text = (turn.get("text") or "").strip()
            speaker = turn.get("speaker")
            if speaker:
                segments.append(f"{speaker}: {text}" if text else str(speaker))
            elif text:
                segments.append(text)
        updated["text"] = " ".join(segment for segment in segments if segment)
    else:
        text = updated.get("text") or ""
        updated["text"] = _replace_text_casefold(text, replacements)

    return updated


def _apply_speaker_map_to_summaries(summary: Dict[str, Any], replacements: Dict[str, str]) -> Dict[str, Any]:
    updated = dict(summary)
    primary = updated.get("primary_speaker")
    if primary:
        updated["primary_speaker"] = _map_speaker_name(primary, replacements)

    speaker_summaries = updated.get("speaker_summaries") or {}
    if isinstance(speaker_summaries, dict):
        new_summaries: Dict[str, Any] = {}
        for raw_speaker, text in speaker_summaries.items():
            mapped = _map_speaker_name(raw_speaker, replacements) or raw_speaker
            new_summaries[mapped] = _replace_text_casefold(str(text), replacements)
        updated["speaker_summaries"] = new_summaries

    summary_text = updated.get("summary") or ""
    updated["summary"] = _replace_text_casefold(summary_text, replacements)
    return updated


def _apply_speaker_replacements(audio_stem: str, replacements: Dict[str, str]) -> None:
    if not replacements:
        return

    metadata_path = OUTPUT_DIR / f"{audio_stem}{METADATA_SUFFIX}"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail=f"No run metadata for '{audio_stem}'.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    files_info = metadata.get("files", {})

    # Transcript segments (translated)
    transcript_info = files_info.get("transcript_segments")
    if transcript_info:
        path = Path(transcript_info["path"])
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                segments = json.load(handle)
            for segment in segments:
                speaker = segment.get("speaker")
                if speaker:
                    segment["speaker"] = _map_speaker_name(speaker, replacements)
            save_transcript_segments(segments, str(path))
            transcript_text_path = Path(files_info.get("transcript", {}).get("path", ""))
            if transcript_text_path.exists():
                save_transcript(segments, str(transcript_text_path))

    # Original transcript segments
    original_segments_info = files_info.get("original_transcript_segments")
    if original_segments_info:
        path = Path(original_segments_info["path"])
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                original_segments = json.load(handle)
            for segment in original_segments:
                speaker = segment.get("speaker")
                if speaker:
                    segment["speaker"] = _map_speaker_name(speaker, replacements)
            save_original_transcript_segments(original_segments, str(path))
            original_text_path = Path(files_info.get("original_transcript", {}).get("path", ""))
            if original_text_path.exists():
                save_original_transcript(original_segments, str(original_text_path))

    # Topic segments
    topic_info = files_info.get("topic_segments")
    if topic_info:
        path = Path(topic_info["path"])
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                topics = json.load(handle)
            updated_topics = [_apply_speaker_map_to_topic(topic, replacements) for topic in topics]
            save_topics(updated_topics, str(path))

    # Topic summaries
    summaries_info = files_info.get("topic_summaries")
    if summaries_info:
        path = Path(summaries_info["path"])
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                summaries = json.load(handle)
            updated_summaries = [_apply_speaker_map_to_summaries(summary, replacements) for summary in summaries]
            save_topic_summaries(updated_summaries, str(path))

    # Diarization segments
    diarization_info = files_info.get("diarization_segments")
    if diarization_info:
        path = Path(diarization_info["path"])
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                diarization_segments = json.load(handle)
            for segment in diarization_segments:
                speaker = segment.get("speaker")
                if speaker:
                    segment["speaker"] = _map_speaker_name(speaker, replacements)
            save_diarization_segments(diarization_segments, str(path))

    # Transcript text replacements (mirror for non-json lines)
    transcript_text_info = files_info.get("transcript")
    if transcript_text_info:
        path = Path(transcript_text_info["path"])
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            updated_lines = [_rename_text_line(line, replacements) for line in lines]
            path.write_text("\n".join(updated_lines) + ("\n" if updated_lines else ""), encoding="utf-8")

    original_text_info = files_info.get("original_transcript")
    if original_text_info:
        path = Path(original_text_info["path"])
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            updated_lines = [_rename_text_line(line, replacements) for line in lines]
            path.write_text("\n".join(updated_lines) + ("\n" if updated_lines else ""), encoding="utf-8")


class RenamePayload(BaseModel):
    replacements: Dict[str, TextLike]

    def normalized(self) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for old, new in self.replacements.items():
            key = str(old or "").strip()
            new_name = str(new or "").strip()
            if not key or not new_name:
                continue
            result[key.lower()] = new_name
        return result


@app.get("/", include_in_schema=False)
async def index(request: Request) -> Any:
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


@app.get("/api/defaults")
async def get_defaults() -> Dict[str, Any]:
    return _default_config_dict()


@app.get("/api/runs")
async def list_runs() -> Dict[str, List[Dict[str, Any]]]:
    runs: List[Dict[str, Any]] = []
    for path in _load_metadata_files():
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        runs.append(meta)
    return {"runs": runs}


@app.get("/api/runs/{audio_stem}")
async def get_run(audio_stem: str) -> Dict[str, Any]:
    return _collect_run(audio_stem)


@app.post("/api/process")
async def process_audio(
    background: BackgroundTasks,
    audio_file: UploadFile | None = File(None),
    config_json: str = Form("{}"),
    audio_path: str | None = Form(None),
) -> JSONResponse:
    try:
        config_payload = json.loads(config_json) if config_json else {}
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail="Invalid JSON payload for configuration.") from error

    if audio_file is None and not audio_path:
        raise HTTPException(status_code=400, detail="Provide an audio file or an existing audio path.")

    resolved_audio_path: Optional[Path] = None
    temp_audio_path: Optional[Path] = None
    if audio_file is not None and audio_file.filename:
        temp_audio_path = _store_uploaded_file(audio_file)
        resolved_audio_path = temp_audio_path
    elif audio_path:
        resolved_audio_path = _resolve_audio_path(audio_path)

    if resolved_audio_path is None:
        raise HTTPException(status_code=400, detail="Unable to determine audio path.")

    overrides = dict(config_payload)
    if "audio_stem" not in overrides and (audio_file and audio_file.filename):
        overrides["audio_stem"] = Path(audio_file.filename).stem

    config = _build_config(resolved_audio_path, overrides)
    result = run_pipeline(config)
    audio_url = _public_url(resolved_audio_path)
    result["source_audio"] = {
        "path": str(resolved_audio_path),
        "filename": resolved_audio_path.name,
        "url": audio_url,
    }
    metadata_path = _save_metadata(result)

    # Ensure metadata path is available via API response for front-end refresh.
    result["metadata_path"] = str(metadata_path)

    return JSONResponse(result)


@app.post("/api/runs/{audio_stem}/rename-speakers")
async def rename_speakers(audio_stem: str, payload: RenamePayload) -> Dict[str, Any]:
    replacements = payload.normalized()
    if not replacements:
        raise HTTPException(status_code=400, detail="No valid speaker replacements supplied.")

    _apply_speaker_replacements(audio_stem, replacements)

    return _collect_run(audio_stem)
