from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from urllib.parse import quote

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

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None

try:
    import yt_dlp
except ImportError:  # pragma: no cover - optional dependency
    yt_dlp = None

try:
    import podcastparser
except ImportError:  # pragma: no cover - optional dependency
    podcastparser = None

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional dependency
    BeautifulSoup = None

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
DOWNLOAD_DIR = OUTPUT_DIR / "link_downloads"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
METADATA_SUFFIX = "_metadata.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

FILENAME_CLEANER = re.compile(r"[^A-Za-z0-9_.-]")

LINK_INDEX_PATH = DOWNLOAD_DIR / "link_index.json"
SHOW_CACHE_PATH = DOWNLOAD_DIR / "show_cache.json"
REQUEST_TIMEOUT = 10
PODCASTINDEX_API_KEY = ""
PODCASTINDEX_API_SECRET = ""

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


class LinkRequest(BaseModel):
    link: str


class JsonStore:
    def __init__(self, path: Path, default_factory=dict):
        self.path = path
        self.lock = Lock()
        self.default_factory = default_factory
        if not self.path.exists():
            self._write(self.default_factory())

    def _read(self) -> Any:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return self.default_factory()

    def _write(self, data: Any) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(self.path)

    def get(self) -> Any:
        with self.lock:
            return self._read()

    def set(self, data: Any) -> None:
        with self.lock:
            self._write(data)


link_index = JsonStore(LINK_INDEX_PATH, default_factory=dict)
show_cache = JsonStore(SHOW_CACHE_PATH, default_factory=dict)


# ---------------------------------------------------------------------------
# Link ingestion helpers
# ---------------------------------------------------------------------------#

def _require_requests() -> None:
    if requests is None:  # pragma: no cover - optional dependency
        raise HTTPException(
            status_code=503,
            detail="The 'requests' package is required for link ingestion. Install it to enable this feature.",
        )


def _require_parser(lib: Any, name: str) -> None:
    if lib is None:  # pragma: no cover - optional dependency
        raise HTTPException(
            status_code=503,
            detail=f"The '{name}' package is required for this operation. Install it to enable this feature.",
        )


def _link_public_url(path: Path | str) -> Optional[str]:
    try:
        return _public_url(Path(path))
    except Exception:
        return None


def _link_download_path(filename: str) -> Path:
    return DOWNLOAD_DIR / filename


def _is_youtube_link(link: str) -> bool:
    lowered = link.lower()
    return "youtube.com" in lowered or "youtu.be" in lowered


def _is_spotify_link(link: str) -> bool:
    lowered = link.lower()
    return "open.spotify.com" in lowered and ("/episode/" in lowered or "/show/" in lowered)


def _spotify_episode_id_from_url(url: str) -> Optional[str]:
    match = re.search(r"/episode/([A-Za-z0-9]+)", url)
    return match.group(1) if match else None


def _requests_headers() -> dict:
    return {"User-Agent": "PodcastPipeline/1.0"}


def _safe_get(url: str, **kwargs) -> "requests.Response":
    _require_requests()
    kwargs.setdefault("timeout", REQUEST_TIMEOUT)
    kwargs.setdefault("headers", _requests_headers())
    response = requests.get(url, **kwargs)
    response.raise_for_status()
    return response


def _youtube_progress_hook(status: dict) -> None:  # pragma: no cover - logging only
    stage = status.get("status")
    if stage == "downloading":
        percent = status.get("_percent_str", "").strip()
        print(f"Downloading YouTube audio: {percent}")
    elif stage == "finished":
        print("YouTube download finished.")


def _download_youtube_audio(link: str) -> tuple[Path, dict]:
    _require_parser(yt_dlp, "yt-dlp")
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(DOWNLOAD_DIR / "%(title)s.%(ext)s"),
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ],
        "progress_hooks": [_youtube_progress_hook],
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(link, download=True)
        filename = ydl.prepare_filename(info)
        audio_path = Path(filename).with_suffix(".mp3")
        meta = {
            "source": "youtube",
            "video_id": info.get("id"),
            "title": info.get("title"),
            "uploader": info.get("uploader"),
            "duration": info.get("duration"),
            "file": str(audio_path),
            "public_url": _link_public_url(audio_path),
            "link": link,
            "created_at": int(time.time()),
        }
        return audio_path, meta


def _get_show_url_from_episode(episode_url: str) -> Optional[str]:
    _require_parser(BeautifulSoup, "beautifulsoup4")
    try:
        response = _safe_get(episode_url)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            if "/show/" in href:
                href = href.split("?")[0]
                if href.startswith("http"):
                    return href
                return "https://open.spotify.com" + href
    except Exception as error:  # pragma: no cover - logging
        print(f"Error extracting show URL: {error}")
    return None


def _is_spotify_exclusive(show_url: str) -> bool:
    try:
        response = _safe_get(show_url)
        return "Only on Spotify" in response.text
    except Exception as error:  # pragma: no cover - logging
        print(f"Error checking exclusivity: {error}")
        return False


def _get_meta_title(url: str) -> Optional[str]:
    _require_parser(BeautifulSoup, "beautifulsoup4")
    try:
        response = _safe_get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        tag = soup.find("meta", property="og:title")
        if tag and tag.get("content"):
            return tag["content"].strip()
    except Exception as error:  # pragma: no cover - logging
        print(f"Error extracting title from {url}: {error}")
    return None


def _podcastindex_search(show_title: str) -> Optional[str]:
    cache_key = show_title.strip().lower()
    cached = show_cache.get()
    if cache_key in cached:
        return cached[cache_key]

    now = int(time.time())
    auth_string = f"{PODCASTINDEX_API_KEY}{PODCASTINDEX_API_SECRET}{now}".encode("utf-8")
    auth_hash = hashlib.sha1(auth_string).hexdigest()
    headers = {
        "User-Agent": "PodcastPipeline/1.0",
        "X-Auth-Date": str(now),
        "X-Auth-Key": PODCASTINDEX_API_KEY,
        "Authorization": auth_hash,
    }
    params = {"q": show_title}
    try:
        response = requests.get(
            "https://api.podcastindex.org/api/1.0/search/byterm",
            headers=headers,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        feeds = response.json().get("feeds") or []
        rss_url = feeds[0].get("url") if feeds else None
        if rss_url:
            cached[cache_key] = rss_url
            show_cache.set(cached)
        return rss_url
    except Exception as error:  # pragma: no cover - logging
        print(f"PodcastIndex error: {error}")
        return None


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().lower())


def _download_episode_from_rss(rss_url: str, episode_title: str) -> tuple[Optional[Path], Optional[dict], Optional[str]]:
    _require_parser(podcastparser, "podcastparser")
    try:
        response = _safe_get(rss_url)
        feed = podcastparser.parse(rss_url, BytesIO(response.content))
        target = _normalize_title(episode_title)
        episodes = feed.get("episodes") or []
        episode = next((ep for ep in episodes if _normalize_title(ep.get("title", "")) == target), None)
        if not episode:
            episode = next((ep for ep in episodes if target in _normalize_title(ep.get("title", ""))), None)
        if not episode:
            return None, None, "Episode title not found in RSS feed."
        enclosures = episode.get("enclosures") or []
        if not enclosures:
            return None, None, "Episode has no audio enclosure."
        audio_url = enclosures[0].get("url")
        if not audio_url:
            return None, None, "Audio URL missing from enclosure."
        guid = episode.get("guid") or hashlib.md5(audio_url.encode()).hexdigest()
        safe_title = re.sub(r'[\\/*?:"<>|]', "_", episode.get("title", "episode"))
        filename = DOWNLOAD_DIR / f"{guid}_{safe_title}.mp3"
        if filename.exists():
            meta = {
                "source": "rss",
                "guid": guid,
                "title": episode.get("title"),
                "show": feed.get("title"),
                "file": str(filename),
                "public_url": _link_public_url(filename),
                "audio_url": audio_url,
                "created_at": int(time.time()),
            }
            return filename, meta, None
        stream = _safe_get(audio_url, stream=True)
        with filename.open("wb") as handle:
            for chunk in stream.iter_content(chunk_size=256 * 1024):
                if chunk:
                    handle.write(chunk)
        meta = {
            "source": "rss",
            "guid": guid,
            "title": episode.get("title"),
            "show": feed.get("title"),
            "file": str(filename),
            "public_url": _link_public_url(filename),
            "audio_url": audio_url,
            "created_at": int(time.time()),
        }
        return filename, meta, None
    except Exception as error:  # pragma: no cover - logging
        return None, None, f"RSS download error: {error}"


def _upsert_link_index(key: str, meta: dict) -> None:
    data = link_index.get()
    data[key] = meta
    link_index.set(data)


def _get_link_index(key: str) -> Optional[dict]:
    return link_index.get().get(key)


def _submit_spotify(link: str) -> dict:
    _require_requests()
    episode_id = _spotify_episode_id_from_url(link)
    if episode_id:
        existing = _get_link_index(f"spotify:episode:{episode_id}")
        if existing and Path(existing.get("file", "")).exists():
            return {"status": "already_downloaded", "meta": existing}

    episode_title = _get_meta_title(link)
    if not episode_title:
        raise HTTPException(status_code=400, detail="Could not determine Spotify episode title.")

    show_url = _get_show_url_from_episode(link)
    if not show_url:
        raise HTTPException(status_code=400, detail="Could not locate show URL from episode link.")

    if _is_spotify_exclusive(show_url):
        raise HTTPException(status_code=400, detail="Spotify-exclusive show. No public RSS feed available.")

    show_title = _get_meta_title(show_url)
    if not show_title:
        raise HTTPException(status_code=400, detail="Unable to read show title.")

    rss_url = _podcastindex_search(show_title)
    if not rss_url:
        raise HTTPException(status_code=404, detail=f"No RSS feed found for show '{show_title}'.")

    path, meta, error = _download_episode_from_rss(rss_url, episode_title)
    if error:
        raise HTTPException(status_code=400, detail=error)
    if not meta or not path:
        raise HTTPException(status_code=500, detail="Failed to download episode from RSS.")

    if episode_id:
        meta_for_index = {
            "source": "spotify",
            "episode_id": episode_id,
            "episode_title": meta.get("title"),
            "show_title": meta.get("show") or show_title,
            "file": meta.get("file"),
            "public_url": meta.get("public_url"),
            "rss_url": rss_url,
            "link": link,
            "created_at": int(time.time()),
        }
        _upsert_link_index(f"spotify:episode:{episode_id}", meta_for_index)
    if meta.get("guid"):
        _upsert_link_index(f"rss:{meta['guid']}", meta)
    return {"status": "downloaded", "meta": meta}


def _submit_youtube(link: str) -> dict:
    _require_parser(yt_dlp, "yt-dlp")
    video_id_match = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{6,})", link)
    video_id = video_id_match.group(1) if video_id_match else None
    if video_id:
        existing = _get_link_index(f"youtube:{video_id}")
        if existing and Path(existing.get("file", "")).exists():
            return {"status": "already_downloaded", "meta": existing}

    path, meta = _download_youtube_audio(link)
    if video_id:
        _upsert_link_index(f"youtube:{video_id}", meta)
    elif meta.get("video_id"):
        _upsert_link_index(f"youtube:{meta['video_id']}", meta)
    return {"status": "downloaded", "meta": meta}


def _ingest_audio_link(link: str) -> tuple[Path, dict]:
    link = link.strip()
    if not link:
        raise HTTPException(status_code=400, detail="Link is empty.")
    if _is_spotify_link(link):
        result = _submit_spotify(link)
    elif _is_youtube_link(link):
        result = _submit_youtube(link)
    else:
        raise HTTPException(status_code=400, detail="Unsupported link. Provide a Spotify or YouTube URL.")
    meta = result.get("meta")
    if not meta:
        raise HTTPException(status_code=500, detail="Link ingestion returned no metadata.")
    file_path = meta.get("file")
    if not file_path:
        raise HTTPException(status_code=500, detail="Downloaded file path missing in metadata.")
    resolved = Path(file_path)
    if not resolved.exists():
        raise HTTPException(status_code=500, detail="Downloaded audio file could not be found.")
    return resolved, meta


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

    audio_link = config_payload.pop("audio_link", None)
    resolved_audio_path: Optional[Path] = None
    temp_audio_path: Optional[Path] = None
    link_meta: Optional[dict] = None

    if audio_file is not None and audio_file.filename:
        temp_audio_path = _store_uploaded_file(audio_file)
        resolved_audio_path = temp_audio_path
    elif audio_path:
        resolved_audio_path = _resolve_audio_path(audio_path)
    elif audio_link:
        resolved_audio_path, link_meta = _ingest_audio_link(audio_link)

    overrides = dict(config_payload)
    audio_stem_hint = str(overrides.get("audio_stem") or "").strip()

    if resolved_audio_path is None:
        if audio_stem_hint:
            resolved_audio_path = OUTPUT_DIR / f"{audio_stem_hint}.placeholder"
        else:
            raise HTTPException(status_code=400, detail="Provide an audio file, audio path, supported link, or existing audio stem.")

    if "audio_stem" not in overrides or not audio_stem_hint:
        if audio_file and audio_file.filename:
            overrides["audio_stem"] = Path(audio_file.filename).stem
        else:
            overrides["audio_stem"] = resolved_audio_path.stem

    config = _build_config(resolved_audio_path, overrides)
    result = run_pipeline(config)
    audio_url = _public_url(resolved_audio_path) if resolved_audio_path.exists() else None
    result["source_audio"] = {
        "path": str(resolved_audio_path),
        "filename": resolved_audio_path.name,
        "url": audio_url,
    }
    if link_meta:
        result["link_download"] = link_meta
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


@app.post("/api/links/submit")
async def submit_link(link_request: LinkRequest) -> Dict[str, Any]:
    link = link_request.link.strip()
    if not link:
        raise HTTPException(status_code=400, detail="Link is required.")

    if _is_spotify_link(link):
        return _submit_spotify(link)
    if _is_youtube_link(link):
        return _submit_youtube(link)

    raise HTTPException(status_code=400, detail="Unsupported link. Provide a Spotify or YouTube URL.")


@app.get("/api/links/downloads")
async def list_link_downloads() -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    for path in sorted(DOWNLOAD_DIR.glob("*")):
        if not path.is_file():
            continue
        files.append(
            {
                "file": str(path),
                "public_url": _link_public_url(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {"count": len(files), "items": files}
