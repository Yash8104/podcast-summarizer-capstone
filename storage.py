from __future__ import annotations

import json
from typing import Iterable, Sequence


def save_transcript(segments: Iterable[dict], transcript_path: str) -> None:
    with open(transcript_path, "w") as file:
        for segment in segments:
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            speaker = segment.get("speaker")
            prefix = f"{speaker}: " if speaker else ""
            file.write(f"{prefix}{text}\n")


def save_original_transcript(segments: Iterable[dict], transcript_path: str) -> None:
    with open(transcript_path, "w") as file:
        for segment in segments:
            text = (segment.get("original_text") or segment.get("text") or "").strip()
            if not text:
                continue
            speaker = segment.get("speaker")
            prefix = f"{speaker}: " if speaker else ""
            file.write(f"{prefix}{text}\n")


def _strip_original_text(entries: Sequence[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for entry in entries:
        item = dict(entry)
        item.pop("original_text", None)
        cleaned.append(item)
    return cleaned


def save_transcript_segments(segments: Sequence[dict], output_path: str) -> None:
    with open(output_path, "w") as file:
        json.dump(_strip_original_text(segments), file, indent=2)


def save_original_transcript_segments(segments: Sequence[dict], output_path: str) -> None:
    originals: list[dict] = []
    for segment in segments:
        copy = dict(segment)
        copy["text"] = (segment.get("original_text") or segment.get("text") or "").strip()
        copy.pop("original_text", None)
        originals.append(copy)
    with open(output_path, "w") as file:
        json.dump(originals, file, indent=2)


def load_transcript_segments(input_path: str) -> list[dict]:
    with open(input_path, "r") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of segments in {input_path}")
    return data


def save_topics(topic_segments: Iterable[dict], output_path: str) -> None:
    with open(output_path, "w") as file:
        json.dump(list(topic_segments), file, indent=2)


def load_topics(input_path: str) -> list[dict]:
    with open(input_path, "r") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of topic segments in {input_path}")
    return data


def save_topic_summaries(topic_summaries: Iterable[dict], output_path: str) -> None:
    with open(output_path, "w") as file:
        json.dump(list(topic_summaries), file, indent=2)


def save_diarization_segments(diarization_segments: Sequence[dict], output_path: str) -> None:
    with open(output_path, "w") as file:
        json.dump(list(diarization_segments), file, indent=2)


def load_diarization_segments(input_path: str) -> list[dict]:
    with open(input_path, "r") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of diarization segments in {input_path}")
    return data
