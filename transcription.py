from __future__ import annotations

from typing import Any

from faster_whisper import WhisperModel


def _serialize_words(words: list[Any]) -> list[dict]:
    serialized: list[dict] = []
    for word in words or []:
        start = getattr(word, "start", None)
        end = getattr(word, "end", None)
        text = getattr(word, "word", "") or ""
        if start is None or end is None:
            continue
        serialized.append({"start": float(start), "end": float(end), "text": text})
    return serialized


def transcribe_audio(
    audio_path: str,
    model_size: str = "small",
    beam_size: int = 5,
    device: str = "auto",
    compute_type: str = "int8",
    word_timestamps: bool = True,
    auto_translate_languages: set[str] | None = None,
    translate_target_language: str = "en",
    language_override: str | None = None,
) -> tuple[list[dict], dict]:
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    translate_languages = {lang.lower() for lang in auto_translate_languages or set()}

    force_translate = False
    if language_override and translate_languages:
        if language_override.lower() in translate_languages and language_override.lower() != translate_target_language.lower():
            force_translate = True

    task = "translate" if force_translate else "transcribe"

    transcribe_kwargs: dict[str, Any] = {
        "beam_size": beam_size,
        "word_timestamps": word_timestamps and task != "translate",
        "task": task,
    }
    if language_override:
        transcribe_kwargs["language"] = language_override

    segments_iter, info = model.transcribe(audio_path, **transcribe_kwargs)

    detected_language = getattr(info, "language", None)
    language_probability = getattr(info, "language_probability", None)

    if language_override:
        detected_language = language_override

    original_segments: list[dict] = []
    for segment in segments_iter:
        words = _serialize_words(getattr(segment, "words", []))
        entry = {
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
            "words": words if task != "translate" else [],
        }
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        original_segments.append(entry)

    segments = [dict(entry) for entry in original_segments]

    translation_applied = task == "translate"
    translated_from = detected_language if translation_applied else None

    info_dict = {
        "language": detected_language,
        "language_probability": language_probability,
        "translated": translation_applied,
        "translated_from": translated_from,
        "translated_to": translate_target_language if translation_applied else None,
    }

    return segments, info_dict


def _align_translated_text(
    original_segments: list[dict],
    translated_segments: list[dict],
) -> list[str]:
    if not translated_segments:
        return [segment["text"] for segment in original_segments]

    if len(original_segments) == len(translated_segments):
        return [segment["text"] for segment in translated_segments]

    translated_segments_sorted = sorted(translated_segments, key=lambda item: item["start"])
    result: list[str] = []
    best_index = 0
    for original in original_segments:
        best_distance = float("inf")
        for idx in range(best_index, len(translated_segments_sorted)):
            candidate = translated_segments_sorted[idx]
            distance = abs(candidate["start"] - original["start"])
            if distance <= best_distance:
                best_distance = distance
                best_index = idx
            else:
                break
        result.append(translated_segments_sorted[best_index]["text"])
    return result
