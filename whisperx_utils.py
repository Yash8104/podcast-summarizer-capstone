from __future__ import annotations

from typing import Any


def _resolve_device(preferred: str | None) -> str:
    try:
        import torch
    except ImportError:
        return preferred or "cpu"

    if preferred:
        if preferred.lower() == "mps":
            print("Warning: faster-whisper backend does not support MPS; falling back to CPU.")
            return "cpu"
        return preferred

    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("Warning: faster-whisper backend does not support MPS; using CPU instead.")
        return "cpu"
    return "cpu"


def transcribe_and_diarize(
    *,
    audio_path: str,
    model_size: str,
    device: str | None,
    compute_type: str,
    batch_size: int,
    auth_token: str | None,
) -> dict[str, Any]:
    try:
        import torch
        import whisperx
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise ImportError(
            "whisperx (and torch) is required for WhisperX-based diarization. "
            "Install with 'pip install whisperx torch torchaudio' and try again."
        ) from error

    resolved_device = _resolve_device(device)
    effective_compute = compute_type
    if resolved_device == "cpu" and compute_type.lower() in {"float16", "float32", "float"}:
        effective_compute = "int8"
        print(
            "Warning: float16/32 are inefficient on CPU for faster-whisper. "
            "Using compute_type=int8 instead."
        )

    model = whisperx.load_model(
        model_size,
        device=resolved_device,
        compute_type=effective_compute,
    )
    result = model.transcribe(audio_path, batch_size=batch_size)

    language = result.get("language")
    if language:
        try:
            align_model, metadata = whisperx.load_align_model(
                language=language,
                device=resolved_device,
            )
        except TypeError:
            try:
                align_model, metadata = whisperx.load_align_model(language, resolved_device)
            except TypeError:
                align_model, metadata = whisperx.load_align_model(language_code=language, device=resolved_device)
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio_path,
            device=resolved_device,
        )
        del align_model

    try:
        diarization_cls = whisperx.DiarizationPipeline
    except AttributeError:
        try:
            from whisperx.diarize import DiarizationPipeline as diarization_cls  # type: ignore
        except Exception as error:  # pragma: no cover - runtime dependency mismatch
            raise AttributeError(
                "Could not locate whisperx.DiarizationPipeline. Update whisperx to the latest version."
            ) from error

    diarization_pipeline = diarization_cls(
        use_auth_token=auth_token,
        device=resolved_device,
    )
    diarization_annotation = diarization_pipeline(audio_path)
    result = whisperx.assign_word_speakers(diarization_annotation, result)

    segments: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        text = (segment.get("text") or "").strip()
        speaker = segment.get("speaker")
        word_entries: list[dict[str, Any]] = []
        for word in segment.get("words", []) or []:
            word_text = (word.get("word") or "").strip()
            if not word_text:
                continue
            start = word.get("start")
            end = word.get("end")
            if start is None or end is None:
                continue
            word_entries.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "text": word_text,
                    "confidence": word.get("confidence"),
                }
            )
        segments.append(
            {
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "text": text,
                "speaker": speaker,
                "words": word_entries,
            }
        )

    diarization_segments: list[dict[str, Any]] = []
    annotation = diarization_annotation
    if hasattr(annotation, "speaker_diarization"):
        annotation = annotation.speaker_diarization
    if hasattr(annotation, "itertracks"):
        for turn, _, spk in annotation.itertracks(yield_label=True):
            diarization_segments.append(
                {
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(spk),
                }
            )

    diarization_segments.sort(key=lambda item: item["start"])

    return {
        "segments": segments,
        "diarization_segments": diarization_segments,
        "language": language,
    }
