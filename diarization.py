from __future__ import annotations

from typing import Any, Iterable, Sequence
import torch

def diarize_audio(
    audio_path: str,
    model_id: str = "pyannote/speaker-diarization",
    auth_token: str | None = None,
    revision: str | None = None,
    device: str | None = None,
) -> list[dict]:
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError:
        Pipeline = None

    if Pipeline is None:
        try:
            from pyannote.pipeline import Pipeline  # type: ignore
        except ImportError as error:  # pragma: no cover - handled at runtime
            raise ImportError(
                "pyannote.audio/pyannote.pipeline is required for speaker diarization. "
                "Install it with 'pip install pyannote.audio<3 pyannote.pipeline<2' and provide a Hugging Face token "
                "with access to the chosen model."
            ) from error

    pipeline_kwargs: dict[str, object] = {}
    if revision:
        pipeline_kwargs["revision"] = revision

    def _instantiate_pipeline(from_pretrained: bool) -> Any:
        auth_kwargs: list[dict[str, object]] = [{}]
        if auth_token:
            auth_kwargs = [{"use_auth_token": auth_token}, {"token": auth_token}, {}]

        if from_pretrained:
            for auth_kw in auth_kwargs:
                try:
                    return Pipeline.from_pretrained(model_id, **pipeline_kwargs, **auth_kw)
                except TypeError:
                    continue
        else:
            for auth_kw in auth_kwargs:
                try:
                    return Pipeline(model_id, **pipeline_kwargs, **auth_kw)
                except TypeError:
                    continue
        if auth_token:
            raise TypeError(
                "The installed pyannote version does not accept known authentication parameters "
                "(use_auth_token/token). Try removing the token or upgrading pyannote."
            )
        raise TypeError("Unable to instantiate pyannote Pipeline with the current installation.")

    try:
        pipeline = _instantiate_pipeline(hasattr(Pipeline, "from_pretrained"))
    except TypeError:
        # Fall back to legacy signature
        pipeline = _instantiate_pipeline(False)
    except ValueError as error:
        if "revision" in str(error) and not revision:
            raise ValueError(
                "The selected diarization model requires a specific revision. "
                "Re-run with --diarization-revision <revision> (e.g. 2024-03-14)."
            ) from error
        raise
    if device:
        pipeline.to(torch.device(device))

    try:
        import torchaudio
    except ImportError as error:  # pragma: no cover - handled at runtime
        raise ImportError(
            "torchaudio is required to load audio for diarization. "
            "Install it with 'pip install torchaudio'."
        ) from error

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    waveform = waveform.to(torch.float32)
    if device and device.lower() != "cpu":
        waveform = waveform.to(torch.device(device))

    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    if hasattr(diarization, "itertracks"):
        annotation = diarization
    elif hasattr(diarization, "speaker_diarization"):
        annotation = diarization.speaker_diarization
    elif isinstance(diarization, dict):
        annotation = diarization.get("speaker_diarization") or diarization.get("annotation")
        if annotation is None:
            raise AttributeError(
                "Diarization output does not expose speaker annotations."
            )
    else:
        raise AttributeError("Unsupported diarization output type: " + type(diarization).__name__)

    diarization_segments: list[dict] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        diarization_segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )

    diarization_segments.sort(key=lambda item: item["start"])
    return diarization_segments


def _speaker_at_time(
    timestamp: float,
    diarization_segments: Sequence[dict],
) -> str | None:
    for segment in diarization_segments:
        if segment["start"] <= timestamp < segment["end"]:
            return segment["speaker"]
    if diarization_segments and timestamp >= diarization_segments[-1]["end"]:
        return diarization_segments[-1]["speaker"]
    return None


def _split_words_by_speaker(
    words: Sequence[dict],
    diarization_segments: Sequence[dict],
) -> list[tuple[list[dict], str | None]]:
    grouped: list[tuple[list[dict], str | None]] = []
    current_speaker: str | None = None
    current_words: list[dict] = []

    for word in words:
        start = word.get("start")
        end = word.get("end")
        if start is None or end is None:
            continue
        midpoint = (float(start) + float(end)) / 2.0
        speaker = _speaker_at_time(midpoint, diarization_segments)

        if current_words and speaker != current_speaker:
            grouped.append((current_words, current_speaker))
            current_words = []

        current_words.append(word)
        current_speaker = speaker

    if current_words:
        grouped.append((current_words, current_speaker))
    return grouped


def apply_diarization_to_transcript(
    transcript_segments: Sequence[dict],
    diarization_segments: Sequence[dict],
) -> list[dict]:
    if not diarization_segments:
        return list(transcript_segments)

    enhanced_segments: list[dict] = []
    for segment in transcript_segments:
        words = segment.get("words") or []
        if not words:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
            midpoint = (start + end) / 2.0
            speaker = _speaker_at_time(midpoint, diarization_segments) or "Unknown"
            enhanced_segment = dict(segment)
            enhanced_segment["speaker"] = speaker
            enhanced_segments.append(enhanced_segment)
            continue

        grouped_words = _split_words_by_speaker(words, diarization_segments)
        for word_group, speaker in grouped_words:
            text = "".join(word.get("text", "") for word in word_group).strip()
            if not text:
                continue

            start = float(word_group[0].get("start", segment.get("start", 0.0)))
            end = float(word_group[-1].get("end", segment.get("end", start)))
            enhanced_segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "speaker": speaker or "Unknown",
                    "words": word_group,
                }
            )

    return enhanced_segments
