from __future__ import annotations

from typing import Iterable, Sequence

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover - handled at runtime
    pipeline = None


def load_summarizer(model_name: str, device: str | None) -> callable:
    if pipeline is None:
        raise ImportError(
            "transformers is required for summarization. "
            "Install it with 'pip install transformers sentencepiece'."
        )

    pipeline_kwargs: dict[str, object] = {"model": model_name}
    if device:
        pipeline_kwargs["device"] = device

    return pipeline("summarization", **pipeline_kwargs)


def _compute_summary_lengths(
    num_words: int,
    base_min_length: int,
    base_max_length: int,
) -> tuple[int, int]:
    # Roughly map words to tokenizer tokens, keeping bounds sensible.
    approx_tokens = max(1, int(num_words * 1.3))
    dynamic_max = max(16, min(base_max_length, max(base_min_length, int(approx_tokens * 0.9))))
    dynamic_min = min(base_min_length, dynamic_max)
    if dynamic_min < 10 and dynamic_max > 20:
        dynamic_min = min(dynamic_max - 8, 20)
    return dynamic_min, dynamic_max


def _word_chunks(words: list[str], max_words: int, overlap_words: int) -> Iterable[list[str]]:
    if max_words <= 0:
        yield words
        return
    step = max(1, max_words - overlap_words)
    for start in range(0, len(words), step):
        yield words[start : start + max_words]


def summarize_text(
    text: str,
    summarizer,
    *,
    chunk_word_limit: int,
    chunk_overlap: int,
    summary_min_length: int,
    summary_max_length: int,
) -> str:
    tokens = text.split()
    if not tokens:
        return ""

    chunk_summaries: list[str] = []
    for chunk in _word_chunks(tokens, chunk_word_limit, chunk_overlap):
        chunk_text = " ".join(chunk)
        effective_min, effective_max = _compute_summary_lengths(
            len(chunk),
            summary_min_length,
            summary_max_length,
        )
        summary = summarizer(
            chunk_text,
            max_length=effective_max,
            min_length=effective_min,
            truncation=True,
        )[0]["summary_text"]
        chunk_summaries.append(summary.strip())

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    combined = " ".join(chunk_summaries)
    effective_min, effective_max = _compute_summary_lengths(
        len(combined.split()),
        summary_min_length,
        summary_max_length,
    )
    final_summary = summarizer(
        combined,
        max_length=effective_max,
        min_length=effective_min,
        truncation=True,
    )[0]["summary_text"]
    return final_summary.strip()


def summarize_topics(
    topic_segments: Sequence[dict],
    summarizer,
    *,
    chunk_word_limit: int,
    chunk_overlap: int,
    summary_min_length: int,
    summary_max_length: int,
    context_before: int = 0,
    context_after: int = 0,
    min_words_for_summary: int = 50,
) -> list[dict]:
    summaries: list[dict] = []
    for index, topic in enumerate(topic_segments):
        text = _prepare_topic_text(topic)
        context_text = _gather_context_text(
            topic_segments,
            index,
            context_before=context_before,
            context_after=context_after,
        )

        combined_input = text
        if context_text:
            combined_input = (
                f"Previous/Next context:\n{context_text.strip()}\n\n"
                f"Current segment:\n{text.strip()}"
            ).strip()

        summary_text = combined_input.strip()
        word_count = len(summary_text.split())
        if word_count < min_words_for_summary:
            summary_text = text.strip()
        else:
            summary_text = summarize_text(
                combined_input,
                summarizer,
                chunk_word_limit=chunk_word_limit,
                chunk_overlap=chunk_overlap,
                summary_min_length=summary_min_length,
                summary_max_length=summary_max_length,
            )

        speaker_summaries: dict[str, str] = {}
        speaker_texts = _collect_speaker_texts(topic)
        for speaker, speaker_text in speaker_texts.items():
            if not speaker_text.strip():
                continue
            words = speaker_text.split()
            if len(words) < 12:
                speaker_summaries[speaker] = speaker_text.strip()
                continue
            speaker_summary = summarize_text(
                speaker_text,
                summarizer,
                chunk_word_limit=max(80, chunk_word_limit // 2),
                chunk_overlap=max(0, chunk_overlap // 2),
                summary_min_length=max(20, summary_min_length // 2),
                summary_max_length=max(60, summary_max_length // 2),
            )
            speaker_summaries[speaker] = speaker_summary

        summaries.append(
            {
                "start": topic.get("start"),
                "end": topic.get("end"),
                "summary": summary_text,
                "primary_speaker": topic.get("primary_speaker"),
                "speaker_counts": topic.get("speaker_counts"),
                "speaker_summaries": speaker_summaries,
            }
        )
    return summaries


def _prepare_topic_text(topic: dict) -> str:
    turns = topic.get("turns") or []
    formatted_turns: list[str] = []
    for turn in turns:
        turn_text = (turn.get("text") or "").strip()
        if not turn_text:
            continue
        speaker = turn.get("speaker")
        if speaker:
            formatted_turns.append(f"{speaker}: {turn_text}")
        else:
            formatted_turns.append(turn_text)

    if formatted_turns:
        return " ".join(formatted_turns)

    return (topic.get("text") or "").strip()


def _collect_speaker_texts(topic: dict) -> dict[str, str]:
    speaker_to_text: dict[str, list[str]] = {}
    for turn in topic.get("turns") or []:
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        speaker = turn.get("speaker") or "Unknown"
        speaker_to_text.setdefault(speaker, []).append(text)
    return {speaker: " ".join(parts) for speaker, parts in speaker_to_text.items()}


def _gather_context_text(
    topic_segments: Sequence[dict],
    index: int,
    *,
    context_before: int,
    context_after: int,
) -> str:
    chunks: list[str] = []

    before_start = max(0, index - context_before)
    for topic in topic_segments[before_start:index]:
        text = _prepare_topic_text(topic)
        if text:
            chunks.append(text)

    after_end = min(len(topic_segments), index + context_after + 1)
    for topic in topic_segments[index + 1 : after_end]:
        text = _prepare_topic_text(topic)
        if text:
            chunks.append(text)

    return " ".join(chunks)
