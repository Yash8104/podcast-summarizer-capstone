from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Literal, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime
    np = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - handled at runtime
    SentenceTransformer = None

# A compact list of common stopwords to reduce noise when comparing segments in lexical mode.
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def find_topic_boundaries(
    transcript_segments: list[dict],
    *,
    method: Literal["embeddings", "lexical", "changepoint"] = "embeddings",
    window_size: int = 6,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    embedding_device: str | None = None,
    score_quantile: float = 0.8,
    min_gap: int = 5,
    smooth_size: int = 5,
    changepoint_penalty: float = 8.0,
    changepoint_min_size: int = 3,
) -> list[int]:
    if method != "changepoint":
        if score_quantile <= 0 or score_quantile >= 1:
            raise ValueError("score_quantile must be between 0 and 1 (exclusive).")
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        if min_gap < 1:
            raise ValueError("min_gap must be at least 1.")
        if smooth_size < 1:
            raise ValueError("smooth_size must be at least 1.")
    if method == "changepoint":
        if changepoint_penalty <= 0:
            raise ValueError("changepoint_penalty must be positive.")
        if changepoint_min_size < 1:
            raise ValueError("changepoint_min_size must be at least 1.")

    if method == "embeddings":
        try:
            return _find_boundaries_with_embeddings(
                transcript_segments,
                window_size=window_size,
                model_name=embedding_model,
                device=embedding_device,
                score_quantile=score_quantile,
                min_gap=min_gap,
                smooth_size=smooth_size,
            )
        except ImportError as error:
            print(
                f"Embedding-based segmentation unavailable ({error}). "
                "Falling back to lexical similarity."
            )
        except RuntimeError as error:
            # Some SentenceTransformer backends throw RuntimeError on device issues or missing weights.
            print(
                f"Embedding-based segmentation failed ({error}). "
                "Falling back to lexical similarity."
            )
        except Exception as error:
            print(
                f"Embedding-based segmentation encountered an unexpected error ({error}). "
                "Falling back to lexical similarity."
            )
        return _find_boundaries_with_lexical_similarity(
            transcript_segments,
            window_size=window_size,
            score_quantile=score_quantile,
            min_gap=min_gap,
            smooth_size=smooth_size,
        )

    if method == "lexical":
        return _find_boundaries_with_lexical_similarity(
            transcript_segments,
            window_size=window_size,
            score_quantile=score_quantile,
            min_gap=min_gap,
            smooth_size=smooth_size,
        )

    if method == "changepoint":
        boundaries = _find_boundaries_with_changepoint_detection(
            transcript_segments,
            model_name=embedding_model,
            device=embedding_device,
            penalty=changepoint_penalty,
            min_size=changepoint_min_size,
        )
        if not boundaries:
            print(
                "Changepoint detection did not find any clear boundaries. "
                "Falling back to embedding similarity segmentation."
            )
            return _find_boundaries_with_embeddings(
                transcript_segments,
                window_size=window_size,
                model_name=embedding_model,
                device=embedding_device,
                score_quantile=score_quantile,
                min_gap=min_gap,
                smooth_size=smooth_size,
            )
        return boundaries
    raise ValueError(f"Unknown topic segmentation method: {method}")


def merge_segments(transcript_segments: list[dict], boundaries: list[int]) -> list[dict]:
    if not transcript_segments:
        return []

    merged = []
    start_index = 0
    for boundary in boundaries:
        chunk = transcript_segments[start_index:boundary]
        if chunk:
            speakers = [part.get("speaker") for part in chunk if part.get("speaker")]
            speaker_counts = Counter(speakers)
            turns = [
                {
                    "start": part.get("start"),
                    "end": part.get("end"),
                    "speaker": part.get("speaker"),
                    "text": part.get("text"),
                }
                for part in chunk
            ]
            entry = {
                "start": chunk[0].get("start"),
                "end": chunk[-1].get("end"),
                "text": " ".join(_segment_text_with_speaker(part) for part in chunk).strip(),
                "turns": turns,
                "speaker_counts": dict(speaker_counts) if speaker_counts else {},
                "primary_speaker": speaker_counts.most_common(1)[0][0] if speaker_counts else None,
            }
            merged.append(entry)
        start_index = boundary

    remainder = transcript_segments[start_index:]
    if remainder:
        speakers = [part.get("speaker") for part in remainder if part.get("speaker")]
        speaker_counts = Counter(speakers)
        turns = [
            {
                "start": part.get("start"),
                "end": part.get("end"),
                "speaker": part.get("speaker"),
                "text": part.get("text"),
            }
            for part in remainder
        ]
        entry = {
            "start": remainder[0].get("start"),
            "end": remainder[-1].get("end"),
            "text": " ".join(_segment_text_with_speaker(part) for part in remainder).strip(),
            "turns": turns,
            "speaker_counts": dict(speaker_counts) if speaker_counts else {},
            "primary_speaker": speaker_counts.most_common(1)[0][0] if speaker_counts else None,
        }
        merged.append(entry)
    return merged


def merge_small_topics(
    topic_segments: list[dict],
    *,
    min_duration: float = 0.0,
    min_turns: int = 0,
) -> list[dict]:
    if not topic_segments:
        return []
    if min_duration <= 0 and min_turns <= 0:
        return topic_segments

    def topic_duration(topic: dict) -> float | None:
        start = topic.get("start")
        end = topic.get("end")
        if start is None or end is None:
            return None
        return float(end) - float(start)

    def topic_turns(topic: dict) -> int:
        turns = topic.get("turns") or []
        return len(turns)

    def too_small(topic: dict) -> bool:
        duration = topic_duration(topic)
        if min_duration > 0 and duration is not None and duration < min_duration:
            return True
        if min_turns > 0 and topic_turns(topic) < min_turns:
            return True
        return False

    def combine_topics(left: dict, right: dict) -> dict:
        new_start_candidates = [val for val in (left.get("start"), right.get("start")) if val is not None]
        new_end_candidates = [val for val in (left.get("end"), right.get("end")) if val is not None]
        new_start = min(new_start_candidates) if new_start_candidates else None
        new_end = max(new_end_candidates) if new_end_candidates else None

        combined_text = " ".join(
            part.strip()
            for part in [left.get("text", ""), right.get("text", "")]
            if part and part.strip()
        ).strip()

        combined_turns = (left.get("turns") or []) + (right.get("turns") or [])

        combined_counts = Counter(left.get("speaker_counts") or {})
        combined_counts.update(right.get("speaker_counts") or {})
        primary = combined_counts.most_common(1)[0][0] if combined_counts else None

        result = {
            "start": new_start,
            "end": new_end,
            "text": combined_text,
            "turns": combined_turns,
            "speaker_counts": dict(combined_counts),
            "primary_speaker": primary,
        }
        return result

    topics = [topic_segments[0]]
    for topic in topic_segments[1:]:
        if topics and too_small(topic):
            topics[-1] = combine_topics(topics[-1], topic)
        else:
            topics.append(topic)

    changed = True
    while changed and len(topics) > 1:
        changed = False
        if too_small(topics[0]):
            topics[1] = combine_topics(topics[0], topics[1])
            topics.pop(0)
            changed = True
            continue

        merged: list[dict] = [topics[0]]
        for topic in topics[1:]:
            if too_small(topic):
                merged[-1] = combine_topics(merged[-1], topic)
                changed = True
            else:
                merged.append(topic)
        topics = merged

    return topics


# --- Embedding-based segmentation -------------------------------------------------


def _ensure_sentence_transformers() -> None:
    if SentenceTransformer is None or np is None:
        raise ImportError(
            "sentence-transformers is required for embedding-based topic segmentation. "
            "Install it with 'pip install sentence-transformers' and rerun."
        )


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _mean_vector(vectors: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    if isinstance(vectors, np.ndarray):
        if vectors.size == 0:
            return np.zeros(1, dtype=float)
        mean = np.mean(vectors, axis=0)
    else:
        if not vectors:
            return np.zeros(1, dtype=float)
        mean = np.mean(vectors, axis=0)
    return _normalize_vector(mean)


def _find_boundaries_with_embeddings(
    transcript_segments: list[dict],
    *,
    window_size: int,
    model_name: str,
    device: str | None,
    score_quantile: float,
    min_gap: int,
    smooth_size: int,
) -> list[int]:
    if len(transcript_segments) <= 1:
        return []

    embeddings = _encode_segments(transcript_segments, model_name=model_name, device=device)
    assert np is not None  # for type checkers

    similarities: list[float] = []
    for index in range(len(embeddings) - 1):
        left_slice = embeddings[max(0, index - window_size + 1) : index + 1]
        right_slice = embeddings[index + 1 : index + 1 + window_size]
        if right_slice.size == 0:
            right_slice = embeddings[index + 1 :]
        left_vector = _mean_vector(left_slice)
        right_vector = _mean_vector(right_slice)
        similarities.append(float(np.dot(left_vector, right_vector)))

    return _scores_to_boundaries(
        similarities,
        score_quantile=score_quantile,
        min_gap=min_gap,
        smooth_size=smooth_size,
    )


def _find_boundaries_with_changepoint_detection(
    transcript_segments: list[dict],
    *,
    model_name: str,
    device: str | None,
    penalty: float,
    min_size: int,
) -> list[int]:
    _ensure_sentence_transformers()
    embeddings = _encode_segments(transcript_segments, model_name=model_name, device=device)
    assert np is not None  # for type checking

    try:
        import ruptures as rpt
    except ImportError as error:  # pragma: no cover - handled at runtime
        raise ImportError(
            "ruptures is required for changepoint segmentation. Install it with 'pip install ruptures'."
        ) from error

    if len(embeddings) <= 1:
        return []

    algorithm = rpt.KernelCPD(kernel="rbf", min_size=max(1, min_size))
    change_points = algorithm.fit_predict(embeddings, pen=penalty)
    # fit_predict returns segment end indices including len(signal); drop the last value.
    candidate_boundaries = [cp for cp in change_points if 0 < cp < len(embeddings)]

    if not candidate_boundaries:
        return []

    # Remove boundaries that are too close to each other based on min_size.
    filtered_boundaries: list[int] = []
    last_boundary = -min_size
    for boundary in sorted(candidate_boundaries):
        if boundary - last_boundary >= max(1, min_size):
            filtered_boundaries.append(boundary)
            last_boundary = boundary

    return filtered_boundaries


def _encode_segments(
    transcript_segments: list[dict],
    *,
    model_name: str,
    device: str | None,
) -> np.ndarray:
    _ensure_sentence_transformers()
    assert np is not None  # for type checking
    model = SentenceTransformer(model_name, device=device)
    texts = [_segment_text_with_speaker(segment) for segment in transcript_segments]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 1)
    return embeddings


# --- Lexical fallback segmentation ------------------------------------------------


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def aggregate_counts(token_lists: Iterable[list[str]]) -> Counter:
    bucket: Counter = Counter()
    for tokens in token_lists:
        bucket.update(tokens)
    return bucket


def cosine_similarity(counter_a: Counter, counter_b: Counter) -> float:
    if not counter_a or not counter_b:
        return 0.0
    dot_product = sum(counter_a[token] * counter_b.get(token, 0) for token in counter_a)
    magnitude_a = math.sqrt(sum(value * value for value in counter_a.values()))
    magnitude_b = math.sqrt(sum(value * value for value in counter_b.values()))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


def _find_boundaries_with_lexical_similarity(
    transcript_segments: list[dict],
    *,
    window_size: int,
    score_quantile: float,
    min_gap: int,
    smooth_size: int,
) -> list[int]:
    tokenized_segments = [tokenize(_segment_text_with_speaker(segment)) for segment in transcript_segments]
    if len(tokenized_segments) <= 1:
        return []

    similarities: list[float] = []
    for index in range(len(tokenized_segments) - 1):
        left_tokens = tokenized_segments[max(0, index - window_size + 1) : index + 1]
        right_tokens = tokenized_segments[index + 1 : index + 1 + window_size] or tokenized_segments[
            index + 1 :
        ]
        left_counter = aggregate_counts(left_tokens)
        right_counter = aggregate_counts(right_tokens)
        similarities.append(cosine_similarity(left_counter, right_counter))

    return _scores_to_boundaries(
        similarities,
        score_quantile=score_quantile,
        min_gap=min_gap,
        smooth_size=smooth_size,
    )


def _scores_to_boundaries(
    similarities: list[float],
    *,
    score_quantile: float,
    min_gap: int,
    smooth_size: int,
) -> list[int]:
    if not similarities:
        return []

    scores = _convert_similarities_to_scores(similarities)
    smoothed_scores = _smooth_scores(scores, smooth_size)
    threshold = _quantile_threshold(smoothed_scores, score_quantile)

    boundaries: list[int] = []
    last_boundary = -min_gap
    for idx, score in enumerate(smoothed_scores):
        boundary_index = idx + 1
        if score >= threshold and boundary_index - last_boundary >= min_gap:
            boundaries.append(boundary_index)
            last_boundary = boundary_index

    if not boundaries:
        # Fall back to the highest score position if nothing crossed the threshold.
        peak_index = int(max(range(len(smoothed_scores)), key=smoothed_scores.__getitem__))
        boundaries = [peak_index + 1]
    return boundaries


def _convert_similarities_to_scores(similarities: list[float]) -> list[float]:
    # Higher scores mean greater change/discontinuity.
    if np is not None:
        similarities_array = np.array(similarities, dtype=float)
        return (1.0 - similarities_array).clip(min=0.0).tolist()
    return [max(0.0, 1.0 - float(value)) for value in similarities]


def _smooth_scores(scores: list[float], window_size: int) -> list[float]:
    if window_size <= 1 or len(scores) <= 2:
        return scores

    window_size = min(len(scores), window_size)

    if np is not None:
        kernel = np.ones(window_size, dtype=float) / window_size
        smoothed = np.convolve(np.array(scores, dtype=float), kernel, mode="same")
        return smoothed.tolist()

    smoothed: list[float] = []
    half_window = window_size // 2
    for index in range(len(scores)):
        start = max(0, index - half_window)
        end = min(len(scores), start + window_size)
        window = scores[start:end]
        smoothed.append(sum(window) / len(window))
    return smoothed


def _quantile_threshold(scores: list[float], quantile: float) -> float:
    if np is not None:
        return float(np.quantile(np.array(scores, dtype=float), quantile))

    sorted_scores = sorted(scores)
    if not sorted_scores:
        return 1.0
    index = max(0, min(len(sorted_scores) - 1, int(round((len(sorted_scores) - 1) * quantile))))
    return sorted_scores[index]


def _segment_text_with_speaker(segment: dict) -> str:
    text = (segment.get("text") or "").strip()
    speaker = segment.get("speaker")
    if speaker:
        return f"{speaker}: {text}" if text else speaker
    return text
