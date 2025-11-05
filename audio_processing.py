from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple


def preprocess_audio(
    audio_path: str,
    *,
    target_sr: int = 16000,
    trim_db: float | None = 30.0,
    highpass_hz: float | None = 60.0,
    lowpass_hz: float | None = 6000.0,
) -> Tuple[Path, bool]:
    try:
        import librosa
        import numpy as np
        import soundfile as sf
    except ImportError:
        print("Audio preprocessing skipped: install librosa and soundfile to enable cleaning.")
        return Path(audio_path), False

    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    if trim_db is not None:
        y, _ = librosa.effects.trim(y, top_db=trim_db)

    y = librosa.effects.preemphasis(y)
    if y.size == 0:
        y = librosa.util.normalize(np.zeros(1, dtype=float))
    else:
        y = librosa.util.normalize(y)

    try:
        import torch
        import torchaudio

        waveform = torch.tensor(y).unsqueeze(0)
        if highpass_hz and highpass_hz > 0:
            waveform = torchaudio.functional.highpass_biquad(waveform, target_sr, highpass_hz)
        if lowpass_hz and lowpass_hz > 0 and lowpass_hz < target_sr / 2:
            waveform = torchaudio.functional.lowpass_biquad(waveform, target_sr, lowpass_hz)
        y = waveform.squeeze(0).numpy()
    except Exception:
        pass

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_path = Path(tmp.name)
    sf.write(str(temp_path), y, target_sr)
    return temp_path, True
