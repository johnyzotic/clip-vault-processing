"""Transcription: download the audio, isolate the vocal over the beat (Demucs),
and transcribe with faster-whisper (built-in Silero VAD, segment timestamps).

We use faster-whisper directly rather than whisperx to avoid the
pyannote.audio / torchaudio.AudioMetaData version conflict. Segment-level
timestamps are enough for lyric search + jump-to-moment.

Runs inside the GPU process image (see modal_app.py). Heavy libs import lazily.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import common

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")  # or "large-v3-turbo" for speed


def _download_audio(youtube_id: str, out_dir: str) -> str:
    tmpl = os.path.join(out_dir, "audio.%(ext)s")
    common.ytdlp_download(
        f"https://www.youtube.com/watch?v={youtube_id}",
        {
            "format": "bestaudio/best",
            "outtmpl": tmpl,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        },
    )
    wav = os.path.join(out_dir, "audio.wav")
    if not os.path.exists(wav):
        for f in os.listdir(out_dir):
            if f.startswith("audio."):
                return os.path.join(out_dir, f)
    return wav


def _isolate_vocals(wav_path: str, out_dir: str) -> str:
    """Demucs two-stem split; return the vocals stem, or the original on failure."""
    try:
        subprocess.run(
            ["python", "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs",
             "-o", out_dir, wav_path],
            check=True, capture_output=True,
        )
        stem = os.path.splitext(os.path.basename(wav_path))[0]
        vocals = os.path.join(out_dir, "htdemucs", stem, "vocals.wav")
        if os.path.exists(vocals):
            return vocals
    except Exception:
        pass
    return wav_path  # fall back to the mixed track


def transcribe_video(video_id: str, youtube_id: str) -> int:
    """Write transcript_lines for a video. Returns the number of lines written."""
    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    with tempfile.TemporaryDirectory() as td:
        wav = _download_audio(youtube_id, td)
        vocals = _isolate_vocals(wav, td)

        model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
        # No VAD: Demucs already isolated the vocal, and Silero VAD tends to cut
        # rap-over-beats down to a few intro bars. condition_on_previous_text=False
        # avoids repeat-loops on hooks.
        segments, _info = model.transcribe(
            vocals, beam_size=5, condition_on_previous_text=False,
        )

        lines = []
        for i, seg in enumerate(segments):  # generator — this drives the transcription
            text = (seg.text or "").strip()
            if not text:
                continue
            lines.append({
                "line_index": i,
                "start_ms": int(seg.start * 1000),
                "end_ms": int(seg.end * 1000),
                "text": text,
            })

    with common.db() as conn:
        common.replace_transcript(conn, video_id, lines)
    return len(lines)
