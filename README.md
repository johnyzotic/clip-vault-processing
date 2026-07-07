# clip-vault-processing

Bulk video-processing workers for an internal media-search tool. This repo is
**public only so GitHub Actions minutes are unlimited** — it contains generic,
non-sensitive scripts (yt-dlp + Whisper + ffmpeg + OpenCLIP/InsightFace glue).

**No credentials live here.** The database URL, proxy, and cookies are provided
at runtime as encrypted GitHub Actions **secrets** and are never printed.

## What it does

Many parallel runners share one work queue (a `videos` table). Each claims
pending rows with `FOR UPDATE SKIP LOCKED`, then per video:

1. download audio, isolate vocals (Demucs), transcribe (faster-whisper) → timestamped lines
2. sample frames → dominant colors (CIELAB k-means), zero-shot tags (OpenCLIP),
   probabilistic on-screen person attributes (InsightFace)
3. write transcript + tags to the database; mark the row `ready`

## Run

Actions → **backfill** → *Run workflow*. Inputs control parallelism
(`workers_json`), per-worker caps, time budget, and the Whisper model.
