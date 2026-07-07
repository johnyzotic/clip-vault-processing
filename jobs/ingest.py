"""Ingest job: enumerate the @4shootersonly channel, keep only From The Block
performances, and upsert them into the `videos` table as 'pending'.

Pure functions here (no Modal decorators) so they're easy to reason about and
call from modal_app.py. Heavy import (yt_dlp) is fine in the ingest image.
"""
from __future__ import annotations

import datetime as dt

import common


def _channel_videos_url() -> str:
    ch = common.youtube_channel()
    if ch.startswith("@"):
        return f"https://www.youtube.com/{ch}/videos"
    return ch


def enumerate_ftb(limit: int = 400) -> list[dict]:
    """Flat-list the channel (cheap, no per-video calls, no bot check), keep From
    The Block titles, newest-first, capped to `limit` (the pilot slice). We only
    scan the first `limit * 4` uploads so smoke tests stay fast. Returns flat entries.
    """
    from yt_dlp import YoutubeDL

    cap = max(limit * 4, 60)
    opts = {
        "quiet": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
        "playlist_items": f"1:{cap}",
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(_channel_videos_url(), download=False)

    entries = info.get("entries") or []
    needle = common.ftb_match()
    ftb = [e for e in entries if e.get("id") and needle in (e.get("title") or "").lower()]
    # A channel's /videos listing is already newest-first.
    return ftb[:limit]


def ingest_one(entry: dict) -> list[str]:
    """Upsert one video straight from its flat listing entry — no per-video
    extract_info (which trips YouTube's bot check from datacenter IPs). Thumbnail
    is derived from the ID. Upload date/description are enriched later during the
    (cookie-enabled) processing step. Returns [video_id, youtube_id].
    """
    yid = entry["id"]
    title = entry.get("title") or yid
    artist, _ = common.parse_artist_title(title)
    duration_ms = int(entry["duration"] * 1000) if entry.get("duration") else None
    thumbnail = f"https://i.ytimg.com/vi/{yid}/hqdefault.jpg"

    with common.db() as conn:
        video_id = common.upsert_video(
            conn,
            youtube_id=yid,
            title=title,
            artist=artist,
            description=None,
            published_at=None,
            duration_ms=duration_ms,
            thumbnail_url=thumbnail,
        )
    return [video_id, yid]


def existing_youtube_ids() -> set[str]:
    with common.db() as conn:
        return {r[0] for r in conn.execute("select youtube_id from videos").fetchall()}
