"""Shared helpers for the Clip Vault Modal jobs.

Config, a Postgres (Neon) connection, small typed write-helpers, and an R2
client. Everything the ingest / transcribe / tag jobs need to talk to the
database lives here so the schema is touched in exactly one place.

Env vars are injected at runtime by a Modal Secret (see modal_app.py). The
same names match `.env` / `.env.example`.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import secrets
from contextlib import contextmanager
from typing import Iterable, Optional

import psycopg


# ── config ───────────────────────────────────────────────────────────
def ftb_match() -> str:
    return (os.environ.get("FTB_TITLE_MATCH") or "from the block").lower()


def youtube_channel() -> str:
    return os.environ.get("YOUTUBE_CHANNEL") or "@4shootersonly"


# ── database ─────────────────────────────────────────────────────────
@contextmanager
def db():
    """Yield a committed-on-success Postgres connection."""
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=False)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_video(
    conn,
    *,
    youtube_id: str,
    title: str,
    artist: Optional[str] = None,
    description: Optional[str] = None,
    published_at: Optional[dt.datetime] = None,
    duration_ms: Optional[int] = None,
    thumbnail_url: Optional[str] = None,
) -> str:
    """Insert or update a video by youtube_id; returns its uuid."""
    row = conn.execute(
        """
        insert into videos (youtube_id, title, artist, description, published_at,
                            duration_ms, thumbnail_url, processing_status)
        values (%s, %s, %s, %s, %s, %s, %s, 'pending')
        on conflict (youtube_id) do update set
          title         = excluded.title,
          artist        = excluded.artist,
          description   = excluded.description,
          published_at  = excluded.published_at,
          duration_ms   = excluded.duration_ms,
          thumbnail_url = excluded.thumbnail_url
        returning id
        """,
        (youtube_id, title, artist, description, published_at, duration_ms, thumbnail_url),
    ).fetchone()
    return str(row[0])


def set_status(conn, video_id: str, status: str, error: Optional[str] = None,
               processed: bool = False) -> None:
    sql = "update videos set processing_status = %s, error = %s"
    if processed:
        sql += ", processed_at = now()"
    sql += " where id = %s"
    conn.execute(sql, (status, error, video_id))


def replace_transcript(conn, video_id: str, lines: Iterable[dict]) -> None:
    """Replace all transcript lines for a video. Each line: {line_index,start_ms,end_ms,text}."""
    conn.execute("delete from transcript_lines where video_id = %s", (video_id,))
    rows = [
        (video_id, l["line_index"], l["start_ms"], l["end_ms"], l["text"])
        for l in lines
        if (l.get("text") or "").strip()
    ]
    if rows:
        with conn.cursor() as cur:
            cur.executemany(
                "insert into transcript_lines (video_id, line_index, start_ms, end_ms, text)"
                " values (%s, %s, %s, %s, %s)",
                rows,
            )


def set_dominant_colors(conn, video_id: str, colors: list) -> None:
    conn.execute("update videos set dominant_colors = %s where id = %s",
                 (json.dumps(colors), video_id))


def upsert_tag(conn, type_: str, slug: str, label: str, extra: Optional[dict] = None) -> str:
    row = conn.execute(
        """
        insert into tags (type, slug, label, extra) values (%s, %s, %s, %s)
        on conflict (slug) do update set label = excluded.label
        returning id
        """,
        (type_, slug, label, json.dumps(extra or {})),
    ).fetchone()
    return str(row[0])


def link_video_tag(conn, video_id: str, tag_id: str, confidence: Optional[float] = None,
                   first_seen_ms: Optional[int] = None, source: str = "auto") -> None:
    conn.execute(
        """
        insert into video_tags (video_id, tag_id, confidence, first_seen_ms, source)
        values (%s, %s, %s, %s, %s)
        on conflict (video_id, tag_id) do update set
          confidence    = greatest(coalesce(video_tags.confidence, 0), coalesce(excluded.confidence, 0)),
          first_seen_ms = least(coalesce(video_tags.first_seen_ms, 2147483647),
                                coalesce(excluded.first_seen_ms, 2147483647))
        """,
        (video_id, tag_id, confidence, first_seen_ms, source),
    )


def apply_tags(conn, video_id: str, tagspecs: Iterable[dict]) -> None:
    """tagspec: {type, slug, label, confidence?, first_seen_ms?, extra?}."""
    for t in tagspecs:
        tid = upsert_tag(conn, t["type"], t["slug"], t["label"], t.get("extra"))
        link_video_tag(conn, video_id, tid, t.get("confidence"), t.get("first_seen_ms"))


# ── title parsing ────────────────────────────────────────────────────
def parse_artist_title(title: str) -> tuple[Optional[str], str]:
    """From The Block titles look like:  'Artist - Song | From The Block Performance 🎥'.
    Best-effort extraction of the artist; the full title is always kept.
    """
    head = re.split(r"\s*[|(\[]", title, maxsplit=1)[0].strip()  # drop the '| From The Block…' tail
    artist = None
    if " - " in head:
        artist = head.split(" - ", 1)[0].strip()
    elif " x " in head.lower():
        artist = head.strip()
    return (artist or None), title


# ── yt-dlp options (TV player client + optional cookies) ────────────
# The TV client needs no PO token; cookies (a throwaway account, stored in the
# YT_COOKIES secret) defeat the datacenter-IP bot wall and unlock non-DRM formats.
def ytdlp_cookie_file() -> Optional[str]:
    """Materialize the YT_COOKIES secret to a Netscape cookies file; returns the
    path, or None when no real cookies are configured (placeholder/empty)."""
    data = os.environ.get("YT_COOKIES", "")
    if "\t" not in data:  # a real cookies.txt is tab-separated; placeholder isn't
        return None
    path = "/tmp/yt-cookies.txt"
    with open(path, "w") as f:
        f.write(data)
    return path


def _sticky_proxy(url: str) -> str:
    """Enrich a DataImpulse proxy URL with a per-call sticky session + US geo, so a
    single yt-dlp download uses ONE residential IP for both the format extract and
    the media fetch. YouTube 403s the download if its IP differs from the extract's,
    which is what a plain rotating endpoint causes."""
    m = re.match(r"^(https?://)([^:@/]+)(:[^@]*@.+)$", url)
    if not m or "sessid." in url:
        return url
    scheme, user, rest = m.groups()
    return f"{scheme}{user}__cr.us;sessid.{secrets.token_hex(5)}{rest}"


def ytdlp_base_opts() -> dict:
    """Shared yt-dlp options for media downloads from a datacenter IP."""
    opts = {
        "quiet": True,
        "retries": 2,  # a flagged IP fails fast; the outer retry rotates to a new IP
        "sleep_interval_requests": 1,
        "extractor_args": {"youtube": {"player_client": ["android_vr", "default"]}},
    }
    cookies = ytdlp_cookie_file()
    if cookies:
        opts["cookiefile"] = cookies
    proxy = os.environ.get("YT_PROXY")  # residential proxy — YouTube blocks datacenter IPs
    if proxy:
        opts["proxy"] = _sticky_proxy(proxy)
    return opts


def ytdlp_download(url: str, extra_opts: dict, attempts: int = 3) -> None:
    """Download with yt-dlp, rotating to a FRESH residential IP each attempt.
    A given DataImpulse IP may be YouTube-flagged (all its retries fail); a new
    sticky session usually lands on a clean IP, so retrying end-to-end recovers
    most transient bot-wall / 403 failures."""
    from yt_dlp import YoutubeDL

    last: Exception | None = None
    for _ in range(attempts):
        try:
            with YoutubeDL({**ytdlp_base_opts(), **extra_opts}) as ydl:
                ydl.download([url])
            return
        except Exception as e:  # noqa: BLE001
            last = e
    raise last if last else RuntimeError("download failed")


# ── R2 (used in Phase 2 for audio/proxy/render output; harmless now) ─
def r2_client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def r2_put(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    r2_client().put_object(Bucket=os.environ["R2_BUCKET"], Key=key, Body=data,
                           ContentType=content_type)
    return key
