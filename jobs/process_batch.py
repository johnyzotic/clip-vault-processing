"""Backfill worker for GitHub Actions (CPU runners, no GPU, no Modal).

Many runners share ONE queue — the `videos` table. Each claims pending rows with
`FOR UPDATE SKIP LOCKED` (so 20 parallel jobs never grab the same video), runs
transcribe + tag, and marks the row ready/error. A runner keeps draining until
the queue is empty, a time budget is hit (stay under GitHub's 6h job cap), or too
many downloads fail in a row (usually a spent proxy).

Stale claims (a runner that died mid-video, leaving a row in 'transcribing'/
'tagging') are reset to 'pending' BETWEEN waves by the orchestrator, when no
runners are live — so no lease column is needed.

Env: DATABASE_URL, YT_PROXY, YT_COOKIES (+ optional TIME_BUDGET_S, MAX_VIDEOS).
"""
from __future__ import annotations

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from transcribe import transcribe_video  # noqa: E402
from tag import tag_video  # noqa: E402

TIME_BUDGET_S = int(os.environ.get("TIME_BUDGET_S", "17000"))  # ~4.7h < 6h cap
MAX_VIDEOS = int(os.environ.get("MAX_VIDEOS", "0"))            # 0 = until budget/empty
MAX_CONSEC_FAILS = int(os.environ.get("MAX_CONSEC_FAILS", "6"))


def claim_one() -> tuple[str, str] | None:
    """Atomically grab one pending/error video and mark it 'transcribing'."""
    with common.db() as conn:
        row = conn.execute(
            """
            update videos set processing_status = 'transcribing'
             where id = (
               select id from videos
                where processing_status in ('pending', 'error')
                order by ingested_at desc
                for update skip locked
                limit 1
             )
            returning id, youtube_id
            """
        ).fetchone()
    return (str(row[0]), row[1]) if row else None


def process_one(video_id: str, youtube_id: str) -> bool:
    try:
        n = transcribe_video(video_id, youtube_id)
        with common.db() as conn:
            common.set_status(conn, video_id, "tagging")
        tag_video(video_id, youtube_id)
        with common.db() as conn:
            common.set_status(conn, video_id, "ready", processed=True)
        print(f"[backfill] {youtube_id}: {n} lines + tags -> ready", flush=True)
        return True
    except Exception as ex:  # noqa: BLE001 — record + keep going
        msg = str(ex)[:500]
        with common.db() as conn:
            common.set_status(conn, video_id, "error", error=msg)
        print(f"[backfill] {youtube_id} ERROR: {msg}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return False


def main() -> None:
    start = time.time()
    done = 0
    consec_fails = 0
    while time.time() - start < TIME_BUDGET_S:
        if MAX_VIDEOS and done >= MAX_VIDEOS:
            print(f"[backfill] hit MAX_VIDEOS={MAX_VIDEOS}", flush=True)
            break
        claimed = claim_one()
        if not claimed:
            print("[backfill] queue empty — nothing left to do", flush=True)
            break
        ok = process_one(*claimed)
        done += 1
        consec_fails = 0 if ok else consec_fails + 1
        if consec_fails >= MAX_CONSEC_FAILS:
            print(
                f"[backfill] {consec_fails} failures in a row — stopping "
                "(proxy credit or upstream likely exhausted)",
                file=sys.stderr,
                flush=True,
            )
            break
    print(f"[backfill] processed {done} video(s) this run", flush=True)


if __name__ == "__main__":
    main()
