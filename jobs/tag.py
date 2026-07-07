"""Visual tagging: sample representative frames, then derive
  • dominant colors        (KMeans in CIELAB → named hex)
  • keywords / objects / scenes (OpenCLIP zero-shot against a curated vocabulary)
  • on-screen people       (InsightFace → a PROBABILISTIC gender/age estimate)

Everything is stored as tags + video_tags with a confidence. Gender is an
estimate, never a hard fact — it lands as 'person:likely-male/female' with the
model's probability in `confidence`, and the UI hides low-confidence guesses.

Runs inside the GPU `process_image`. Models load once per container (cached in a
Modal Volume at /root/.cache). RAM++/richer scene models can be added later; this
uses only pip-reliable pieces so the scaffold runs without git-installed weights.
"""
from __future__ import annotations

import collections
import os
import tempfile

import common

# CLIP zero-shot vocabulary — phrased as prompts, mapped to (tag_type, slug, label).
CLIP_VOCAB: list[tuple[str, str, str, str]] = [
    # prompt,                         type,      slug,               label
    ("a luxury car",                  "object",  "object:car",        "car"),
    ("gold chains and jewelry",       "object",  "object:jewelry",    "jewelry"),
    ("a stack of cash / money",       "object",  "object:money",      "money"),
    ("a microphone",                  "object",  "object:microphone", "microphone"),
    ("visible tattoos",               "object",  "object:tattoos",    "tattoos"),
    ("designer / streetwear clothing","object",  "object:streetwear", "streetwear"),
    ("sneakers",                      "object",  "object:sneakers",   "sneakers"),
    ("sunglasses",                    "object",  "object:sunglasses", "sunglasses"),
    ("a luxury watch",                "object",  "object:watch",      "watch"),
    ("a firearm",                     "object",  "object:firearm",    "firearm"),
    ("a city street",                 "scene",   "scene:street",      "street"),
    ("an indoor studio",              "scene",   "scene:indoor",      "indoor"),
    ("outdoors at night",             "scene",   "scene:night",       "night"),
    ("graffiti on a wall",            "scene",   "scene:graffiti",    "graffiti"),
    ("a large crowd of people",       "scene",   "scene:crowd",       "crowd"),
    ("a basketball court",            "scene",   "scene:court",       "basketball court"),
]
CLIP_THRESHOLD = 0.24  # cosine-similarity threshold for keeping a zero-shot tag

# Curated color palette people actually filter by (name → representative RGB).
COLOR_PALETTE = {
    "red": (210, 30, 30), "orange": (235, 140, 30), "yellow": (240, 215, 50),
    "green": (45, 160, 65), "teal": (35, 150, 150), "blue": (45, 95, 205),
    "purple": (130, 60, 190), "pink": (230, 120, 180), "brown": (120, 70, 40),
    "gold": (200, 165, 70), "tan": (210, 190, 150),
    "black": (18, 18, 18), "white": (238, 238, 238), "gray": (128, 128, 128),
}

_clip = {"model": None, "preprocess": None, "tokenizer": None, "text_feats": None}
_face = {"app": None}


# ── frame sampling ───────────────────────────────────────────────────
def _download_video(youtube_id: str, out_dir: str, height: int = 360) -> str:
    tmpl = os.path.join(out_dir, "v.%(ext)s")
    common.ytdlp_download(
        f"https://www.youtube.com/watch?v={youtube_id}",
        {
            # Prefer H.264 (avc1) — OpenCV can't decode YouTube's AV1/VP9 streams.
            "format": (
                f"bv*[height<={height}][vcodec^=avc1]+ba/b[height<={height}][vcodec^=avc1]/"
                f"bv*[height<={height}]+ba/b[height<={height}]/best"
            ),
            "outtmpl": tmpl,
            "merge_output_format": "mp4",
        },
    )
    for f in os.listdir(out_dir):
        if f.startswith("v."):
            return os.path.join(out_dir, f)
    raise FileNotFoundError("tag: video did not download")


def _sample_frames(video_path: str, max_frames: int = 40):
    """One frame per detected scene (fallback: uniform). Returns [(ms, BGR ndarray)]."""
    import cv2

    times_ms: list[int] = []
    try:
        from scenedetect import detect, ContentDetector
        for start, end in detect(video_path, ContentDetector()):
            times_ms.append(int(((start.get_seconds() + end.get_seconds()) / 2) * 1000))
    except Exception:
        times_ms = []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not times_ms:
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        dur = (total / fps) if fps else 0
        step = max(dur / max_frames, 1.0) if dur else 1.0
        t = 0.0
        while t < dur and len(times_ms) < max_frames:
            times_ms.append(int(t * 1000))
            t += step

    times_ms = times_ms[:max_frames]
    frames = []
    for ms in times_ms:
        cap.set(cv2.CAP_PROP_POS_MSEC, ms)
        ok, img = cap.read()
        if ok:
            frames.append((ms, img))
    cap.release()
    return frames


# ── colors ───────────────────────────────────────────────────────────
def _dominant_colors(frames, k: int = 5) -> list[dict]:
    import cv2
    import numpy as np
    from sklearn.cluster import KMeans

    chunks = []
    for _, img in frames:
        small = cv2.resize(img, (64, 36))
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        chunks.append(lab.reshape(-1, 3))
    if not chunks:
        return []

    data = np.concatenate(chunks, axis=0).astype("float32")
    k = min(k, max(1, len(data)))
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(data)
    counts = collections.Counter(km.labels_.tolist())
    total = sum(counts.values())

    out = []
    for idx, cnt in counts.most_common():
        lab = np.uint8([[km.cluster_centers_[idx]]])
        b, g, r = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0][0].tolist()
        out.append({
            "hex": f"#{int(r):02x}{int(g):02x}{int(b):02x}",
            "name": _color_name(int(r), int(g), int(b)),
            "ratio": round(cnt / total, 3),
        })
    return out


def _color_name(r: int, g: int, b: int) -> str:
    """Nearest name in the curated palette (squared RGB distance)."""
    best, bestd = "unknown", 1e18
    for name, (cr, cg, cb) in COLOR_PALETTE.items():
        d = (cr - r) ** 2 + (cg - g) ** 2 + (cb - b) ** 2
        if d < bestd:
            best, bestd = name, d
    return best


# ── FTB logo color ───────────────────────────────────────────────────
def _ftb_logo_color(frames) -> list[dict]:
    """The persistent 'FTB' bug sits in the bottom-left corner; its color varies
    per video. Grab the most consistent VIVID color from that corner (vivid
    filtering + cross-frame aggregation rejects the muted background)."""
    import cv2
    import numpy as np
    from sklearn.cluster import KMeans

    vivid = []
    for _ms, img in frames:
        h, w = img.shape[:2]
        crop = img[int(0.80 * h):h, 0:int(0.17 * w)]
        if crop.size == 0:
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 1] > 95) & (hsv[:, :, 2] > 60)  # saturated + not too dark
        px = crop[mask]
        if len(px) >= 8:
            vivid.append(px.reshape(-1, 3))
    if not vivid:
        return []
    data = np.concatenate(vivid, axis=0)
    if len(data) < 25:
        return []

    lab = cv2.cvtColor(data.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype("float32")
    k = min(2, len(lab))
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(lab)
    counts = collections.Counter(km.labels_.tolist())
    idx = counts.most_common(1)[0][0]
    labc = np.uint8([[km.cluster_centers_[idx]]])
    b, g, r = cv2.cvtColor(labc, cv2.COLOR_LAB2BGR)[0][0].tolist()
    name = _color_name(int(r), int(g), int(b))
    return [{
        "type": "ftb_logo_color",
        "slug": f"ftb:{name.replace(' ', '-')}",
        "label": f"{name} FTB logo",
        "confidence": round(counts[idx] / sum(counts.values()), 3),
        "first_seen_ms": 0,
        "extra": {"hex": f"#{int(r):02x}{int(g):02x}{int(b):02x}"},
    }]


# ── CLIP zero-shot keywords/objects/scenes ───────────────────────────
def _load_clip():
    if _clip["model"] is not None:
        return
    import open_clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    prompts = [f"a photo of {p[0]}" for p in CLIP_VOCAB]
    with torch.no_grad():
        text = tokenizer(prompts).to(device)
        feats = model.encode_text(text)
        feats /= feats.norm(dim=-1, keepdim=True)
    _clip.update(model=model, preprocess=preprocess, tokenizer=tokenizer, text_feats=feats)


def _clip_tags(frames) -> dict[int, int]:
    """Return {vocab_index: first_seen_ms} for vocabulary items seen above threshold."""
    if not frames:
        return {}
    import torch
    from PIL import Image

    _load_clip()
    device = next(_clip["model"].parameters()).device
    seen: dict[int, int] = {}
    with torch.no_grad():
        for ms, img in frames:
            pil = Image.fromarray(img[:, :, ::-1])  # BGR→RGB
            t = _clip["preprocess"](pil).unsqueeze(0).to(device)
            feat = _clip["model"].encode_image(t)
            feat /= feat.norm(dim=-1, keepdim=True)
            sims = (feat @ _clip["text_feats"].T)[0]  # cosine; both sides L2-normalized
            for i, s in enumerate(sims.tolist()):
                if s >= CLIP_THRESHOLD and i not in seen:
                    seen[i] = ms
    return seen


# ── people (probabilistic) ───────────────────────────────────────────
def _load_face():
    if _face["app"] is not None:
        return
    from insightface.app import FaceAnalysis

    fa = FaceAnalysis(name="buffalo_l",
                      providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    fa.prepare(ctx_id=0, det_size=(640, 640))
    _face["app"] = fa


def _detect_faces(frames) -> list[tuple]:
    """Run face detection once; return [(ms, img, faces)] for reuse by people +
    clothing-color tagging (InsightFace on CPU is the slow part — do it once)."""
    if not frames:
        return []
    _load_face()
    fa = _face["app"]
    out = []
    for ms, img in frames:
        try:
            faces = fa.get(img)
        except Exception:
            faces = []
        out.append((ms, img, faces))
    return out


def _clothing_colors(detections) -> list[dict]:
    """Dominant color of the torso region below the largest face per frame,
    aggregated → clothing_color tags. Approximate (color, not garment)."""
    import cv2
    import numpy as np
    from sklearn.cluster import KMeans

    crops = []
    for _ms, img, faces in detections:
        if not faces:
            continue
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = (int(v) for v in f.bbox)
        fw, fh = x2 - x1, y2 - y1
        if fw <= 0 or fh <= 0:
            continue
        h, w = img.shape[:2]
        tx1, tx2 = max(0, int(x1 - 0.35 * fw)), min(w, int(x2 + 0.35 * fw))
        ty1, ty2 = min(h, int(y2 + 0.15 * fh)), min(h, int(y2 + 2.6 * fh))
        if ty2 - ty1 < 12 or tx2 - tx1 < 12:
            continue
        small = cv2.resize(img[ty1:ty2, tx1:tx2], (32, 48))
        crops.append(cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3))
    if not crops:
        return []

    data = np.concatenate(crops, axis=0).astype("float32")
    k = min(3, max(1, len(data)))
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(data)
    counts = collections.Counter(km.labels_.tolist())
    total = sum(counts.values())

    tags, seen = [], set()
    for idx, cnt in counts.most_common(2):  # up to 2 dominant clothing colors
        lab = np.uint8([[km.cluster_centers_[idx]]])
        b, g, r = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0][0].tolist()
        name = _color_name(int(r), int(g), int(b))
        if name in seen:
            continue
        seen.add(name)
        tags.append({
            "type": "clothing_color",
            "slug": f"clothing:{name.replace(' ', '-')}",
            "label": f"{name} clothing",
            "confidence": round(cnt / total, 3),
            "first_seen_ms": 0,
            "extra": {"hex": f"#{int(r):02x}{int(g):02x}{int(b):02x}"},
        })
    return tags


def _people_tags(detections) -> list[dict]:
    """Aggregate face detections into person_attribute tags. Gender/age are ESTIMATES."""
    if not detections:
        return []

    genders = {"male": [], "female": []}  # confidence samples
    first_seen = {"male": None, "female": None}
    ages = []
    any_person_ms = None

    for ms, _img, faces in detections:
        for f in faces:
            if any_person_ms is None:
                any_person_ms = ms
            conf = float(getattr(f, "det_score", 0.8))
            sex = getattr(f, "sex", None) or ("M" if getattr(f, "gender", 1) == 1 else "F")
            key = "male" if str(sex).upper().startswith("M") else "female"
            genders[key].append(conf)
            if first_seen[key] is None:
                first_seen[key] = ms
            if getattr(f, "age", None) is not None:
                ages.append(int(f.age))

    tags: list[dict] = []
    if any_person_ms is not None:
        tags.append({"type": "person_attribute", "slug": "person:present",
                     "label": "people on screen", "confidence": 0.99,
                     "first_seen_ms": any_person_ms})
    for key, samples in genders.items():
        if samples:
            tags.append({
                "type": "person_attribute",
                "slug": f"person:likely-{key}",
                "label": f"likely {key}",
                "confidence": round(sum(samples) / len(samples), 3),
                "first_seen_ms": first_seen[key],
                "extra": {"estimate": True, "count": len(samples)},
            })
    if ages:
        avg = sum(ages) / len(ages)
        bucket = ("teens" if avg < 20 else "20s" if avg < 30 else "30s" if avg < 40 else "40+")
        tags.append({"type": "person_attribute", "slug": f"person:age-{bucket}",
                     "label": f"likely age {bucket}", "confidence": 0.5,
                     "first_seen_ms": any_person_ms, "extra": {"estimate": True}})
    return tags


# ── entrypoint ───────────────────────────────────────────────────────
def tag_video(video_id: str, youtube_id: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        path = _download_video(youtube_id, td)
        frames = _sample_frames(path)

        colors = _dominant_colors(frames)
        clip_seen = _clip_tags(frames)
        detections = _detect_faces(frames)
        people = _people_tags(detections)
        clothing = _clothing_colors(detections)
        ftb = _ftb_logo_color(frames)

    tagspecs: list[dict] = []
    for i, ms in clip_seen.items():
        _, ttype, slug, label = CLIP_VOCAB[i]
        tagspecs.append({"type": ttype, "slug": slug, "label": label,
                         "confidence": 0.6, "first_seen_ms": ms})
    tagspecs.extend(people)
    tagspecs.extend(clothing)
    tagspecs.extend(ftb)
    # dominant colors also become filterable color tags
    for c in colors[:3]:
        tagspecs.append({"type": "color", "slug": f"color:{c['name'].replace(' ', '-')}",
                         "label": c["name"], "confidence": c["ratio"], "first_seen_ms": 0,
                         "extra": {"hex": c["hex"]}})

    with common.db() as conn:
        common.set_dominant_colors(conn, video_id, colors)
        common.apply_tags(conn, video_id, tagspecs)
