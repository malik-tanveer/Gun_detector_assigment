import io
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision
from PIL import Image, ImageOps
from ultralytics import YOLO

st.set_page_config(page_title="Sentinel Vision", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")

MODEL_PATH = Path(__file__).parent / "best.pt"


def _ver():
    try:
        return tuple(int(p) for p in st.__version__.split(".")[:2])
    except Exception:
        return (1, 0)


STRETCH = {"width": "stretch"} if _ver() >= (1, 50) else {"use_container_width": True}


# ---------------- MODEL ----------------
@st.cache_resource
def load_model(path):
    return YOLO(path)


if not MODEL_PATH.exists():
    st.error("best.pt not found. Put it in the same folder as app.py.")
    st.stop()
try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

NAMES = [str(v) for v in model.names.values()]
TERMS = ("gun", "weapon", "pistol", "firearm", "rifle", "handgun", "shotgun", "knife")

# ---------------- STYLE ----------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{{--bg:#0a111d;--surface:#111b2e;--raise:#16233a;--line:#23344f;--text:#e8eefb;--muted:#8fa3c4;--accent:#3dd6c6;--danger:#ff5d6c;}}
.stApp{{background:var(--bg);color:var(--text);font-family:'IBM Plex Sans',sans-serif;}}
.stApp h1,.stApp h2,.stApp h3{{font-family:'Space Grotesk',sans-serif;color:var(--text);}}
[data-testid="stHeader"]{{background:transparent;}}
.block-container{{padding-top:3.4rem;max-width:1200px;}}
[data-testid="stSidebar"]{{background:#0d1524;border-right:1px solid var(--line);}}

[data-testid="stSidebarCollapsedControl"] button,[data-testid="collapsedControl"] button{{
  width:46px;height:46px;border-radius:12px;background:var(--surface);border:1px solid var(--line);font-size:0;}}
[data-testid="stSidebarCollapsedControl"] button > *,[data-testid="collapsedControl"] button > *{{display:none !important;}}
[data-testid="stSidebarCollapsedControl"] button::after,[data-testid="collapsedControl"] button::after{{
  content:"☰";font-size:1.4rem;line-height:1;color:var(--text);}}
[data-testid="stSidebarCollapsedControl"] button:hover,[data-testid="collapsedControl"] button:hover{{border-color:var(--accent);}}

.sv-hero{{display:flex;justify-content:space-between;align-items:center;gap:16px;padding:24px 28px;
  border-radius:20px;background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--accent);margin-bottom:22px;}}
.sv-hero h1{{margin:0;font-size:2rem;letter-spacing:-.015em;}}
.sv-hero p{{margin:4px 0 0;color:var(--muted);}}
.sv-pill{{white-space:nowrap;padding:7px 14px;border-radius:999px;background:var(--raise);border:1px solid var(--line);
  color:var(--accent);font-size:.85rem;font-weight:600;}}

.stTabs [data-baseweb="tab-list"]{{gap:6px;border-bottom:1px solid var(--line);}}
.stTabs [data-baseweb="tab"]{{height:48px;padding:0 20px;color:var(--muted);font-weight:600;}}
.stTabs [aria-selected="true"]{{color:var(--accent);}}

[data-testid="stFileUploaderDropzone"]{{background:var(--surface);border:1.5px dashed #34507a;border-radius:18px;padding:30px;}}
[data-testid="stFileUploaderDropzone"]:hover{{border-color:var(--accent);}}

.sv-card{{padding:16px 18px;border-radius:14px;margin:2px 0 14px;border:1px solid;}}
.sv-card h3{{margin:0 0 6px;font-size:1.05rem;}}
.sv-card p{{margin:0;font-size:.92rem;}}
.sv-danger{{background:#3a1a22;border-color:var(--danger);color:#ffe1e5;}}
.sv-ok{{background:#0f2a2c;border-color:#1f6b66;color:#cdf5f1;}}
.sv-info{{background:var(--raise);border-color:var(--line);color:#d4e3f7;}}
.sv-chip{{display:inline-block;margin:0 6px 8px 0;padding:5px 12px;border-radius:999px;background:var(--raise);
  border:1px solid var(--line);font-size:.85rem;}}
.sv-chip.hot{{border-color:var(--danger);color:#ffb3bc;}}

div[data-testid="stMetric"]{{background:var(--surface);border:1px solid var(--line);padding:12px 14px;border-radius:14px;}}
div[data-testid="stMetricLabel"]{{color:var(--muted);}}
.stButton>button,.stDownloadButton>button{{border-radius:12px;font-weight:600;min-height:46px;}}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{{background:var(--accent);color:#06201d;border:none;}}
[data-testid="stImage"] img,[data-testid="stVideo"] video{{border-radius:16px;border:1px solid var(--line);}}
</style>
<div class="sv-hero">
  <div><h1>Computer Vision</h1>
  <p>Upload an image or video. You get the detection result only, ready to download.</p></div>
  <span class="sv-pill">● Model ready · {len(NAMES)} classes</span>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR (☰ menu) ----------------
LEVELS = {"Strict": 0.50, "Balanced": 0.30, "Sensitive": 0.15, "Max": 0.07}

with st.sidebar:
    st.title("Menu")
    st.subheader("Detection")
    level = st.select_slider("Sensitivity", options=list(LEVELS), value="Sensitive",
                             help="Higher sensitivity finds more objects but may add false alarms.")
    tiles = st.checkbox("Small-object mode (slower)", True,
                        help="Scans the picture in zoomed-in pieces so small or far objects are not missed.")

    st.subheader("Alerts")
    default_alert = [n for n in NAMES if any(t in n.lower() for t in TERMS)] or NAMES
    alert = st.multiselect("Raise an alert for these classes", NAMES, default=default_alert)

    st.subheader("Output")
    boxes = st.checkbox("Draw outlines", True)
    labels = st.checkbox("Show class names", False)
    show_conf = st.checkbox("Show confidence on image", False)
    line = st.slider("Outline width", 1, 5, 2)

    with st.expander("Advanced"):
        imgsz = st.select_slider("Detection size", [640, 800, 960, 1280], value=960)
        iou = st.slider("Overlap filter (IoU)", 0.1, 0.9, 0.45, 0.05)
        tta = st.checkbox("Test-time augmentation (slower)", False)
        frame_step = st.select_slider("Video: analyze every Nth frame", [1, 2, 3, 5], value=2)
    with st.expander("Model classes"):
        st.write(", ".join(NAMES))
    st.success("Model Connected!")

PARAMS = dict(conf=LEVELS[level], iou=iou, imgsz=imgsz, tta=tta, tiles=tiles, boxes=boxes,
              labels=labels, show_conf=show_conf, line=line, alert=tuple(sorted(alert)))
PARAMS_T = tuple(sorted(PARAMS.items()))


# ---------------- DETECTION ----------------
def _starts(total, size, step):
    xs = list(range(0, max(total - size, 0) + 1, step))
    if xs[-1] + size < total:
        xs.append(total - size)
    return xs


def detect(frame, p):
    """frame is BGR. Returns [(x1, y1, x2, y2, conf, class_id), ...]."""
    h, w = frame.shape[:2]
    passes = [(0, 0, w, h)]
    if p["tiles"] and max(h, w) > 800:
        ts = 640
        step = int(ts * 0.7)
        passes += [(x, y, min(x + ts, w), min(y + ts, h))
                   for y in _starts(h, ts, step) for x in _starts(w, ts, step)]
    bs, ss, cs = [], [], []
    for x0, y0, x1, y1 in passes:
        r = model.predict(source=np.ascontiguousarray(frame[y0:y1, x0:x1]), conf=p["conf"], iou=p["iou"],
                          imgsz=p["imgsz"], augment=p["tta"], max_det=300, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        b = r.boxes.xyxy.cpu().numpy().copy()
        b[:, [0, 2]] += x0
        b[:, [1, 3]] += y0
        bs.append(b)
        ss.append(r.boxes.conf.cpu().numpy())
        cs.append(r.boxes.cls.cpu().numpy())
    if not bs:
        return []
    b, s, c = np.concatenate(bs), np.concatenate(ss), np.concatenate(cs)
    if len(passes) > 1:
        keep = torchvision.ops.batched_nms(torch.from_numpy(b), torch.from_numpy(s),
                                           torch.from_numpy(c).long(), p["iou"]).numpy()
        b, s, c = b[keep], s[keep], c[keep]
    return [(int(v[0]), int(v[1]), int(v[2]), int(v[3]), float(sc), int(cl)) for v, sc, cl in zip(b, s, c)]


def draw(frame, dets, p):
    out = frame.copy()
    h, w = out.shape[:2]
    t = max(1, int(round(p["line"] * max(h, w) / 800)))
    scale = max(0.45, max(h, w) / 1600)
    for x1, y1, x2, y2, conf, cl in dets:
        name = str(model.names[cl])
        col = (108, 93, 255) if name in p["alert"] else (198, 214, 61)  # BGR red / teal
        if p["boxes"]:
            cv2.rectangle(out, (x1, y1), (x2, y2), col, t)
        text = ((name if p["labels"] else "") + (f" {conf:.0%}" if p["show_conf"] else "")).strip()
        if text:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            y0 = max(y1 - th - 10, 0)
            cv2.rectangle(out, (x1, y0), (x1 + tw + 10, y0 + th + 10), col, -1)
            cv2.putText(out, text, (x1 + 5, y0 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (10, 20, 30), 1, cv2.LINE_AA)
    return out


def to_rows(dets):
    return [{"Class": str(model.names[cl]), "Confidence": round(conf, 4),
             "X1": x1, "Y1": y1, "X2": x2, "Y2": y2} for x1, y1, x2, y2, conf, cl in dets]


@st.cache_data(show_spinner=False, max_entries=8)
def analyze_image(data: bytes, params: tuple):
    p = dict(params)
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    dets = detect(bgr, p)
    out = Image.fromarray(cv2.cvtColor(draw(bgr, dets, p), cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue(), to_rows(dets)


def show_summary(rows):
    hits = [r for r in rows if r["Class"] in alert]
    if hits:
        best = max(hits, key=lambda r: r["Confidence"])
        st.markdown(f"""<div class="sv-card sv-danger"><h3>⚠️ Alert: {best['Class']} detected</h3>
        <p>Best confidence {best['Confidence']:.1%}. This is an automatic prediction, so please review it manually.</p></div>""",
                    unsafe_allow_html=True)
    elif rows:
        st.markdown("""<div class="sv-card sv-ok"><h3>✅ Nothing from the alert list</h3>
        <p>Other objects were found. See the list below.</p></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="sv-card sv-info"><h3>No objects found</h3>
        <p>Try Sensitivity: Max, keep Small-object mode on, or use a clearer picture.</p></div>""",
                    unsafe_allow_html=True)
    counts = Counter(r["Class"] for r in rows)
    if counts:
        st.markdown("".join(f'<span class="sv-chip{" hot" if k in alert else ""}">{k} · {v}</span>'
                            for k, v in counts.most_common()), unsafe_allow_html=True)
    return hits


def to_h264(src: Path, dst: Path) -> bool:
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return False
    cmd = [exe, "-y", "-i", str(src), "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(dst)]
    try:
        done = subprocess.run(cmd, capture_output=True, timeout=1800)
        return done.returncode == 0 and dst.exists() and dst.stat().st_size > 0
    except Exception:
        return False


def process_video(data: bytes, step: int, p: dict, bar):
    with tempfile.TemporaryDirectory() as tmp:
        src, raw, final = Path(tmp) / "in_video", Path(tmp) / "raw.mp4", Path(tmp) / "out.mp4"
        src.write_bytes(data)
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError("The video could not be opened. Try an MP4 file.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 120:
            fps = 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if total <= 0 or w <= 0 or h <= 0:
            cap.release()
            raise RuntimeError("Could not read the video information.")
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not create the output video.")

        rows, dets, n, analyzed = [], [], 0, 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if n % step == 0:  # in-between frames reuse the last boxes on the NEW frame
                    dets = detect(frame, p)
                    for r in to_rows(dets):
                        r["Frame"], r["Time (s)"] = n, round(n / fps, 2)
                        rows.append(r)
                    analyzed += 1
                writer.write(draw(frame, dets, p))
                n += 1
                if n % 5 == 0 or n == total:
                    pct = min(n / total, 1.0)
                    bar.progress(pct, text=f"Detecting… {int(pct * 100)}%")
        finally:
            cap.release()
            writer.release()

        playable = to_h264(raw, final)
        out = (final if playable else raw).read_bytes()
        if not out:
            raise RuntimeError("The output video is empty.")
        return dict(video=out, rows=rows, duration=n / fps, analyzed=analyzed, playable=playable)


tab_img, tab_vid = st.tabs(["🖼️  Image", "🎬  Video"])

with tab_img:
    up_img = st.file_uploader("Drop an image to detect", type=["jpg", "jpeg", "png", "webp", "bmp"],
                              key="image_upload")
    if up_img:
        try:
            with st.spinner("Detecting…"):
                png, rows = analyze_image(up_img.getvalue(), PARAMS_T)
        except Exception as e:
            st.error(f"Image detection failed: {e}")
            st.stop()

        left, right = st.columns([1.5, 1], gap="large")
        with left:
            st.image(png, **STRETCH)
        with right:
            hits = show_summary(rows)
            a, b, c = st.columns(3)
            a.metric("Objects", len(rows))
            b.metric("Alerts", len(hits))
            c.metric("Top conf.", f"{max((r['Confidence'] for r in rows), default=0):.0%}" if rows else "—")
            stem = Path(up_img.name).stem
            st.download_button("⬇️ Download result image", data=png, file_name=f"{stem}_detected.png",
                               mime="image/png", type="primary", **STRETCH)
            # if rows:
            #     df = pd.DataFrame(rows)
            #     st.download_button("⬇️ Download report (CSV)", data=df.to_csv(index=False).encode("utf-8"),
            #                        file_name=f"{stem}_report.csv", mime="text/csv", **STRETCH)
            #     with st.expander("Detection details"):
            #         st.dataframe(df, hide_index=True, **STRETCH)

with tab_vid:
    up_vid = st.file_uploader("Drop a video to detect", type=["mp4", "mov", "avi", "mkv"], key="video_upload")
    if up_vid:
        if up_vid.size / (1024 * 1024) > 300:
            st.error("This video is over 300 MB. Upload a shorter one.")
        else:
            key = (up_vid.name, up_vid.size, PARAMS_T, frame_step)
            if st.button("Start detection", type="primary", **STRETCH):
                bar = st.progress(0.0, text="Detecting… 0%")
                try:
                    res = process_video(up_vid.getvalue(), frame_step, PARAMS, bar)
                    res["key"] = key
                    st.session_state["video_result"] = res
                except Exception as e:
                    st.session_state.pop("video_result", None)
                    st.error(f"Video detection failed: {e}")
                bar.empty()

            res = st.session_state.get("video_result")
            if res and res["key"] == key:
                rows = res["rows"]
                hits = show_summary(rows)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Duration", f"{res['duration']:.1f} s")
                m2.metric("Frames analyzed", f"{res['analyzed']:,}")
                m3.metric("Detections", f"{len(rows):,}")
                m4.metric("Alert frames", len({r["Frame"] for r in hits}))
                st.video(res["video"])
                stem = Path(up_vid.name).stem
                d1, d2 = st.columns(2)
                d1.download_button("⬇️ Download result video", data=res["video"], file_name=f"{stem}_detected.mp4",
                                   mime="video/mp4", type="primary", **STRETCH)
                if rows:
                    df = pd.DataFrame(rows)
                    d2.download_button("⬇️ Download report (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                                       file_name=f"{stem}_report.csv", mime="text/csv", **STRETCH)
            elif res:
                st.info("The file or settings changed. Press Start detection to run it again.")

st.caption("Results are model predictions, not confirmed findings. Always review them manually.")