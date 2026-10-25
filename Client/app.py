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
from PIL import Image, ImageOps
from ultralytics import YOLO

st.set_page_config(
    page_title="Sentinel Vision",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "best.pt"
MAX_VIDEO_MB = 300
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]

# ---------------------------- STYLE ----------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #0a111d; --surface: #111b2e; --raised: #16233a;
        --line: #263853; --text: #e8eefb; --muted: #9aabc5;
        --accent: #49d9c8; --danger: #ff6474;
    }
    .stApp { background: var(--bg); color: var(--text); }
    .block-container { max-width: 1250px; padding-top: 2rem; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #0d1524; border-right: 1px solid var(--line); }
    .hero {
        padding: 24px 26px; margin-bottom: 20px; border-radius: 18px;
        border: 1px solid var(--line); border-left: 4px solid var(--accent);
        background: var(--surface);
    }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p { margin: 6px 0 0; color: var(--muted); }
    .hint { color: var(--muted); font-size: .9rem; }
    div[data-testid="stMetric"] {
        background: var(--surface); border: 1px solid var(--line);
        padding: 12px 14px; border-radius: 12px;
    }
    .stButton > button, .stDownloadButton > button { border-radius: 10px; min-height: 42px; }
    [data-testid="stImage"] img, [data-testid="stVideo"] video {
        border-radius: 14px; border: 1px solid var(--line);
    }
    </style>
    <div class="hero">
      <h1>Sentinel Vision</h1>
      <p>Upload an image or video, review model predictions, and download the processed result.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------- MODEL ----------------------------
if not MODEL_PATH.is_file():
    st.error("Model file `best.pt` was not found. Put it in the same folder as `app.py`.")
    st.stop()

DEVICE = 0 if torch.cuda.is_available() else "cpu"

@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(model_path: str):
    return YOLO(model_path)

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Model could not be loaded: {exc}")
    st.stop()

class_names = [str(name) for name in model.names.values()]
device_label = "CUDA GPU" if torch.cuda.is_available() else "CPU"

# ---------------------------- SIDEBAR ----------------------------
with st.sidebar:
    st.title("Settings")
    st.caption(f"Compute device: **{device_label}**")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    image_size = st.select_slider("Image size", options=[320, 480, 640, 800, 960], value=640)
    draw_boxes = st.checkbox("Draw boxes", value=True)
    show_labels = st.checkbox("Show class names", value=True)
    show_confidence = st.checkbox("Show confidence", value=True)
    line_width = st.slider("Box line width", 1, 5, 2)
    alert_classes = st.multiselect(
        "Highlight selected classes",
        options=class_names,
        default=[],
        help="Optional: selected classes are highlighted in the summary.",
    )
    video_step = st.select_slider(
        "Analyze every Nth video frame",
        options=[1, 2, 3, 5],
        value=2,
        help="1 analyzes every frame but is slower. Larger values run faster.",
    )
    st.caption("GPU is used only when the hosting machine actually provides CUDA.")

# ---------------------------- DETECTION ----------------------------
def detect(frame: np.ndarray):
    """Return detections as x1, y1, x2, y2, confidence, class_id."""
    result = model.predict(
        source=frame,
        conf=confidence,
        imgsz=image_size,
        device=DEVICE,
        verbose=False,
        max_det=300,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    return [
        (int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(score), int(cls))
        for box, score, cls in zip(boxes, scores, classes)
    ]


def render(frame: np.ndarray, detections):
    output = frame.copy()
    height, width = output.shape[:2]
    thickness = max(1, round(line_width * max(height, width) / 1000))
    font_scale = max(0.45, max(height, width) / 1800)
    for x1, y1, x2, y2, score, class_id in detections:
        name = str(model.names[class_id])
        highlighted = name in alert_classes
        color = (90, 90, 255) if highlighted else (65, 215, 190)  # BGR
        if draw_boxes:
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        label_parts = []
        if show_labels:
            label_parts.append(name)
        if show_confidence:
            label_parts.append(f"{score:.0%}")
        label = " ".join(label_parts)
        if label:
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            top = max(0, y1 - text_h - 8)
            cv2.rectangle(output, (x1, top), (x1 + text_w + 8, top + text_h + 8), color, -1)
            cv2.putText(
                output, label, (x1 + 4, top + text_h + 3),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (10, 20, 30), 1, cv2.LINE_AA
            )
    return output


def rows_from_detections(detections):
    return [
        {
            "Class": str(model.names[class_id]),
            "Confidence": round(score, 4),
            "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
        }
        for x1, y1, x2, y2, score, class_id in detections
    ]


def show_summary(rows):
    if not rows:
        st.info("No objects were returned by the model for this file.")
        return []
    counts = Counter(row["Class"] for row in rows)
    selected = [row for row in rows if row["Class"] in alert_classes]
    if selected:
        st.warning(
            f"Selected class match: {len(selected)} prediction(s). "
            "Review the image/video manually; model output can be incorrect."
        )
    else:
        st.success("Processing complete. Review the predictions below.")
    st.write(" · ".join(f"**{name}**: {count}" for name, count in counts.most_common()))
    return selected


@st.cache_data(show_spinner=False, max_entries=4)
def analyze_image(image_bytes: bytes, _settings_key: tuple):
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    detections = detect(bgr)
    annotated = render(bgr, detections)
    output_image = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    output_image.save(buffer, format="PNG")
    return buffer.getvalue(), rows_from_detections(detections)


def encode_mp4(source: Path, destination: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return False
    command = [
        ffmpeg, "-y", "-i", str(source), "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=1800)
        return completed.returncode == 0 and destination.exists() and destination.stat().st_size > 0
    except (OSError, subprocess.SubprocessError):
        return False


def process_video(video_bytes: bytes, frame_step: int):
    with tempfile.TemporaryDirectory() as temporary_dir:
        folder = Path(temporary_dir)
        input_path = folder / "input_video"
        raw_path = folder / "processed_raw.mp4"
        final_path = folder / "processed.mp4"
        input_path.write_bytes(video_bytes)

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError("Could not open this video. Try converting it to MP4.")

        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not fps or fps <= 0 or fps > 120:
            fps = 25.0
        if total_frames <= 0 or width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError("The video metadata could not be read.")

        writer = cv2.VideoWriter(
            str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Could not create the output video.")

        all_rows = []
        last_detections = []
        frame_index = 0
        analyzed_frames = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % frame_step == 0:
                    last_detections = detect(frame)
                    for row in rows_from_detections(last_detections):
                        row["Frame"] = frame_index
                        row["Time (s)"] = round(frame_index / fps, 2)
                        all_rows.append(row)
                    analyzed_frames += 1

                writer.write(render(frame, last_detections))
                frame_index += 1
        finally:
            capture.release()
            writer.release()

        chosen_path = final_path if encode_mp4(raw_path, final_path) else raw_path
        result_bytes = chosen_path.read_bytes()
        if not result_bytes:
            raise RuntimeError("The processed video is empty.")
        return {
            "video": result_bytes,
            "rows": all_rows,
            "duration": frame_index / fps,
            "analyzed": analyzed_frames,
            "mime": "video/mp4",
        }


# ---------------------------- MAIN UI ----------------------------
image_tab, video_tab = st.tabs(["🖼️ Image", "🎬 Video"])

settings_key = (
    confidence, image_size, draw_boxes, show_labels, show_confidence,
    line_width, tuple(sorted(alert_classes)), DEVICE,
)

with image_tab:
    uploaded_image = st.file_uploader(
        "Upload an image", type=IMAGE_TYPES, key="image_file"
    )
    if uploaded_image is not None:
        try:
            png_bytes, image_rows = analyze_image(uploaded_image.getvalue(), settings_key)
            left, right = st.columns([1.4, 1], gap="large")
            with left:
                st.image(png_bytes, caption="Processed image", use_container_width=True)
            with right:
                selected_rows = show_summary(image_rows)
                metric1, metric2, metric3 = st.columns(3)
                metric1.metric("Objects", len(image_rows))
                metric2.metric("Highlighted", len(selected_rows))
                metric3.metric(
                    "Top confidence",
                    f"{max((r['Confidence'] for r in image_rows), default=0):.0%}"
                    if image_rows else "—",
                )
        except Exception as exc:
            st.error(f"Image processing failed: {exc}")


with video_tab:
    uploaded_video = st.file_uploader(
        "Upload a video (maximum 300 MB)",
        type=VIDEO_TYPES,
        key="video_file",
    )

    if uploaded_video is not None:
        if uploaded_video.size > MAX_VIDEO_MB * 1024 * 1024:
            st.error(f"Please upload a video smaller than {MAX_VIDEO_MB} MB.")

        else:
            video_key = (
                uploaded_video.name,
                uploaded_video.size,
                settings_key,
                video_step,
            )

            cached_result = st.session_state.get("video_result")

            # Clear old result when video or settings change
            if cached_result and cached_result.get("key") != video_key:
                st.session_state.pop("video_result", None)
                cached_result = None

            # Automatically process after upload
            if cached_result is None:
                try:
                    result = process_video(
                        uploaded_video.getvalue(),
                        video_step,
                    )

                    result["key"] = video_key
                    st.session_state["video_result"] = result
                    cached_result = result

                except Exception as exc:
                    st.session_state.pop("video_result", None)
                    cached_result = None
                    st.error(f"Video processing failed: {exc}")

            # Display result
            if cached_result and cached_result.get("key") == video_key:
                video_rows = cached_result["rows"]
                highlighted = show_summary(video_rows)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Duration", f"{cached_result['duration']:.1f} s")
                m2.metric("Frames analyzed", f"{cached_result['analyzed']:,}")
                m3.metric("Predictions", f"{len(video_rows):,}")
                m4.metric("Highlighted", len(highlighted))

                st.video(cached_result["video"])


                st.download_button(
                    "Download video",
                    data=cached_result["video"],
                    file_name=f"video_processed.mp4",
                    mime=cached_result["mime"],
                    type="primary",
                    use_container_width=True,
                )


st.divider()
st.caption(
    "Model predictions can be wrong and are not verified facts. "
    "Review all results manually. A GPU is available only when the hosting environment provides CUDA."
)
