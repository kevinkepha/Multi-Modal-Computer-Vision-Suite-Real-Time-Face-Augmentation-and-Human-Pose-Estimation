"""
Multi-Modal Computer Vision Suite: Real-Time Face Augmentation and Human Pose Estimation
---------------------------------------------------------------------------------------
- Tab 1: Face Augmentation (emoji/brand mark overlays) using OpenCV Haar cascades
- Tab 2: Human Pose Estimation (keypoints & skeleton) using Ultralytics YOLOv8

Author: Duncan
"""

import os
import cv2
import numpy as np
import random
import gradio as gr
from typing import List, Tuple, Optional

# -----------------------
# Model & Asset Loading
# -----------------------

# Face detector (kept for speed and simplicity; easy to swap to DNN if needed)
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load YOLOv8 pose lazily to keep startup snappy (created on first use)
_yolo_pose_model = None
def get_yolo_pose():
    global _yolo_pose_model
    if _yolo_pose_model is None:
        from ultralytics import YOLO
        # Small, fast model suitable for laptops; switch to yolov8s/ m/ l for accuracy
        _yolo_pose_model = YOLO("yolov8n-pose.pt")
    return _yolo_pose_model


def load_rgba(path: str) -> Optional[np.ndarray]:
    """Load an RGBA image (emoji/overlay). Returns None if missing."""
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # Ensure 4 channels
    if img is None:
        return None
    if img.shape[-1] == 3:
        # If no alpha, synthesize opaque alpha
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=-1)
    return img


# Default asset directory
ASSETS_DIR = "assets"
DEFAULT_EMOJI_FILES = [
    "emoji_laugh.png",
    "emoji_sunglasses.png",
    "emoji_mindblown.png",
]

EMOJI_PATHS = [os.path.join(ASSETS_DIR, f) for f in DEFAULT_EMOJI_FILES]
EMOJIS = [(os.path.basename(p), load_rgba(p)) for p in EMOJI_PATHS]
# Filter out missing files gracefully
EMOJIS = [(n, im) for (n, im) in EMOJIS if im is not None]

if not EMOJIS:
    # Create a simple fallback RGBA circle overlay if no assets are found
    fallback = np.zeros((256, 256, 4), dtype=np.uint8)
    cv2.circle(fallback, (128, 128), 110, (0, 255, 255, 255), -1)  # yellow circle
    cv2.circle(fallback, (95, 110), 15, (0, 0, 0, 255), -1)
    cv2.circle(fallback, (160, 110), 15, (0, 0, 0, 255), -1)
    cv2.ellipse(fallback, (128, 160), (60, 30), 0, 0, 180, (0, 0, 0, 255), 12)
    EMOJIS = [("fallback_smiley.png", fallback)]


# -----------------------
# Computer Vision Utils
# -----------------------

def alpha_blend_rgba(
    frame_bgr: np.ndarray, overlay_rgba: np.ndarray, top_left: Tuple[int, int], scale_w: int
) -> np.ndarray:
    """
    Efficient alpha blending of an RGBA overlay on a BGR frame.

    Args:
        frame_bgr: target frame (H, W, 3) uint8
        overlay_rgba: source with alpha (h, w, 4) uint8
        top_left: (x, y) on frame where overlay top-left should be placed
        scale_w: desired overlay width in pixels (height scaled to keep aspect ratio)
    """
    x, y = top_left
    h, w = overlay_rgba.shape[:2]

    # Scale overlay keeping aspect ratio
    if scale_w <= 0:  # guard
        return frame_bgr
    scale = scale_w / float(w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    overlay_rgba = cv2.resize(overlay_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    H, W = frame_bgr.shape[:2]
    oh, ow = overlay_rgba.shape[:2]

    # Clip if going out of bounds
    if x >= W or y >= H:
        return frame_bgr
    x2 = min(W, x + ow)
    y2 = min(H, y + oh)
    ox1, oy1 = max(0, x), max(0, y)
    ox2, oy2 = x2, y2

    # Corresponding overlay region
    sx1, sy1 = ox1 - x, oy1 - y
    sx2, sy2 = sx1 + (ox2 - ox1), sy1 + (oy2 - oy1)

    roi = frame_bgr[oy1:oy2, ox1:ox2]  # (h, w, 3)
    overlay_crop = overlay_rgba[sy1:sy2, sx1:sx2]  # (h, w, 4)

    overlay_rgb = overlay_crop[..., :3].astype(np.float32)
    alpha = overlay_crop[..., 3:4].astype(np.float32) / 255.0  # (h, w, 1)
    roi_float = roi.astype(np.float32)

    blended = alpha * overlay_rgb + (1.0 - alpha) * roi_float
    frame_bgr[oy1:oy2, ox1:ox2] = blended.astype(np.uint8)
    return frame_bgr


def detect_faces_bgr(frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


# -----------------------
# Pipelines (Gradio fns)
# -----------------------

def face_augmentation(
    image: np.ndarray,
    emoji_name: str,
    overlay_scale: float = 1.0,
    randomize_each_frame: bool = True,
    max_faces: int = 5,
) -> np.ndarray:
    """
    Replace detected faces with selected overlay (emoji/brand mark).
    Args:
        image: RGB image from Gradio
        emoji_name: which overlay to use (from dropdown)
        overlay_scale: multiplicative scale relative to face width (0.6–1.6 typical)
        randomize_each_frame: if True, random overlay each frame
        max_faces: cap processing for speed
    """
    if image is None:
        return None

    # Convert to BGR for OpenCV ops
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Select overlay
    selected_overlays = [im for (name, im) in EMOJIS if name == emoji_name]
    overlay_rgba = selected_overlays[0] if selected_overlays else EMOJIS[0][1]

    faces = detect_faces_bgr(frame)[:max_faces]

    for (x, y, w, h) in faces:
        if randomize_each_frame:
            overlay_rgba = random.choice(EMOJIS)[1]
        target_w = int(w * float(overlay_scale))
        # position slightly above face for a “mask” feel
        top_left = (x, max(0, y - int(0.15 * h)))
        frame = alpha_blend_rgba(frame, overlay_rgba, top_left, scale_w=target_w)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def pose_estimation(image: np.ndarray, box_score: float = 0.25, keypoint_score: float = 0.25) -> np.ndarray:
    """
    Run YOLOv8 pose estimation and return annotated frame.
    Args:
        image: RGB image from Gradio
        box_score: confidence threshold for detections
        keypoint_score: confidence threshold for keypoints
    """
    if image is None:
        return None

    frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    model = get_yolo_pose()
    results = model.predict(frame_bgr, conf=box_score, imgsz=640, verbose=False)
    annotated = results[0].plot(kpt_line=True, conf=keypoint_score)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


# -----------------------
# Gradio App (Multi-Tab)
# -----------------------

def build_app():
    with gr.Blocks(title="Multi-Modal Computer Vision Suite") as demo:
        gr.Markdown(
            """
# **Multi-Modal Computer Vision Suite**
**Real-Time Face Augmentation and Human Pose Estimation**

- **Face Augmentation:** fast face detection + alpha-blended overlays (emojis / brand marks)
- **Pose Estimation:** YOLOv8 keypoints & skeletons for human motion analytics
            """
        )

        with gr.Tab("🎭 Face Augmentation"):
            gr.Markdown("Upload a frame or use your webcam. Select an overlay and adjust scaling.")
            with gr.Row():
                inp = gr.Image(sources=["webcam", "upload"], type="numpy", label="Input", streaming=True)
                out = gr.Image(label="Augmented Output", type="numpy")

            emoji_choices = [name for (name, _) in EMOJIS]
            with gr.Row():
                emoji_dropdown = gr.Dropdown(choices=emoji_choices, value=emoji_choices[0], label="Overlay")
                scale_slider = gr.Slider(0.6, 1.6, value=1.0, step=0.05, label="Overlay Scale (× face width)")
            with gr.Row():
                randomize = gr.Checkbox(value=True, label="Randomize overlay each frame")
                max_faces = gr.Slider(1, 10, value=5, step=1, label="Max faces to process")

            # Live processing
            inp.change(
                fn=face_augmentation,
                inputs=[inp, emoji_dropdown, scale_slider, randomize, max_faces],
                outputs=out,
                queue=True,
            ).then(None, None, None)  # keep queue

        with gr.Tab("🏃 Pose Estimation (YOLOv8)"):
            gr.Markdown("Real-time human keypoint detection & skeletal visualization.")
            with gr.Row():
                pose_in = gr.Image(sources=["webcam", "upload"], type="numpy", label="Input", streaming=True)
                pose_out = gr.Image(label="Pose Output", type="numpy")
            with gr.Row():
                conf = gr.Slider(0.1, 0.75, value=0.25, step=0.05, label="Box Confidence Threshold")
                kpt_conf = gr.Slider(0.1, 0.75, value=0.25, step=0.05, label="Keypoint Confidence Threshold")

            pose_in.change(
                fn=pose_estimation,
                inputs=[pose_in, conf, kpt_conf],
                outputs=pose_out,
                queue=True,
            )

        gr.Markdown(
            "Built with **OpenCV**, **Ultralytics YOLOv8**, and **Gradio**. © Duncan"
        )
    return demo


if __name__ == "__main__":
    app = build_app()
    # Set server_name="0.0.0.0" to expose externally (e.g., on a cloud VM)
    app.launch()
