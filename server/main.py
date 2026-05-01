from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import cv2
import io
import tempfile
import os
import subprocess

app = FastAPI()

# Enable CORS (important for React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = Path(__file__).resolve().parent.parent
model = YOLO(str(base_dir / 'server' / 'model' / 'traffic_sign_v2' / 'weights' / 'best.pt'))

# ---------------- HELPERS ----------------

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def is_image(file: UploadFile):
    if file.content_type and file.content_type.startswith("image/"):
        return True
    return file.filename.lower().endswith(IMAGE_EXTS)


def is_video(file: UploadFile):
    if file.content_type and file.content_type.startswith("video/"):
        return True
    return file.filename.lower().endswith(VIDEO_EXTS)


def cleanup_files(*paths):
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"Cleanup failed for {path}: {e}")


# ---------------- MAIN ENDPOINT ----------------

@app.post("/predict")
async def predict(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):

    if not file.filename:
        return JSONResponse({"error": "No file uploaded"}, status_code=400)

    # ---------- IMAGE ----------
    if is_image(file):
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            results = model(image)
            annotated = results[0].plot()

            annotated_img = Image.fromarray(annotated[..., ::-1])

            buf = io.BytesIO()
            annotated_img.save(buf, format="JPEG")
            buf.seek(0)

            return StreamingResponse(buf, media_type="image/jpeg")

        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # ---------- VIDEO ----------
    elif is_video(file):
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

        try:
            temp_input.write(await file.read())
            temp_input.close()
            temp_raw.close()
            temp_output.close()

            cap = cv2.VideoCapture(temp_input.name)

            if not cap.isOpened():
                raise RuntimeError("Failed to open video")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 1:
                fps = 25

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if width == 0 or height == 0:
                raise RuntimeError("Invalid video dimensions")

            width = width // 2 * 2
            height = height // 2 * 2

            # More compatible codec
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            out = cv2.VideoWriter(temp_raw.name, fourcc, fps, (width, height))

            if not out.isOpened():
                raise RuntimeError("Failed to open VideoWriter")

            frame_count = 0
            annotated = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (width, height))

                if frame_count % 2 == 0:
                    results = model(frame)
                    annotated = results[0].plot()

                out.write(annotated if annotated is not None else frame)
                frame_count += 1

            cap.release()
            out.release()

            # Check raw video exists
            if not os.path.exists(temp_raw.name) or os.path.getsize(temp_raw.name) == 0:
                raise RuntimeError("Raw video file not created")

            # FFmpeg conversion
            ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"  # 👈 adjust if needed

            ffmpeg_cmd = [
                ffmpeg_path,
                "-y",
                "-i", temp_raw.name,
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                temp_output.name
            ]

            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True
            )

            print("FFmpeg stderr:", result.stderr)

            if result.returncode != 0:
                raise RuntimeError("FFmpeg conversion failed")

            background_tasks.add_task(
                cleanup_files,
                temp_input.name,
                temp_raw.name,
                temp_output.name
            )

            return FileResponse(
                temp_output.name,
                media_type="video/mp4",
                filename="result.mp4"
            )

        except Exception as e:
            print("ERROR:", str(e))
            cleanup_files(temp_input.name, temp_raw.name, temp_output.name)
            return JSONResponse({"error": str(e)}, status_code=500)

    # ---------- UNKNOWN ----------
    return JSONResponse({"error": "Unsupported file type"}, status_code=400)

from fastapi import WebSocket, WebSocketDisconnect
import base64
import numpy as np

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")

    try:
        while True:
            # Receive base64 image
            data = await websocket.receive_text()

            # Decode base64 → image
            image_data = base64.b64decode(data.split(",")[1])
            np_arr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Run YOLO
            results = model(frame)
            annotated = results[0].plot()

            # Encode back to JPEG
            _, buffer = cv2.imencode(".jpg", annotated)
            encoded = base64.b64encode(buffer).decode("utf-8")

            # Send back
            await websocket.send_text(f"data:image/jpeg;base64,{encoded}")

    except WebSocketDisconnect:
        print("WebSocket disconnected")