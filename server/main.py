# Import necessary libraries
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import io
import os

# Initialize FastAPI
app = FastAPI()


app.add_middleware( # Allow CORS for all origins (for development purposes)
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
base_dir = Path(__file__).resolve().parent.parent
model = YOLO(str(base_dir / 'server' / 'model' / 'traffic_sign_v2' / 'weights' / 'best.pt'))
# model.to('cuda')  # Try this later, GPU stuff

# Image prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)) #.convert("RGB")

    # Run inference
    results = model(image, imgsz=640) # , conf=0.25)  # Adjust conf threshold as needed

    # Get annotated image
    annotated = results[0].plot()

    # Convert to PIL
    annotated_image = Image.fromarray(annotated[..., ::-1])

    # Save to buffer
    buf = io.BytesIO()
    annotated_image.save(buf, format='JPEG')
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")

# Video prediction endpoint
@app.post("/predict-video/")
async def predict_video(file: UploadFile = File(...)):
    # Save uploaded video temporarily
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(await file.read())
    temp_input.close()

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_output.close()

    cap = cv2.VideoCapture(temp_input.name)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame, imgsz=640) # , conf=0.25)

        # Get annotated frame
        annotated = results[0].plot()

        out.write(annotated)
    
    cap.release()
    out.release()

    return FileResponse(temp_output.name, media_type="video/mp4", filename="result.mp4")
