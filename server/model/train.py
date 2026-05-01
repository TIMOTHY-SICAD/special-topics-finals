# Import necessary libraries
from pathlib import Path
from ultralytics import YOLO

# File paths
base_dir = Path(__file__).resolve().parent.parent.parent
file_path = base_dir / 'server' / 'model' / 'data' / 'Traffic and Road Signs.yolov8' / 'data.yaml'

# Train model from pre-trained weights
model = YOLO('yolov8n.pt')
model.train(data=str(file_path), epochs=30, imgsz=640, name='traffic_sign_v2')

# # Show results
# result = model(str(base_dir / 'server' / 'model' / 'data' / 'sign.jpg'))
# result[0].show()