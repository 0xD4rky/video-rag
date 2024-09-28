import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using device : {device}")

model = YOLO('yolov8n.pt')
model.to(device)

video_path = r'f1-det/f1.mp4'
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = r'f1_inf.mp4'
if os.path.exists(output_path) == False:
    os.makedirs(output_path, exist_ok = True)

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width,frame_height))

while cap.isOpened():

    success, frame = cap.read()
    if success:
        results = model(frame, device=device)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
