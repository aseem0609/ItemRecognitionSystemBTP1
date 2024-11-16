import torch
from ultralytics import YOLO
import os


model = YOLO('yolov5n.pt')


data_path = r"E:\Trial15BTP\data.yaml"

#Training parameters
img_size = 640
batch_size = 32
epochs = 20

for epoch in range(epochs):
    results = model.train(data=data_path, epochs=1, batch=batch_size, imgsz=img_size, device='cpu')


    checkpoint_path = f"./checkpoint_epoch_{epoch + 1}.pt"
    model.save(checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

print("Training complete.")
