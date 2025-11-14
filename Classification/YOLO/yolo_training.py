from ultralytics import YOLO
import torch

def main():
    print("cuda available:", torch.cuda.is_available())
    print("num devices:", torch.cuda.device_count())

    model = YOLO("yolo11n.pt")  # or yolo11s.pt, etc.

    results = model.train(
        data="kitti.yaml",
        epochs=10,
        imgsz=640,
        classes=[0, 3, 5],
        device=0,
    )

if __name__ == "__main__":
    main()