from ultralytics import YOLO

modelp = YOLO(
    r"C:\Users\V4lde\Downloads\Perception-for-Autonomous-Systems-Final-Project\runs\detect\train3\weights\best.pt"
)

prediction_results = modelp.predict(
    "https://ultralytics.com/assets/kitti-inference-im0.png",
    save=True,
)
