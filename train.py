from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("yolov9s-obb.yaml").load("yolov9s.pt")  # build from YAML and transfer weights
    # Train the model
    results = model.train(
        data='C:/Users/dians/Desktop/ultralytics-main/ultralytics/cfg/mycfg/obb.yaml', epochs=120,
        batch=1, imgsz=640,workers=2, nbs=2)

