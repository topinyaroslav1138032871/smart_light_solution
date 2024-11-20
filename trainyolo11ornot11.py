from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
if __name__ == '__main__':
    results = model.train(data='dataset.yaml', imgsz=640, epochs=10, batch=4, workers=8)