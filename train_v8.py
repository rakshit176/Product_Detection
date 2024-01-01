from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m.pt')

# Specify training data and start training
results = model.train(
    data='coco128.yaml',
    epochs=3,
    batch_size=16,
    img_size=640,
    resume=False,
    weights='yolov8n.pt',
    device='0',
    evolve=False,
    project='my_project',
    name='exp',
    exist_ok=False
)