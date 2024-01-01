import argparse
import subprocess

def train_yolov5(args):
    command = [
        'python', 'yolov5/train.py',
        '--img', str(args.img_size),
        '--batch', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--data', args.data_file,
        '--weights', args.weights
    ]

    # Execute the training command
    subprocess.run(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv5 model')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--data_file', type=str, default='coco128.yaml', help='Path to data file')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='Path to weights file')

    args = parser.parse_args()

    train_yolov5(args)
