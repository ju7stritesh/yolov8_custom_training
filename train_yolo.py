from ultralytics import YOLO
# Load the model.
model = YOLO('yolov8x.pt')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Training.
if __name__ == '__main__':
   results = model.train(
      data='data.yaml',
      imgsz=640,
      epochs=20,
      batch=8,
      name='yolov8n_custom'
   )


