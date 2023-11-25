from ultralytics import YOLO as yolo
import cv2

# Initialize
model = yolo('yolov8s-seg.pt')

results = model(source = r'E:\OneDrive - Suranaree University of Technology\Work\AgTech\Detection\img1.jpg', conf = 0.7, save = True)

for r in results:
    mask = r.masks.xy # Mask is a list of list of arrays
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            print(mask[i][j])