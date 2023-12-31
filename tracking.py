from ultralytics import YOLO as yolo
import cv2
import numpy as np
from serial import Serial

# Initialize
model = yolo('yolov8s-seg.pt')

# video_path = r"E:\Work\TrimTestVideo.avi"
# cap = cv2.VideoCapture(video_path)

# Jetson Xavier NX with Realsense D435i
cap = cv2.VideoCapture(4)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(source = frame, conf = 0.6, save = False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()