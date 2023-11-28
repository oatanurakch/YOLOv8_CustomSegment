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

# PC or Laptop camera 
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Variable for stored centroid
x_centroid_arr = np.zeros(10)
y_centroid_arr = np.zeros(10)

# Variable for count centroid
count = 0

# Variable for store average centroid
x_pos_avg = 0
y_pos_avg = 0

# Variable for stored old centroid
x_pos_old = 0
y_pos_old = 0

# Variable for stored new centroid
x_pos_new = 0
y_pos_new = 0

ESP32_SER = Serial('/dev/ttyUSB0', 115200, timeout = 1)
# ESP32_SER = Serial('COM3', 115200, timeout = 1)


inRange = False

count_frame = 0

while cap.isOpened():
    ret, frame = cap.read()

    # Check if frame is not empty
    if ret:
        # Inference
        results = model.track(source = frame, conf = 0.7, save = False)

        # Check if there is any object detected
        if not len(results[0]):
            print('No object detected')
            annotated_frame = frame

        else:
            # Plotting the result
            annotated_frame = results[0].plot()

            # Add Point to the frame
            for r in results:
                # Get mask
                mask = results[0].masks.xy
                # IDS of the object
                ids = results[0].boxes.id.numpy().astype(int)
                # IDs of class names
                ids_cls = results[0].boxes.cls.numpy().astype(int)
                for i in range(len(mask)):
                    # Point quantity
                    p_qty = len(mask[i])
                    # Store point coordinate
                    x_arr = []
                    y_arr = []
                    # print(f'ID: {ids[i]}')
                    # print(f'Class: {ids_cls[i]}')
                    for j in range(len(mask[i])):
                        # print(f'x: {mask[i][j][0]}, y: {mask[i][j][1]}')
                        # append x and y to array
                        x_arr.append(mask[i][j][0])
                        y_arr.append(mask[i][j][1])
                    # Calculate centroid
                    x_arr = np.array(x_arr)
                    y_arr = np.array(y_arr)
                    p_centroid = (int(np.sum(x_arr) / p_qty), int(np.sum(y_arr) / p_qty))
                    # Draw point and text
                    cv2.putText(annotated_frame, f'{p_centroid}', (p_centroid[0] - 5, p_centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(annotated_frame, (p_centroid[0], p_centroid[1]), 1, (255, 255, 255), 2)
                    # Calculate position of the point
                    if ids_cls[i] == 39:
                        # Check position is in range
                        if (p_centroid[0] >= 190 and p_centroid[0] <= 490) and (p_centroid[1] >= 220 and p_centroid[1] <= 400):
                            count += 1
                            # Append centroid to array for calculate average
                            x_centroid_arr = np.append(x_centroid_arr[1:], p_centroid[0])
                            y_centroid_arr = np.append(y_centroid_arr[1:], p_centroid[1])
                            if count == 10:    
                                x_centroid_avg = np.sum(x_centroid_arr) / 10
                                y_centroid_avg = np.sum(y_centroid_arr) / 10
                                # Calculate X position from pixel to meter
                                x_pos_avg = (x_centroid_avg * (-0.0021)) + 0.7223
                                # Calculate Y position from pixel to meter
                                y_pos_avg = (y_centroid_avg * (0.0018)) - 0.7223
                                # x and y pos new
                                x_pos_new = x_pos_avg
                                y_pos_new = y_pos_avg
                                # Reset count
                                count = 0
                            cv2.putText(annotated_frame, f'{x_pos_avg:.2f}, {y_pos_avg:.2f}', (p_centroid[0] - 5, p_centroid[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  
                            # Check if position is change
                            if (x_pos_new != x_pos_old and abs(x_pos_new - x_pos_old) < 0.02) or (y_pos_new != y_pos_old and abs(y_pos_new - y_pos_old) < 0.02):
                                # Send position to ESP32
                                ESP32_SER.write(f'mx{x_pos_avg:.2f},y{y_pos_avg:.2f}z-1\n'.encode())
                                print(f'mx{x_pos_avg:.2f},y{y_pos_avg:.2f}z-1\n')
                                # Update old position
                                x_pos_old = x_pos_new
                                y_pos_old = y_pos_new
                        else:
                            cv2.putText(annotated_frame, f'Out of range', (p_centroid[0] - 5, p_centroid[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('YOLOv8', annotated_frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()