from ultralytics import YOLO as yolo
import cv2

# Initialize
model = yolo('yolov8s-seg.pt')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        # Inference
        results = model(source = frame, conf = 0.7, save = False, classes = 0)

        # Check if there is any object detected
        if results[0].pred is None:
            print('No object detected')
            continue

        # Plotting the result
        annotated_frame = results[0].plot()

        # Add Point to the frame
        for r in results:
            mask = results[0].masks.xy
            for i in range(len(mask)):
                for j in range(len(mask[i])):
                    # print(f'x: {mask[i][j][0]}, y: {mask[i][j][1]}')
                    cv2.circle(annotated_frame, (int(mask[i][j][0]), int(mask[i][j][1])), 1, (0, 255, 0), 2)
        
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