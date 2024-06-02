import torch
import cv2
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Initialize video capture (use 0 for webcam or provide path to a video file)
cap = cv2.VideoCapture('path/ke/folder/video/apa?')  #pathnya belom

# Initialize variables to track detection durations
detection_start_times = {}
detected_frames_count = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)
    detections = results.pandas().xyxy[0]  # get pandas dataframe

    # Check if there are any detections
    current_time = time.time()
    for index, row in detections.iterrows():
        class_id = int(row['class'])
        if class_id not in detection_start_times:
            detection_start_times[class_id] = current_time
            detected_frames_count[class_id] = 0
        detected_frames_count[class_id] += 1

        # Draw bounding boxes and labels on the frame
        xmin, ymin, xmax, ymax, conf, cls, name = row
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(frame, name, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the frame
    cv2.imshow('YOLOv5 Detection', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Calculate and print total detection durations for each class
for class_id, start_time in detection_start_times.items():
    total_detection_time = time.time() - start_time
    print(f"Class {class_id} detection duration: {total_detection_time:.2f} seconds")
    print(f"Class {class_id} detected frames: {detected_frames_count[class_id]}")
    
