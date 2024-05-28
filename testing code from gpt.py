import cv2
import time
import mediapipe as mp
from scipy.spatial import distance as dist

# Threshold and constants
EAR_THRESHOLD = 0.08
CONSECUTIVE_FRAMES = 48  # Number of frames the eye must be below the threshold to be considered as closed
EAR_DURATION_THRESHOLD = 3  # Time duration (in seconds) to consider as sleeping

# Initialize variables
ear_below_threshold_frames = 0
start_time = None

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Eye aspect ratio calculation function
def calculate_ear(eye_landmarks):
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Capture video from webcam
#cap = cv2.VideoCapture("C:/Users/Arya Revansyah/Downloads/video_1.mp4")
cap = cv2.VideoCapture("C:/Users/Arya Revansyah/Downloads/video_2.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame color to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and find face landmarks
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Get coordinates of left and right eye landmarks
        left_eye_landmarks = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in [33, 160, 158, 133, 153, 144]]
        right_eye_landmarks = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in [362, 385, 387, 263, 373, 380]]
        
        # Calculate EAR for both eyes
        left_ear = calculate_ear(left_eye_landmarks)
        right_ear = calculate_ear(right_eye_landmarks)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Check if EAR is below threshold
        if avg_ear < EAR_THRESHOLD:
            if ear_below_threshold_frames == 0:
                start_time = time.time()
            ear_below_threshold_frames += 1
        else:
            ear_below_threshold_frames = 0
            start_time = None
        
        # Determine if the person is sleeping
        if ear_below_threshold_frames > CONSECUTIVE_FRAMES:
            elapsed_time = time.time() - start_time
            if elapsed_time >= EAR_DURATION_THRESHOLD:
                cv2.putText(frame, "Sleeping", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Send a message or alert here
                print("Person is sleeping")
                # Reset to avoid repeated messages
                ear_below_threshold_frames = 0
                start_time = None
        
        # Draw landmarks
        for (x, y) in left_eye_landmarks:
            color = (0, 255, 0) if avg_ear >= EAR_THRESHOLD else (0, 0, 255)
            cv2.circle(frame, (int(x), int(y)), 1, color, -1)
        for (x, y) in right_eye_landmarks:
            color = (0, 255, 0) if avg_ear >= EAR_THRESHOLD else (0, 0, 255)
            cv2.circle(frame, (int(x), int(y)), 1, color, -1)
    
    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
