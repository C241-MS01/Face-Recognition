import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture("C:/Users/Arya Revansyah/Downloads/video_1.mp4")

# drawing on faces
mpDraw = mp.solutions.drawing_utils

# create face mesh
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()

# drawing specification
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# first info
pTime = 0
width = 800
height = 450

while True:
    # read
    success, img = cap.read()
    
    # resize the video
    img = cv2.resize(img, (width, height))

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, 
                                  faceLms, 
                                  mpFaceMesh.FACEMESH_CONTOURS,
                                  drawSpec, 
                                  drawSpec)
            
            #for lm in faceLms.landmark:
                #print(lm)
     
    # frame rate
    #cTime = time.time()
    #fps = 1/cTime-pTime
    #pTime = cTime
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cv2.putText(img, f'FPS:{int(fps)}', (20,70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
    '''
    (20, 70)    => Font place or location
    3           => Scale
    (0, 255, 0) => Color
    3           => Thickness
    '''

    cv2.imshow("image", img)
    cv2.waitKey(1)