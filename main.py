import mediapipe as mp
from mediapipe.tasks.python import vision
import cv2
import time
import pygame
from pygame import mixer

pygame.mixer.init()
pygame.mixer.music.load("party.mp3")
is_playing = False
cap = cv2.VideoCapture(0)
BaseOptions = mp.tasks.BaseOptions
base_options = BaseOptions(model_asset_path="hand_landmarker.task")
thumbx = 0
thumby = 0
indexx = 0
indexy = 0
middlex = 0
middley = 0
ringx = 0
ringy = 0
pinkyx = 0
pinkyy = 0
last_print_time = time.time() 

def callback(detection_result, image, timestamp):
    global thumbx, thumby, indexx, indexy, pinch, middlex, middley, ringx, ringy, pinkyx, pinkyy 
    if len(detection_result.hand_landmarks) > 0: 
        h, w, _ = image.numpy_view().shape 
        thumb = detection_result.hand_landmarks[0][4]
        thumbx = int(thumb.x * w)
        thumby = int(thumb.y * h)
        index = detection_result.hand_landmarks[0][8]
        indexx = int(index.x * w) 
        indexy = int(index.y * h)
        middle = detection_result.hand_landmarks[0][12]
        middlex = int(middle.x * w)
        middley = int(middle.y * h) 
        ring = detection_result.hand_landmarks[0][16]
        ringx = int(ring.x * w)
        ringy = int(ring.y * h)
        pinky = detection_result.hand_landmarks[0][20]
        pinkyx = int(pinky.x * w) 
        pinkyy = int(pinky.y * h)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=callback
)

detector = vision.HandLandmarker.create_from_options(options)

while True:
    ret, frame = cap.read() 
    frame = cv2.flip(frame, 2) 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)  
    timestamp = int(time.time() * 1000)  
    detector.detect_async(mp_image, timestamp)

    head    = (middlex, middley + 300)  
    leftHand  = (indexx,  indexy  + 350)   
    rightHand  = (ringx,   ringy   + 350) 
    leftKnee  = (thumbx,  thumby  + 380) 
    rightKnee  = (pinkyx,  pinkyy  + 380)  

    torsoTop    = (head[0], head[1] + 30)
    torsoBottom = (head[0], head[1] + 120)
    leftFoot = (leftKnee[0], leftKnee[1] + 70)  
    rightFoot = (rightKnee[0], rightKnee[1] + 70)

    cv2.line(frame, (middlex, middley), head,   (180,180,180), 1) 
    cv2.line(frame, (indexx,  indexy),  leftHand, (180,180,180), 1)  
    cv2.line(frame, (ringx,   ringy),   rightHand, (180,180,180), 1)  
    cv2.line(frame, (thumbx,  thumby),  leftKnee, (180,180,180), 1) 
    cv2.line(frame, (pinkyx,  pinkyy),  rightKnee, (180,180,180), 1) 
    cv2.circle(frame, head, 25, (255,255,255), -1)
    cv2.line(frame, torsoTop, torsoBottom, (255,255,255), 3)
    cv2.line(frame, torsoTop, leftHand, (255,255,255), 3)
    cv2.line(frame, torsoTop, rightHand, (255,255,255), 3) 
    cv2.line(frame, torsoBottom, leftKnee, (255,255,255), 3)
    cv2.line(frame, torsoBottom, rightKnee, (255,255,255), 3)
    cv2.line(frame, leftKnee, leftFoot, (255,255,255), 3)    
    cv2.circle(frame, (thumbx,  thumby),  8, (2,250,70), -1)
    cv2.circle(frame, (indexx,  indexy),  8, (2,250,70), -1)
    cv2.circle(frame, (middlex, middley), 8, (2,250,70), -1)
    cv2.circle(frame, (ringx,   ringy),   8, (2,250,70), -1)
    cv2.circle(frame, (pinkyx,  pinkyy),  8, (2,250,70), -1)
    cv2.circle(frame, leftHand, 6, (255,255,255), -1)
    cv2.circle(frame, rightHand, 6, (255,255,255), -1)
    cv2.circle(frame, leftKnee, 6, (255,255,255), -1)
    cv2.circle(frame, rightKnee, 6, (255,255,255), -1)

    current_time = time.time()
    if current_time - last_print_time >= 1.0:
        print(f"[{time.strftime('%H:%M:%S')}]")
        print(f"  Thumb:  ({thumbx}, {thumby})")
        print(f"  Index:  ({indexx}, {indexy})")
        print(f"  Middle: ({middlex}, {middley})")
        print(f"  Ring:   ({ringx}, {ringy})")
        print(f"  Pinky:  ({pinkyx}, {pinkyy})")
        last_print_time = current_time

    # REACTION SYSTEM
    if leftHand[1] < torsoTop[1] and rightHand[1] < torsoTop[1]:
        cv2.putText(frame, "PARTYY!", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 10)
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(-1)
    else:
        pygame.mixer.music.stop()
    cv2.imshow("CV - GuestureStringPuppet", frame)
    if cv2.waitKey(1) == 27:
        break
