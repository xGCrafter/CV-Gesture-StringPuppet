import mediapipe as mp
from mediapipe.tasks.python import vision
import cv2
import time
import pygame
from pygame import mixer

pygame.mixer.init()
pygame.mixer.music.load("CV-Guesture-Puppet/party.mp3")
is_playing = False
cap = cv2.VideoCapture(0)
BaseOptions = mp.tasks.BaseOptions
base_options = BaseOptions(model_asset_path="CV-Guesture-Puppet/hand_landmarker.task")
thumb_x = 0
thumb_y = 0
index_x = 0
index_y = 0
middle_x = 0
middle_y = 0
ring_x = 0
ring_y = 0
pinky_x = 0
pinky_y = 0
last_print_time = time.time() 

def callback(detection_result, image, timestamp):
    global thumb_x, thumb_y, index_x, index_y, pinch, middle_x, middle_y, ring_x, ring_y, pinky_x, pinky_y 
    if len(detection_result.hand_landmarks) > 0: 
        h, w, _ = image.numpy_view().shape 
        thumb = detection_result.hand_landmarks[0][4]
        thumb_x = int(thumb.x * w)
        thumb_y = int(thumb.y * h)
        index = detection_result.hand_landmarks[0][8]
        index_x = int(index.x * w) 
        index_y = int(index.y * h)
        middle = detection_result.hand_landmarks[0][12]
        middle_x = int(middle.x * w)
        middle_y = int(middle.y * h) 
        ring = detection_result.hand_landmarks[0][16]
        ring_x = int(ring.x * w)
        ring_y = int(ring.y * h)
        pinky = detection_result.hand_landmarks[0][20]
        pinky_x = int(pinky.x * w) 
        pinky_y = int(pinky.y * h)

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

    head    = (middle_x, middle_y + 300)  
    left_hand  = (index_x,  index_y  + 350)   
    right_hand  = (ring_x,   ring_y   + 350) 
    left_knee  = (thumb_x,  thumb_y  + 380) 
    right_knee  = (pinky_x,  pinky_y  + 380)  

    torso_top    = (head[0], head[1] + 30)
    torso_bottom = (head[0], head[1] + 120)
    left_foot = (left_knee[0], left_knee[1] + 70)  
    right_foot = (right_knee[0], right_knee[1] + 70)

    cv2.line(frame, (middle_x, middle_y), head,   (180,180,180), 1) 
    cv2.line(frame, (index_x,  index_y),  left_hand, (180,180,180), 1)  
    cv2.line(frame, (ring_x,   ring_y),   right_hand, (180,180,180), 1)  
    cv2.line(frame, (thumb_x,  thumb_y),  left_knee, (180,180,180), 1) 
    cv2.line(frame, (pinky_x,  pinky_y),  right_knee, (180,180,180), 1) 
    cv2.circle(frame, head, 25, (255,255,255), -1)
    cv2.line(frame, torso_top, torso_bottom, (255,255,255), 3)
    cv2.line(frame, torso_top, left_hand, (255,255,255), 3)
    cv2.line(frame, torso_top, right_hand, (255,255,255), 3) 
    cv2.line(frame, torso_bottom, left_knee, (255,255,255), 3)
    cv2.line(frame, torso_bottom, right_knee, (255,255,255), 3)  
    cv2.line(frame, left_knee, left_foot, (255,255,255), 3)  
    cv2.line(frame, right_knee, right_foot, (255,255,255), 3) 
    cv2.circle(frame, (thumb_x,  thumb_y),  8, (2,250,70), -1)
    cv2.circle(frame, (index_x,  index_y),  8, (2,250,70), -1)
    cv2.circle(frame, (middle_x, middle_y), 8, (2,250,70), -1)
    cv2.circle(frame, (ring_x,   ring_y),   8, (2,250,70), -1)
    cv2.circle(frame, (pinky_x,  pinky_y),  8, (2,250,70), -1)
    cv2.circle(frame, left_hand, 6, (255,255,255), -1)
    cv2.circle(frame, right_hand, 6, (255,255,255), -1)
    cv2.circle(frame, left_knee, 6, (255,255,255), -1)
    cv2.circle(frame, right_knee, 6, (255,255,255), -1)

    current_time = time.time()
    if current_time - last_print_time >= 1.0:
        print(f"[{time.strftime('%H:%M:%S')}]")
        print(f"  Thumb:  ({thumb_x}, {thumb_y})")
        print(f"  Index:  ({index_x}, {index_y})")
        print(f"  Middle: ({middle_x}, {middle_y})")
        print(f"  Ring:   ({ring_x}, {ring_y})")
        print(f"  Pinky:  ({pinky_x}, {pinky_y})")
        last_print_time = current_time

    # REACTION SYSTEM
    if left_hand[1] < torso_top[1] and right_hand[1] < torso_top[1]:
        cv2.putText(frame, "PARTYY!", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 10)
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(-1)
    else:
        pygame.mixer.music.stop()
    cv2.imshow("CV - GuestureStringPuppet", frame)
    if cv2.waitKey(1) == 27:
        break
