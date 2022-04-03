# Made By Addison Chua (https://github.com/NotAddison)
# SideNote : Not using CUDA because MAC doesn't have dedicated GPUs (?) :: windows superiority 

# Modules (OpenCV, time (FPS))
import cv2 as cv 
from time import time

# --- ⚙ OpenCV bbox Settings ⚙ ---
threshold = 0.55        # Main threshold for obj detection [aka, sensitivity]
Left_threshold = 0.65   # Left_threshold should be higher than main, more accurate detection of num of people on the left
toMirror = True         # Mirrors the projected frames (Use True if you're using a webcam & Left and right are mirrored)

font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
thickness = 2
colour = (0,255,0)

# Load Dependency Files
config = r'Assets\Dependencies\coco-config.pbtxt'
frozen_model = r'Assets\Dependencies\frozen_inference_graph.pb'

# Read Pretrained Model
model = cv.dnn_DetectionModel(frozen_model, config)

# Model Setup
model.setInputSize(320, 320)
model.setInputScale(1.0/ 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Labels
lables = open('coco-labels.txt', 'r').read().rstrip('\n').split('\n')
print(f">> Loaded {len(lables)} classes...")


# // -- OpenCV Read Video (frames) --
# VideoCapture(0)       : 0 = Default Camera
# VideoCapture(1)       : 1 = External Camera
# VideoCapture(addr)    : addr = Path to Video File
video = cv.VideoCapture(0)

## Webcam Settings
video.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

## Checks if camera opened successfully
if not video.isOpened():
    video = cv.VideoCapture(0)
if not video.isOpened():
    raise IOError("Cannot Open Video")

# Main Function
looptime = time() # Time Bookmark
while True:
    count = 0
    left_count = 0
    right_count = 0
    ret,frame = video.read()

    if(toMirror):
        frame = cv.flip(frame, 1)

    roi_left = frame[0:1280, 0:640]
    classIndex, confidence, bbox = model.detect(frame, threshold)

    # print(classIndex)
    if(len(classIndex) != 0):
        for classIndex, confidence, bbox in zip(classIndex.flatten(), confidence.flatten(), bbox):
            if (classIndex <= 80):
                if(lables[classIndex-1] == 'person'):                                                           # Filter so it displays only People
                    count +=1
                    cv.rectangle(frame, bbox, (255,169,0), thickness)                                           # Draw Bounding Box
                    cv.putText(frame, lables[classIndex-1], (bbox[0], bbox[1]), font, font_scale, colour, 1)    # Draw Labels

    L_classIndex, L_confidence, L_bbox = model.detect(roi_left, Left_threshold)
    if(len(L_classIndex) != 0):
        for L_classIndex, L_confidence, L_bbox in zip(L_classIndex.flatten(), L_confidence.flatten(), L_bbox):
            if (L_classIndex <= 80):
                    left_count +=1

    # FPS Calculation & output
    print("No. of people: {count} | Left No:{left_count} | Right No.(Est): {right_count}  | FPS: {fps}".format(count= count, left_count = left_count, right_count = count-left_count ,fps=(1/(time() - looptime))))
    looptime = time()
    
    # Display OpenCV Video Result
    frame = cv.line(frame,(640,0),(640,1000),(255,255,255),7)
    cv.imshow('Human Detection', frame)
    # cv.imshow('ROI Left',roi_left)

    # Exit on 'ESC' Key
    if cv.waitKey(1) == 27: 
        break 
video.release()
cv.destroyAllWindows()