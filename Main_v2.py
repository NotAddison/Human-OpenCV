# Made By Addison Chua (https://github.com/NotAddison)
# SideNote : V2 uses an estimated center dot from the coordinates of the Bondary Boxes (BBox) to determine the number of people on each side (left & right)

# Modules (OpenCV, time (FPS))
import cv2 as cv 
from time import time

# --- ⚙ OpenCV bbox Settings ⚙ ---
threshold = 0.55        # Main threshold for obj detection [aka, sensitivity]
Left_threshold = 0.65   # Left_threshold should be higher than main, more accurate detection of num of people on the left
toMirror = True         # Mirrors the projected frames (Use True if you're using a webcam & Left and right are mirrored)
center_offset = 200     # Offset for center dot (Note To Self: Need to fix for better accuracy)

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

## Checks if camera opened successfully
if not video.isOpened():
    video = cv.VideoCapture(0)
if not video.isOpened():
    raise IOError("Cannot Open Video")

## Webcam Settings
video.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

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

                    # Bbox Tracking postiton (Using center point of Bbox)
                    # 0-> left top corner, 1-> left bottom corner, 2-> right bottom corner, 3-> right top corner

                    width = bbox[2] - bbox[1]   # Right Bottom - Left Bottom
                    width_center_coord = int((bbox[0]+ (width/2)) + center_offset)

                    # print(f"Width : {width} : {bbox[0]} | ", f"Center: ({width_center_coord})")
                    
                    frame = cv.circle(frame, (width_center_coord, bbox[1]), 3, (255,255,255), thickness)

                    if (width_center_coord > 640):
                        right_count += 1
                    else:
                        left_count += 1


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