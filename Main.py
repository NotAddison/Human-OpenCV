# Modules (OpenCV, time (FPS))
# SideNote : Not using CUDA because MAC doesn't have dedicated GPUs (?) :: windows superiority 
import cv2 as cv 
from time import time


# // OpenCV bbox Settings ⚙
threshold = 0.55
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
video = cv.VideoCapture(r'Assets\Test\Footage of People Walking _ Free Stock Footage (1080p).mp4')

## Checks if camera opened successfully
if not video.isOpened():
    video = cv.VideoCapture(0)
if not video.isOpened():
    raise IOError("Cannot Open Video")

# Main Function
looptime = time() # Time Bookmark
while True:
    count = 0
    ret,frame = video.read()
    classIndex, confidence, bbox = model.detect(frame, threshold)

    # print(classIndex)
    if(len(classIndex) != 0):
        for classIndex, confidence, bbox in zip(classIndex.flatten(), confidence.flatten(), bbox):
            if (classIndex <= 80):
                if(lables[classIndex-1] == 'person'):   # Filter so it displays only People
                    count +=1 
                    cv.rectangle(frame, bbox, (255,169,0), thickness)
                    cv.putText(frame, lables[classIndex-1], (bbox[0], bbox[1]), font, font_scale, colour, 1)

    # FPS Calculation & output
    print("No of people: {count} | FPS: {fps}".format(count=count, fps=(1/(time() - looptime))))
    looptime = time()
    
    # Display OpenCV Video Result
    cv.imshow('Human Detection', frame)

    # Exit on 'ESC' Key
    if cv.waitKey(1) == 27: 
        break 
video.release()
cv.destroyAllWindows()