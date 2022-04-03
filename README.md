# Human-OpenCV
<p align = "center">
  <img src = "Assets\Test\example.jpg" size = 100>
</p>

**OpenCV** Human Detection Based on **COCO Dataset**. <br>
The project counts the number of humans of each side of the frame (Left & Right), which could be used for input emulation based on positions of the humans ; similar to a voting system of which input should be emulated.


## Features
### Existing Features
- **Object Detection** - Detect Objects based on COCO Dataset, Not only for humans.
- **Object Region Counter** - Calculates and Estimates the number of obj in a part of the frame. (aka, Region of interest (ROI))

### Features Left to Implement
- Implement **CUDA** - Harness GPU powers to compute (Smoother, Higher FPS) [Not Supported On All Systems]


## Dependency Installations
1. ### **Download Project**
   >- Download the project
   >- Drag all files into a folder


2. ### **Install Virtual Environment** (venv) [**Optional**]
   >- Open CMD
   >- Type into CMD: >> **pip install virtualenv**


3. ### **Setup Virtual Environment** (venv)
   >- Open CMD
   >- Navigate to folder where files are extracted
   >- Type into CMD: >> **python -m venv (name)**
   >- Replace (name) with your desired Virtual Environment Name
   >- **Activation**: Type into CMD: >> **.\(name)\Scripts\activate**


4. ### **Install OpenCV**
   >- Ensure you have activated your Virtual Environment (prev step) [optional]
   >- Type into Terminal/CMD: >> **pip install opencv-python**
   >- Done! 🏁


