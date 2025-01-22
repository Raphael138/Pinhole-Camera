# Pinhole Tracker
This project aims to model the following:
1. A simple pinhole camera and depth sensor. In other words, given a 3D point cloud, the goal is to project the points onto a 2D plane and visualize the distance from the imaginary camera to the each point. I also try to simulate different types of noise such as random gaussian or blur.
2. A tracking device. We remain in the same 3D point cloud and create a moving target, I then try to model a tracking device, e.g. drone, which follows the target using the depth sensor model. I also explore methods to mitigate the impact of the noise or blur present in the depth sensor.

This code was part of my final project for *ECE 4240: Robot Perception* class at Cornell.

Much of the cell outputs in the notebook are sliders and use interactive widgets. Thus the web browser doesn't display properly them. If you are curious about the sliders, I suggest to pull the repository and follow the instructions. Otherwise, I also include a GIF of the final product:

<p align="center"><b>Tracking Device using a Pinhole Camera with Gaussian Noise and Blur</b></p>
![](https://github.com/Raphael138/Pinhole-Tracker/blob/main/tracking_blurry.gif)

# How to Use
1. Clone the repository:
```bash
git clone https://github.com/Raphael138/Pinhole-Tracker.git
cd Pinhole-Tracker
```
2. Install the required dependencies (e.g., Jupyter Notebook, matplotlib, etc.).
3. Run the notebook and explore the interactive widgets.