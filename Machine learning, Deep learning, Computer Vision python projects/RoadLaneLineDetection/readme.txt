Road Lane Lines Detection Project

This project aims to identify road lines in which autonomous cars must run, 
which is a critical part of autonomous cars, as self-driving cars should not
cross its lane and should not go in opposite lane to avoid accidents.

1. Descriptions
The data used for detecting road lane lines are video footages including car-running scenes
with white and yellow solid road lane lines. For each frame of a footage, road lane lines detection
follows the steps below:

1) convert the orginal image to gray-scale image gray1 and hsv image hsv1.
 
2) use color thresholding method to extract yellow and white lane lines respectively from hsv1 and gray1,
   after that we can get binary mask images mask_yellow and mask_white of yellow and white lane lines.
   
3) perform or operation to combine mask_yellow and mask_white to a new mask image mask_yw, then convert it
   back to gray image mask_yw_image.

4) perform gaussian blurring processing to mask_yw_image, then use Canny operator to get edge detection image canny_edges.

5) use hough transform to detect mutiple left and right lane lines from our interested ROI (Region of Interest). Now multiple lines
   are detected, we will filter out abnormal detected lines by setting up thresholds for slopes, then use the least square
   method to make fittness for multiple detected left and right lane lines to get the final results.
   
2. Getting started

1) Dependencies
   OS: Windows 10
   libaries: Python 3.9.4 + OpenCV 4.5.4
2) Executing program
   execute the order python line_detection.py we can get road lane line detection results in footages at the folder road_lane_detection_output
   execute the order python GUI.py we can get visualization results of road lane line detection in GUI

3. Project Organization
├── README.md          				   <- The README for developers using this project.
├── image_processing_process           <- The folder including images to display whole image processing process for road lane lines detection.
├── road_lane_detection_output         <- The road lane lines detection results of all footages in the folder videos_input
├── videos_input                       <- The folder including testing video footages used for road lane lines detection.
├── detection_results_demo             <- A demo footage for showing running results of GUI.
├── line_detection.py                  <- Source code of building road lane lines detection method.
├── GUI.py             				   <- Source code used for building GUI to display results getting from running line_detection.py.



