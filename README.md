
# Surgical Pose Estimation

This repository include all showcase projects related to advance computer vision and image processing Pose Estimation techniques in Python.
Using Mediapipe and Opencv
The landmark model in MediaPipe Pose predicts the location of 33 pose landmarks other then that it uses two more additional point which are making the spine by joining center of both hip and shoulders
in some cases we check the Highest landmark in body to determine pose  
## Acknowledgements

 - [Refrence for  Pose Detaction ](https://google.github.io/mediapipe/)
 - [Pose Refrances](https://www.steris.com/healthcare/knowledge-center/surgical-equipment/complete-guide-to-patient-positioning)
 

## Pose  Reference

| Pose             | Logic use to classify                                                                |
| ----------------- | ------------------------------------------------------------------ |
| High/Full Fowler’s: | head of the bed 90 degrees |
| Standard-Fowler’s:  | head of the bed 45-60 degrees |
| Semi-Fowler’s:  | head of the bed 30-45 degrees |
| Low-Fowler’s: | head of the bed raised 15-30 degrees |
|Lithotomy :|lying on back with legs flexed 90 degrees at  hips knees will be bent at 70 to 90 degrees|
|Trendelenburg:|supine on the table with their head declined below their feet at an angle of roughly 16°|
|Supine:|lying horizontally with the face and torso facing up|
| Prone:|lie on their abdomen in a monitored setting|
|Reverse Trendelenburg :|a position in which patients' hip and knee are not flexed but the head and chest are elevated at 30° than the abdomen and legs|


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`sudo yum install opencv-develmediapipe`

`pip install opencv-python`


## Screenshots
High Fowler


![Pose classification Screenshorts high Fowler ](https://github.com/bhavyasoni3377/mediapipe/blob/main/Images/high%20Fowler.jpg)


Trendelenburg

![Pose classification Screenshorts high Fowler ](https://github.com/bhavyasoni3377/mediapipe/blob/main/Images/savedImage.jpg)


Reverse Trendelenburg

![Pose classification Screenshorts high Fowler ](https://github.com/bhavyasoni3377/mediapipe/blob/main/Images/rtb.jpg)
## Running Tests

To run tests, run the following command

import unittest

from ser import *

import cv2

import numpy as np
import mediapipe as mp

import matplotlib.pyplot as plt

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7, model_complexity=2)

 


class Test_syp(unittest.TestCase):
    filename = "f3.png"
    image=cv2.imread(filename)
    mp_pose = mp.solutions.pose


    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7, model_complexity=2)

    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        #         
        # Iterate over the detected landmarks.
            for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
                landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
                             
    
    # Check if the original input image and the resultant image are specified to be displayed.
         
        
        # Also Plot the Pose landmarks in 3D.
        
    # Otherwise
        
        
        # Return the output image and the found landmarks.
        
                output_image, landmarks
            if landmarks:
             output_image1,label = ser.classifyPose(landmarks, output_image, display=True)
        
    actual = "Higer Fowler"
     
    expected = label
    image.assertEqual(actual, expected)

if __name__ == '__main__':  
    unittest.main()  


#Program Code

from cProfile import label
import math
from socket import TIPC_SUB_SERVICE
import cv2
import numpy as np
from time import time
import mediapipe as mp
#import pandas as pd
import matplotlib.pyplot as plt

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 

def detectPose(image, pose, display=True):
    
    
    
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''

    # Create a copy of the input image.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    results = pose.process(imageRGB)

    # Retrieve the height and width of the input image.
    height, width, _ = image.shape

    # Initialize a list to store the detected landmarks.
    landmarks = []

    # Check if any landmarks are detected.
    if results.pose_landmarks:

        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)

        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:

            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
            for i in range(33):

        # Display the found normalized landmarks.
                print(f'{mp_pose.PoseLandmark(i).name}:\n{results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]}')                 

    # Check if the original input image and the resultant image are specified to be displayed.
    if display:

        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');

        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    # Otherwise
    else:

        # Return the output image and the found landmarks.
        return output_image, landmarks
def calx(PoseLandmark1,PoseLandmark2) :

    x1, y1, _ = PoseLandmark1
    x2, y2, _ = PoseLandmark2
    cal=x1-x2
    return cal        

def caly(PoseLandmark1,PoseLandmark2) :

    x1, y1, _ = PoseLandmark1
    x2, y2, _ = PoseLandmark2
    cal=y1-y2
    return cal            

def cal_hip_angle(PoseLandmark1,PoseLandmark2):

    x1, y1, _ = PoseLandmark1
    x2, y2, _ = PoseLandmark2
    x_h=x1+x2/2
    y_h=y1+y2/2
    PoseLandmark_hip= x_h,y_h,_

    return PoseLandmark_hip
def tb1(PoseLandmark1,PoseLandmark2):

    x1, y1, _ = PoseLandmark1
    x2, y2, _ = PoseLandmark2
      
    if y2<y1:
            pp=1
    if y1<y2:
            pp=0    
    return pp
def tb(PoseLandmark1,PoseLandmark2):

    x1, y1, _ = PoseLandmark1
    x2, y2, _ = PoseLandmark2
     
    if x2<x1:
            pp=0
    if x1<x2:
            pp=1    
    return pp




def calculateAngle(landmark1, landmark2, landmark3):
    
    
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.
    '''

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360

    # Return the calculated angle.
    return angle        

def classifyPose(landmarks, output_image, display=False):
    
    '''
    This function classifies poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    # Get the angle between the right  shoulder, hip and knee points. 
    right_HIP_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
# Get the angle between the left  shoulder, hip and knee points. 
    left_HIP_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
    nose_Rankel_cal=calx(landmarks[mp_pose.PoseLandmark.NOSE.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    nose_Lankel_cal=calx(landmarks[mp_pose.PoseLandmark.NOSE.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    lib_legs_ag=cal_hip_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])

    r_l_h= calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     lib_legs_ag,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])                                                           
    #----------------------------------------------------------------------------------------------------------------
    
    if tb(landmarks[mp_pose.PoseLandmark.NOSE.value],landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]) == 1 or tb(landmarks[mp_pose.PoseLandmark.NOSE.value],landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])==1:  
                label = ' Trendelenburg '
   
    #----------------------------------------------------------------------------------------------------------------# Check if it is the Fowler pose.
    if tb(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]) == 0 or tb(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])==0:  
                label = 'Reverse Trendelenburg '
    if right_HIP_angle >=60  and right_HIP_angle <= 100 and right_knee_angle >=240 and right_knee_angle <=280 and r_l_h >= 30 and r_l_h < 44 or left_HIP_angle > 61  and left_HIP_angle <=100 and left_knee_angle >=240 and left_knee_angle <=280 and r_l_h >= 30 and r_l_h < 44:

                # Specify the label of the pose that is Fowler pose.
                label = 'Lithotomy'
    else :   
          
           if right_HIP_angle >89  and right_HIP_angle < 120 or left_HIP_angle > 89  and left_HIP_angle <120 and tb1(landmarks[mp_pose.PoseLandmark.NOSE.value],landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]) == 0 or tb1(landmarks[mp_pose.PoseLandmark.NOSE.value],landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]) == 0 :

                # Specify the label of the pose that is Fowler pose.
            label = 'Higer Fowler'
                    
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the Fowler pose.
            if right_HIP_angle >=120  and right_HIP_angle <=145 or left_HIP_angle >=120  and left_HIP_angle <=145 :

                # Specify the label of the pose that is Fowler pose.
             label = ' Standard Fowler'
        
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the  Semi_Fowler pose.
    #----------------------------------------------------------------------------------------------------------------
    
       # Check if both legs are straight
            if right_HIP_angle >=135 and right_HIP_angle <= 150 or left_HIP_angle>=135 and left_HIP_angle <=150 : 

                # Specify the label of the pose that is Semi_Fowler pose.
               label = 'Semi_Fowler'
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the Fowler pose.
            if right_HIP_angle >=150  and right_HIP_angle <= 160 or left_HIP_angle >=150  and left_HIP_angle <=160 :

                # Specify the label of the pose that is Fowler pose.
                label = 'Low-Fowler'

            if nose_Rankel_cal>=0 or nose_Lankel_cal >=0 and right_HIP_angle == 180 or left_HIP_angle == 180:   
                        label = 'Supine'
            
    # Check if it is the Fowler pose.

    #----------------------------------------------------------------------------------------------------------------
    # Check if the pose is classified successfully
            if label != 'Unknown Pose':

        # Update the color (to green) with which the label will be written on the image.
                color = (0, 255, 0)  

    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    # Check if the resultant image is specified to be displayed.
    if display:

        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');

    else:

        # Return the output image and the classified label.
        return output_image, label

## Authors

- [Bhavya](https://github.com/bhavyasoni3377)

