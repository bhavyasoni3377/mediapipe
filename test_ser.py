import unittest 
from ser import classifyPose 
import mediapipe as mp



import cv2
sample_img = cv2.imread('f3.png')
class Test_syp(unittest.TestCase):
    def test_Pose(self):
        actual = "Higer Fowler"
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
                             
        if landmarks:
             output_image1,label = classifyPose(landmarks, output_image)
        expected = label
        self.assertEqual(actual, expected)

if __name__ == '__main__':  
    unittest.main()  