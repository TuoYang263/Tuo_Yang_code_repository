In this sign language recognition project, we create a sign detector, which detects numbers from 0 to 9 that
can very easily be extended to cover a vast multitude of other signs and hand gestures including the alphabets

The prerequistes software and libraries for the sign project are:
-Python(3.7.4)
-IDE(Jupyter)
-Numpy (version 1.16.5)
-cv2 (openCV) (version 3.4.2) 
-Keras(version 2.3.1)
-Tensorflow(version 2.0.0)

Steps to develop sign language recognition project
This project can be divided into three parts:

1.Creating the dataset
2.Training a CNN on the captured dataset
3.Predicting the data

The files of this project in the current folder include:
gesture - the folder of the datasets, which includes two folders: train and test. Each folder includes the binary images collected 
          from digit 0 to 9. Samples in the folder train are used for the training dataset (7020 images in total),  
		  while samples in the folder test are used for the validation dataset (410 images in total).
		  
recognition_results - the trained CNN model's recognition results for digits hand gestures made in the living cam 

create_gesture.py - the program file used for collecting samples of digits hand gestures in the living cam

digit_seven_samples_collection_footage.mp4 - demo video showing how to collect training samples for the digit seven

model_007_Adam.h5 - the best performance model trained with the Adam optimizer, with the validation accuracy of 94.63%

model-020_SGD.h5 - the best performance model trained with the SGD optimizer, with the validation accuracy of 88.29%

readme.txt - helping doc

reference_gesture.JPG - standard digit hand gesture used for reference

test.py - the program file used for testing the model's performance

sign_language_recognition.ipynb - the program file used for training the CNN model

