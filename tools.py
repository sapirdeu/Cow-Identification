from PIL import Image
import numpy as np
import os
import cv2
import keras


# Making images into array
def imgToArray(rootdir):
    data = []
    labels = []
    for subdir, dirs, files in os.walk(rootdir):
        for dir in dirs:
            currDir = os.path.join(subdir, dir)
            for file in os.listdir(currDir):
                if ".jpg" in file:
                    imag = cv2.imread(os.path.join(currDir, file))
                    img_from_ar = Image.fromarray(imag, 'RGB')
                    resized_image = img_from_ar.resize((50, 50))
                    data.append(np.array(resized_image))
                    labels.append(int(dir)-1)
    return data, labels


# Convert data and labels to numpy arrays
def convertToNumpy(data, labels):
    cows=np.array(data)
    labels=np.array(labels)
    return cows,labels


# Create data - X and Y
def createData(cows, labels):
    # Ensure that the input features are scaled between 0.0 and 1.0
    x_data = cows.astype('float32')/255
    # One hot encoding
    num_classes = len(np.unique(labels))
    y_data=keras.utils.to_categorical(labels,num_classes)
    return x_data, y_data
