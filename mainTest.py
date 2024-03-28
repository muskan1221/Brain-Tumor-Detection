import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
model=load_model('BrainTumor10Epochs.h5')
image=cv2.imread('C:\\Users\\Muskan Sharma\\Downloads\\Braintumordetection\\pred\\pred0.jpg')

img=Image.fromarray(image)
img=img.resize((64,64))

img=np.array(img)
input_img=np.expand_dims(img,axis=0)
prediction=model.predict(input_img)
predicted_class=1 if prediction[0][0]>0.5 else 0 
print(predicted_class)
if(predicted_class==1):
    print("Brain Tumor Positive")
else:
    print("Brain Tumor Negative")
    
