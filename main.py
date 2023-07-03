import cv2
from tensorflow import keras
import numpy as np
from keras.utils import img_to_array
from keras.models import load_model
import time
# load model
model = load_model('model_MobileNetV2.h5')
# camera access
# cam = cv2.VideoCapture("http://192.168.137.56:8080/video")
cam = cv2.VideoCapture(0) # Webcam
#timer started
time1 = time.time()
#haar cascade classifer to detect face in the frame
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#data about known and unknown
known = 0
access = 'denied'
for _ in range(20):
    ret, test_image = cam.read()
    #test_image = cv2.flip(test_image, 0)
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    detected_face = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    for (x, y, w, h) in detected_face:
        print(x,y)
        cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 0, 255), thickness=4)
        #pre-processing frames
        #roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(gray_img, (224, 224))
        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)
        print(predictions)
        # find the label with max probability
        max_index = np.argmax(predictions[0])
        predicted_class = 'unknown'
        classes = ('shashank','subigya','sushil')
        if max_index > 0.9:
            predicted_class = classes[max_index]

        if predicted_class == 'unknown':
            cv2.putText(test_image, predicted_class, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), thickness=4)
            cv2.putText(test_image, 'known', (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,cv2.LINE_AA)
            known += 1
        count_var = 'Known Counter: '+str(known)
        cv2.putText(test_image,count_var,(int(x),int(y)-50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    #resized_img = cv2.resize(test_image, (1000, 700))
    cv2.imshow('Facial Recognition Demo for Home Security System', test_image)
    
    if known > 15:
        access = 'granted'
        break
        
    if cv2.waitKey(10) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
time2 = time.time()
time_taken = time2 - time1
print('Time taken: ',time_taken)
if access == 'granted':
    cv2.imshow('Access!', cv2.resize( cv2.imread('granted.png'),(960,540)))
    
else:
    cv2.imshow('Access!', cv2.resize(cv2.imread('denied.png'),(960,540)))

# if cv2.waitKey(100) == ord('q'):
#     cv2.destroyAllWindows()
cv2.waitKey(0)

