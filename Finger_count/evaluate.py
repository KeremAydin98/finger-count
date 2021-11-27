import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2


model = load_model("C:/Users/KEREM/Desktop/Python Projects/Projects/Finger_count/my_model.h5")

cap = cv2.VideoCapture(0)
cv2.namedWindow('Test')
cv2.namedWindow('Smaller Test')

while True:

    ret, frame = cap.read()

    cv2.rectangle(frame,(50,100),(350,350),(0,0,255),5)

    smaller_frame = frame[50:350,100:350]

    smaller_frame = cv2.cvtColor(smaller_frame,cv2.COLOR_RGB2GRAY)

    img = image.img_to_array(smaller_frame)
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)
    prediction = np.argmax(prediction, axis=1)
    prediction = prediction + 1

    cv2.putText(smaller_frame,f"{prediction}",(200,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))

    cv2.imshow('Test', frame)
    cv2.imshow('Smaller Test', smaller_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()








