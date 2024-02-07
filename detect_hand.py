import cv2 
import numpy as np
import tensorflow as tf

model = tf.keras.model.load_model("weights.bin")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    image = cv2.resize(frame,(224,224))
    test_img = np.array(image,dtype=np.float32)
    test_img = np.expand_dims(test_img,axis=0)
    test_img = test_img/255.0
    prediction = model.predict(test_img)
    print(prediction)
    cv2.imshow("Hand shape detection", frame)
    if cv2.waitKey(1) == 32:
        break
        print('Stoped')

cap.release()
cv2.destroyAllWindows()