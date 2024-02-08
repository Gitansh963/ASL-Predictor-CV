# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:32:22 2024

@author: gitan
"""


import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# model = tf.keras.models.load_model('other_model.h5')

loaded_model = tf.keras.models.load_model('other_model_3')

loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'blank','c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

cv2.namedWindow('Video')

cap = cv2.VideoCapture(0)

lower = np.array([0, 20, 70], dtype = "uint8")
upper = np.array([255, 255, 255], dtype = "uint8")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    converted = frame

    skinMask = cv2.inRange(converted, lower, upper)

    skinMask = cv2.erode(skinMask, None, iterations = 2)
    skinMask = cv2.dilate(skinMask, None, iterations = 2)

    skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    contours, _ = cv2.findContours(skinMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key = cv2.contourArea)
        
        (x, y, w, h) = cv2.boundingRect(c)
        
        
        x = max(0, x - (200 - w) // 2)
        y = max(0, y - (200 - h) // 2)
        w = h = 200
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        hand = np.zeros((h, w, 3), dtype = "uint8")
        hand = cv2.resize(frame[y:y + h, x:x + w], (200, 200))


        reshaped = np.reshape(hand, (1, 200, 200, 3))

        plt.imshow(reshaped[0])
        plt.axis('off')
        plt.show()
        
        # Predict the label using the model
        prediction = loaded_model.predict(reshaped)
        label = labels[np.argmax(prediction)]

        # Display the label on the frame
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Video', frame)

    # Wait for 1 millisecond
    key = cv2.waitKey(10)

    # If the user presses 'q', exit the loop
    if key == ord('q'):
        break

# Release the webcam
cap.release()

# Destroy the window
cv2.destroyAllWindows()
