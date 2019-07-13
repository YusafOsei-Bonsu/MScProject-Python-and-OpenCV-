import numpy as np
import cv2
import pickle
from modified_LBPH import ModifiedLBPH

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
# The face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Trained data
recognizer.read("trainer.yml")

# Save labels so they can be used by main.py
labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

# Accesses the device's camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Convert captured frame from RGB to greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    # Prints a set of 4 values if a face is detected
    for (x, y, w, h) in faces:
        # roi = region of interest
        print(x, y, w, h)
        # (ycord_start, ycord_end)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Make predictions
        id_, confidence = recognizer.predict(roi_gray)

        if confidence >= 4:
            print(id_)
            print(labels[id_])

            # Printing the name of the detected person on screen
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        # img_item = "my-image.png"
        # cv2.imwrite(img_item, roi_gray)

        # BGR
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        # Draws the rectangle around the face
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Show the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()