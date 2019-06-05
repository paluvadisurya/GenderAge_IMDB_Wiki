import numpy as np
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
import cv2
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
while True:
    _, frame = cap.read()
    faces = detector(frame,0)
    if len(faces) > 0:
        text = "{} Nos of Faces Detected".format(len(faces))
        cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_DUPLEX,
        0.5, (0,0,255),1)
    for face in faces:
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(frame, (x,w),(x+h,y+h), (255,0,0),3)
    cv2.imshow("Detecting Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()