import cv2
import numpy as np
import os


def start_webcam_feed():
   
    cascade_path = '../haarcascades/haarcascade_frontalface_default.xml'
    
    if not os.path.exists(cascade_path):
        print(f"[ERROR] Haar cascade file not found at {cascade_path}")
        print("Please download it from the official OpenCV GitHub repository.")
        return

    face_cascade = cv2.CascadeClassifier(cascade_path)
    
   
    
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    cap = cv2.VideoCapture(0)
    print("[INFO] Starting webcam. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
       
            predicted_emotion = "Emotion: Placeholder"
            predicted_age = "Age: 25"
            predicted_gender = "Gender: M"
            
            text = f"{predicted_gender}, {predicted_age}, {predicted_emotion}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Realtime Facial Details Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_webcam_feed()
