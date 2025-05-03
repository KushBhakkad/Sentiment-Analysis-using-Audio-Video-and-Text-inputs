import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

FACE_CLASSIFIER_PATH = "haarcascade_frontalface_default.xml"
MODEL_PATH = "model.h5"
face_classifier = cv2.CascadeClassifier(FACE_CLASSIFIER_PATH)
classifier = load_model(MODEL_PATH)

# Define the emotion labels
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detect_emotion(frame):
    """
    Detect emotions in the provided frame.
    
    Args:
    frame: A single frame from the video capture.
    
    Returns:
    frame: The frame with detected faces and emotion labels.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            predictions = classifier.predict(roi)[0]
            max_index = np.argmax(predictions)
            label = EMOTION_LABELS[max_index]
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def main():
    """
    Main function to start the emotion detection from webcam.
    """
    # Start video capture from the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = detect_emotion(frame)
        cv2.imshow('Emotion Detector', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
