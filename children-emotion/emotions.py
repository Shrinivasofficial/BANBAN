import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import draw_text, draw_bounding_box, apply_offsets
from utils.preprocessor import preprocess_input
from datetime import datetime
import os

# Paths to model files
emotion_model_path = './models/emotion_model.hdf5'
face_cascade_path = './models/haarcascade_frontalface_default.xml'

# Emotion labels
emotion_labels = get_labels('fer2013')

# Bounding box and emotion offsets
frame_window = 10
emotion_offsets = (20, 40)

# Load models
face_cascade = cv2.CascadeClassifier(face_cascade_path)
emotion_classifier = load_model(emotion_model_path)

# Get input shape for the emotion model
emotion_target_size = emotion_classifier.input_shape[1:3]

# For calculating modes of detected emotions
emotion_window = []

# User Input Modes
print("Select Input Mode:")
print("1. Use Webcam")
print("2. Submit Video")
print("3. Submit Photo")
mode_selection = int(input("Enter your choice (1/2/3): "))

if mode_selection == 1:
    video_source = 0  # Webcam source
elif mode_selection == 2:
    video_source = input("Enter the path to the video file: ")
elif mode_selection == 3:
    photo_path = input("Enter the path to the photo file: ")
else:
    print("Invalid selection!")
    exit()

# CSV output file to save emotions and probabilities
csv_file = 'emotion_results.csv'
with open(csv_file, 'w') as f:
    f.write("emotion,percent\n")  # Add header

    if mode_selection in [1, 2]:  # Webcam or Video Mode
        cap = cv2.VideoCapture(video_source)

        while cap.isOpened():
            ret, bgr_image = cap.read()
            if not ret:
                print("End of video or unable to read frame.")
                break

            # Convert frame to grayscale and RGB
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2BGR)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(
                gray_image,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            for face_coordinates in faces:
                x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                gray_face = gray_image[y1:y2, x1:x2]

                try:
                    gray_face = cv2.resize(gray_face, emotion_target_size)
                except Exception as e:
                    print(f"Error resizing face region: {e}")
                    continue

                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)

                # Predict emotion
                emotion_prediction = emotion_classifier.predict(gray_face)
                emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[emotion_label_arg]
                emotion_window.append(emotion_text)

                if len(emotion_window) > frame_window:
                    emotion_window.pop(0)

                try:
                    emotion_mode = mode(emotion_window)
                except:
                    emotion_mode = emotion_text

                # Set bounding box color based on emotion
                if emotion_text == 'angry':
                    color = emotion_probability * np.asarray((255, 0, 0))
                elif emotion_text == 'sad':
                    color = emotion_probability * np.asarray((0, 0, 255))
                elif emotion_text == 'happy':
                    color = emotion_probability * np.asarray((255, 255, 0))
                elif emotion_text == 'surprise':
                    color = emotion_probability * np.asarray((0, 255, 255))
                else:
                    color = emotion_probability * np.asarray((0, 255, 0))

                color = color.astype(int).tolist()

                # Write to CSV if emotion probability is above threshold
                if emotion_probability >= 0.5:
                    csv_data = f"{emotion_text},{int(emotion_probability * 100)}\n"
                    f.write(csv_data)

                # Draw bounding box and text
                draw_bounding_box(face_coordinates, rgb_image, color)
                draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)
                draw_text(face_coordinates, rgb_image, f"{int(emotion_probability * 100)}%", color, 0, -20, 1, 1)

            # Convert RGB back to BGR for display
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            # Display the processed frame
            cv2.imshow('Emotion Detection', bgr_image)

            # Quit when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

    elif mode_selection == 3:  # Photo Mode
        bgr_image = cv2.imread(photo_path)
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]

            try:
                gray_face = cv2.resize(gray_face, emotion_target_size)
            except Exception as e:
                print(f"Error resizing face region: {e}")
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)

            # Predict emotion
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]

            # Write to CSV if emotion probability is above threshold
            if emotion_probability >= 0.5:
                csv_data = f"{emotion_text},{int(emotion_probability * 100)}\n"
                f.write(csv_data)

            # Draw bounding box and text
            draw_bounding_box(face_coordinates, rgb_image, (0, 255, 0))
            draw_text(face_coordinates, rgb_image, emotion_text, (0, 255, 0), 0, -45, 1, 1)
            draw_text(face_coordinates, rgb_image, f"{int(emotion_probability * 100)}%", (0, 255, 0), 0, -20, 1, 1)

        # Convert RGB back to BGR for display
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Display the processed photo
        cv2.imshow('Emotion Detection - Photo', bgr_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
