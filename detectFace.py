import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta

def train_faces(data_folder):
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        print(f"Error: Face detection model not found at {cascade_path}")
        return

    face_detector = cv2.CascadeClassifier(cascade_path)
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    face_images = []
    face_labels = []
    name_by_id = {}
    current_id = 0

    if not os.path.exists(data_folder):
        print(f"Error: Dataset folder '{data_folder}' not found!")
        return

    person_folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
    if not person_folders:
        print("Error: No person folders found in dataset!")
        return

    print("Starting training...")
    for person_name in person_folders:
        person_path = os.path.join(data_folder, person_name)
        image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            continue

        name_by_id[current_id] = person_name
        for img_name in image_files:
            img_path = os.path.join(person_path, img_name)
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue

            faces = face_detector.detectMultiScale(gray, 1.2, 6, minSize=(40, 40))
            for (x, y, w, h) in faces:
                face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                face = cv2.equalizeHist(face)
                face_images.append(face)
                face_labels.append(current_id)
                break
        current_id += 1

    if not face_images:
        print("No faces found. Aborting.")
        return

    face_recognizer.train(face_images, np.array(face_labels))
    face_recognizer.save('face_model.yml')
    np.save('labels.npy', name_by_id)
    print("Training complete.")

def recognize_faces_live():
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    try:
        face_recognizer.read('face_model.yml')
        name_by_id = np.load('labels.npy', allow_pickle=True).item()
    except Exception as e:
        print(f"Model loading failed: {e}")
        return

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print("Error: Could not access camera.")
        return

    file_path = 'attendance.xlsx'
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
    else:
        df = pd.DataFrame(columns=['Name', 'In Time', 'Out Time'])

    last_seen = {}
    active_names = set()

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 6, minSize=(10, 10))
        now = datetime.now()
        seen_now = set()

        for (x, y, w, h) in faces:
            face = cv2.resize(gray [y:y+h, x:x+w], (100, 100))
            label, confidence = face_recognizer.predict(face)
            name = name_by_id.get(label, "Unknown")

            if confidence < 60 and name != "Unknown":
                seen_now.add(name)
                last_seen[name] = now

                if name not in active_names:
                    df = pd.concat([df, pd.DataFrame([{'Name': name, 'In Time': now.strftime("%Y-%m-%d %H:%M:%S"), 'Out Time': ''}])], ignore_index=True)
                    active_names.add(name)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.1f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        for name in list(active_names):
            if name not in seen_now:
                last_time = last_seen.get(name)
                if (now - last_time > timedelta(minutes=1)) or (cv2.waitKey(1) & 0xFF == ord('q')):
                    rows = df[(df['Name'] == name) & (df['Out Time'] == '')]
                    if not rows.empty:
                        idx = rows.index[-1]
                        df.at[idx, 'Out Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


        df.to_excel(file_path, index=False)
        cv2.imshow("Live Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # On exit, write Out Time for all still-active names
            for name in list(active_names):
                idx = df[(df['Name'] == name) & (df['Out Time'] == '')].last_valid_index()
                if idx is not None:
                    df.at[idx, 'Out Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df.to_excel(file_path, index=False)
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if os.path.exists('dataset'):
        train_faces('dataset')
        recognize_faces_live()
    else:
        print("Error: 'dataset' folder not found.")
