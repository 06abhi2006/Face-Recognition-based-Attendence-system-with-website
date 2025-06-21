import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

def verify_cascade():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        print(f"ERROR: Cascade file not found at {cascade_path}")
        print("Please ensure OpenCV is properly installed")
        return None
    return cv2.CascadeClassifier(cascade_path)

def train_faces(data_folder):
    face_detector = verify_cascade()
    if face_detector is None:
        return

    # Create face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    face_samples = []
    ids = []
    id_to_name = {}
    current_id = 0

    print(f"Scanning dataset in {data_folder}...")
    for root, dirs, files in os.walk(data_folder):
        for person_name in dirs:
            person_dir = os.path.join(root, person_name)
            print(f"Processing {person_name}...")
            
            id_to_name[current_id] = person_name
            image_count = 0
            
            for file in os.listdir(person_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_dir, file)
                    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
                    if gray_image is None:
                        print(f"  Could not read image: {file}")
                        continue
                        
                    # Detect faces with more sensitive parameters
                    faces = face_detector.detectMultiScale(
                        gray_image,
                        scaleFactor=1.05,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    if len(faces) == 0:
                        print(f"  No face detected in {file}")
                        continue
                        
                    for (x, y, w, h) in faces:
                        face_roi = gray_image[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (200, 200))
                        face_samples.append(face_roi)
                        ids.append(current_id)
                        image_count += 1
            
            if image_count > 0:
                print(f"  Added {image_count} samples for {person_name}")
                current_id += 1
            else:
                print(f"  WARNING: No valid faces found for {person_name}")
                del id_to_name[current_id]

    if len(face_samples) == 0:
        print("ERROR: No face samples found in the entire dataset!")
        return

    print(f"Training on {len(face_samples)} samples...")
    face_recognizer.train(face_samples, np.array(ids))
    
    # Save the trained model
    face_recognizer.save('face_model.yml')
    np.save('labels.npy', id_to_name)
    print("Training completed successfully!")

def test_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return False
    
    print("Camera test - press 'q' to exit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Couldn't capture frame")
            break
            
        cv2.imshow('Camera Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return True

def recognize_faces():
    # Verify face detector
    face_detector = verify_cascade()
    if face_detector is None:
        return

    # Load trained model
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('face_model.yml')
        id_to_name = np.load('labels.npy', allow_pickle=True).item()
        print("Model loaded successfully")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    print("Starting face recognition...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Couldn't capture frame")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with more sensitive parameters
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        print(f"Detected {len(faces)} faces")  # Debug output
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            
            id, confidence = recognizer.predict(face_roi)
            
            if confidence < 150:  # Adjust threshold as needed
                name = id_to_name.get(id, "Unknown")
                color = (0, 255, 0)  # Green
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red
                
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} {confidence:.1f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def run_attendance():
    """
    Runs face recognition, updates attendance.xlsx, and returns a summary dict.
    """
    face_detector = verify_cascade()
    if face_detector is None:
        return {'status': 'error', 'message': 'Cascade not found'}

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('face_model.yml')
        id_to_name = np.load('labels.npy', allow_pickle=True).item()
    except Exception as e:
        return {'status': 'error', 'message': f'Error loading model: {e}'}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {'status': 'error', 'message': 'Could not open camera'}

    recognized = set()
    attendance_data = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            id, confidence = recognizer.predict(face_roi)
            if confidence < 150:
                name = id_to_name.get(id, "Unknown")
                if name != "Unknown":
                    recognized.add(name)
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    # Update attendance.xlsx
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    if os.path.exists('attendance.xlsx'):
        df = pd.read_excel('attendance.xlsx')
    else:
        df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
    for name in recognized:
        df = pd.concat([df, pd.DataFrame({'Name': [name], 'Date': [date_str], 'Time': [time_str]})], ignore_index=True)
    df.to_excel('attendance.xlsx', index=False)
    return {'status': 'success', 'recognized': list(recognized), 'date': date_str, 'time': time_str}