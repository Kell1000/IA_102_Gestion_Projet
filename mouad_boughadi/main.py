import cv2
from datetime import datetime

from config import (
    EMPLOYEES_DIR, ATTENDANCE_FILE, CAMERA_INDEX,
    MODEL_FILE, LABELS_FILE, CONFIDENCE_THRESHOLD
)
from utils.face_utils import get_face_detector, load_model, train_model
from utils.attendance_utils import mark_attendance, init_attendance_file


def run_attendance_system():
    print("=" * 50)
    print("  POINTAGE PAR RECONNAISSANCE FACIALE")
    print("=" * 50)
    
    init_attendance_file(ATTENDANCE_FILE)
    
    recognizer, label_map = load_model(MODEL_FILE, LABELS_FILE)
    if recognizer is None:
        print("[INFO] Aucun modèle trouvé. Entraînement en cours...")
        if not train_model(EMPLOYEES_DIR, MODEL_FILE, LABELS_FILE):
            print("[!] Enregistrez d'abord un employé avec register_employee.py")
            return
        recognizer, label_map = load_model(MODEL_FILE, LABELS_FILE)
    
    print(f"[INFO] {len(label_map)} employé(s) : {list(label_map.values())}")
    
    detector = get_face_detector()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print("[!] Impossible d'ouvrir la caméra.")
        return
    
    print("\n[INFO] Système lancé. Appuyez sur 'q' pour quitter.\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            label_id, confidence = recognizer.predict(face_roi)
            
            if confidence < CONFIDENCE_THRESHOLD:
                name = label_map.get(label_id, "Inconnu")
                color = (0, 255, 0)
                mark_attendance(name, ATTENDANCE_FILE)
            else:
                name = "Inconnu"
                color = (0, 0, 255)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-35), (x+w, y), color, cv2.FILLED)
            cv2.putText(frame, f"{name} ({int(confidence)})", (x+5, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Pointage - Reconnaissance Faciale", frame)