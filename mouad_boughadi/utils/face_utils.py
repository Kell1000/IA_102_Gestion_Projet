import cv2
import os
import numpy as np


def get_face_detector():
    """Retourne le détecteur de visage Haar Cascade d'OpenCV."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def detect_faces(frame, detector):
    """Détecte les visages dans une image."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
    return gray, faces


def train_model(employees_dir, model_file, labels_file):
    """Entraîne le modèle LBPH sur les photos des employés."""
    detector = get_face_detector()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    faces = []
    labels = []
    label_map = {}
    current_id = 0
    
    if not os.path.exists(employees_dir):
        print(f"[!] Dossier '{employees_dir}' introuvable.")
        return False
    
    for file in os.listdir(employees_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(file)[0]
            path = os.path.join(employees_dir, file)
            
            image = cv2.imread(path)
            if image is None:
                continue
            
            gray, detected = detect_faces(image, detector)
            
            for (x, y, w, h) in detected:
                face_roi = gray[y:y+h, x:x+w]
                
                if name not in label_map.values():
                    label_map[current_id] = name
                    label_id = current_id
                    current_id += 1
                else:
                    label_id = [k for k, v in label_map.items() if v == name][0]
                
                faces.append(face_roi)
                labels.append(label_id)
                print(f"[✔] Visage détecté pour : {name}")
    
    if not faces:
        print("[!] Aucun visage détecté dans les photos.")
        return False
    
    recognizer.train(faces, np.array(labels))
    
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    recognizer.save(model_file)
    
    with open(labels_file, "w", encoding="utf-8") as f:
        for id_, name in label_map.items():
            f.write(f"{id_},{name}\n")
    
    print(f"[✔] Modèle entraîné avec {len(faces)} visage(s).")
    return True


def load_model(model_file, labels_file):
    """Charge le modèle et les labels."""
    if not os.path.exists(model_file) or not os.path.exists(labels_file):
        return None, {}
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_file)
    
    label_map = {}
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            id_, name = line.strip().split(",")
            label_map[int(id_)] = name
    
    return recognizer, label_map
