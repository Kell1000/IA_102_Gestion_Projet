import cv2
import os
from config import EMPLOYEES_DIR, CAMERA_INDEX, MODEL_FILE, LABELS_FILE
from utils.face_utils import get_face_detector, train_model


def register_employee(name):
    os.makedirs(EMPLOYEES_DIR, exist_ok=True)
    detector = get_face_detector()
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[!] Impossible d'ouvrir la caméra.")
        return
    
    print(f"\n[INFO] Enregistrement : {name}")
    print("[INFO] Appuyez sur 's' pour sauvegarder | 'q' pour quitter\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.putText(frame, f"Employe: {name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Appuyez sur 's' pour sauvegarder", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Enregistrement Employe", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s') and len(faces) > 0:
            filepath = os.path.join(EMPLOYEES_DIR, f"{name}.jpg")
            cv2.imwrite(filepath, frame)
            print(f"[✔] Photo sauvegardée : {filepath}")
            break
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n[INFO] Entraînement du modèle...")
    train_model(EMPLOYEES_DIR, MODEL_FILE, LABELS_FILE)


if __name__ == "__main__":
    nom = input("Entrez le nom de l'employé : ").strip()
    if nom:
        register_employee(nom)
    else:
        print("[!] Nom invalide.")