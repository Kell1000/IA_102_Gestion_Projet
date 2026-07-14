import os
import pandas as pd
from datetime import datetime


def init_attendance_file(file_path):
    """Crée le fichier de pointage s'il n'existe pas."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=["Nom", "Date", "Heure"])
        df.to_csv(file_path, index=False)


def mark_attendance(name, file_path):
    """Enregistre le pointage d'un employé (une seule fois par jour)."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    init_attendance_file(file_path)
    df = pd.read_csv(file_path)
    
    if not ((df["Nom"] == name) & (df["Date"] == date_str)).any():
        new_row = pd.DataFrame([[name, date_str, time_str]], columns=["Nom", "Date", "Heure"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(file_path, index=False)
        print(f"[✔] {name} pointé à {time_str}")
        return True
    return False