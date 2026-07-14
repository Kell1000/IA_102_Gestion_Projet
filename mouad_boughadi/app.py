import streamlit as st
import cv2
import os
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image

from config import (
    EMPLOYEES_DIR, ATTENDANCE_FILE, CAMERA_INDEX,
    MODEL_FILE, LABELS_FILE, CONFIDENCE_THRESHOLD
)
from utils.face_utils import get_face_detector, load_model, train_model
from utils.attendance_utils import mark_attendance, init_attendance_file
# ==================== FONCTION UTILITAIRE ====================
def safe_read_attendance(file_path):
    """Lit le CSV de manière sécurisée, même s'il est vide ou corrompu."""
    try:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            df = pd.read_csv(file_path)
            # Vérifier que les colonnes existent
            if not all(col in df.columns for col in ["Nom", "Date", "Heure"]):
                df = pd.DataFrame(columns=["Nom", "Date", "Heure"])
                df.to_csv(file_path, index=False)
            return df
        else:
            # Fichier vide ou inexistant : le recréer avec les en-têtes
            df = pd.DataFrame(columns=["Nom", "Date", "Heure"])
            df.to_csv(file_path, index=False)
            return df
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        df = pd.DataFrame(columns=["Nom", "Date", "Heure"])
        df.to_csv(file_path, index=False)
        return df


# ==================== CONFIGURATION DE LA PAGE ====================
st.set_page_config(
    page_title="Pointage Facial",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLE CSS ====================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #17a2b8;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# ==================== INITIALISATION ====================
init_attendance_file(ATTENDANCE_FILE)
os.makedirs(EMPLOYEES_DIR, exist_ok=True)


# ==================== SIDEBAR - NAVIGATION ====================
st.sidebar.image("https://img.icons8.com/color/96/face-id.png", width=80)
st.sidebar.title("🎯 Menu")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "📸 Enregistrer un Employé", "✅ Marquer la Présence", "📊 Historique", "⚙️ Paramètres"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"📅 **Date** : {datetime.now().strftime('%d/%m/%Y')}\n\n🕐 **Heure** : {datetime.now().strftime('%H:%M:%S')}")


# ==================== PAGE 1 : ACCUEIL ====================
if page == "🏠 Accueil":
    st.markdown('<h1 class="main-header">👤 Système de Pointage par Reconnaissance Faciale</h1>', unsafe_allow_html=True)
    
    st.markdown("### Bienvenue dans votre système de gestion de présence ! 🎉")
    
    col1, col2, col3 = st.columns(3)
    
    # Statistiques
    employees_count = len([f for f in os.listdir(EMPLOYEES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    df = pd.read_csv(ATTENDANCE_FILE) if os.path.exists(ATTENDANCE_FILE) else pd.DataFrame(columns=["Nom", "Date", "Heure"])
    today = datetime.now().strftime("%Y-%m-%d")
    today_count = len(df[df["Date"] == today]) if not df.empty else 0
    total_records = len(df)
    
    with col1:
        st.metric("👥 Employés enregistrés", employees_count)
    with col2:
        st.metric("✅ Présents aujourd'hui", today_count)
    with col3:
        st.metric("📊 Total pointages", total_records)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📖 Comment utiliser l'application ?
    
    1. **📸 Enregistrer un Employé** : Ajoutez un nouvel employé en capturant sa photo via la caméra
    2. **✅ Marquer la Présence** : Lancez le système de reconnaissance pour pointer automatiquement
    3. **📊 Historique** : Consultez et exportez l'historique des présences
    4. **⚙️ Paramètres** : Ajustez la sensibilité et gérez les employés
    """)
    
    st.markdown("---")
    st.success("💡 **Astuce** : Assurez-vous d'avoir une bonne luminosité pour une meilleure reconnaissance !")


# ==================== PAGE 2 : ENREGISTRER UN EMPLOYÉ ====================
elif page == "📸 Enregistrer un Employé":
    st.markdown('<h1 class="main-header">📸 Enregistrer un Nouvel Employé</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Informations")
        employee_name = st.text_input("Nom de l'employé", placeholder="Ex: mouad").strip()
        
        st.markdown("### 📷 Capture Photo")
        st.info("Cliquez sur le bouton ci-dessous, positionnez votre visage face à la caméra, puis prenez la photo.")
        
        camera_photo = st.camera_input("Prendre une photo")
    
    with col2:
        st.markdown("### 👁️ Aperçu & Détection")
        
        if camera_photo is not None:
            # Convertir en image OpenCV
            image = Image.open(camera_photo)
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Détecter le visage
            detector = get_face_detector()
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 5)
            
            # Dessiner les rectangles
            img_display = img_bgr.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            img_display_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
            st.image(img_display_rgb, caption=f"✅ {len(faces)} visage(s) détecté(s)", use_container_width=True)
            
            if len(faces) == 0:
                st.error("❌ Aucun visage détecté. Reprenez la photo.")
            elif len(faces) > 1:
                st.warning("⚠️ Plusieurs visages détectés. Assurez-vous qu'une seule personne soit visible.")
            else:
                st.success("✅ Visage détecté avec succès !")
                
                if st.button("💾 Enregistrer cet Employé", type="primary", use_container_width=True):
                    if not employee_name:
                        st.error("❌ Veuillez entrer un nom avant d'enregistrer.")
                    else:
                        # Sauvegarder la photo
                        filepath = os.path.join(EMPLOYEES_DIR, f"{employee_name}.jpg")
                        cv2.imwrite(filepath, img_bgr)
                        
                        # Réentraîner le modèle
                        with st.spinner("🔄 Entraînement du modèle en cours..."):
                            success = train_model(EMPLOYEES_DIR, MODEL_FILE, LABELS_FILE)
                        
                        if success:
                            st.success(f"🎉 Employé **{employee_name}** enregistré avec succès !")
                            st.balloons()
                        else:
                            st.error("❌ Erreur lors de l'entraînement.")
    
    # Liste des employés déjà enregistrés
    st.markdown("---")
    st.markdown("### 👥 Employés Déjà Enregistrés")
    
    employees_files = [f for f in os.listdir(EMPLOYEES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if employees_files:
        cols = st.columns(4)
        for idx, file in enumerate(employees_files):
            with cols[idx % 4]:
                img_path = os.path.join(EMPLOYEES_DIR, file)
                st.image(img_path, caption=os.path.splitext(file)[0], use_container_width=True)
    else:
        st.info("Aucun employé enregistré pour le moment.")


# ==================== PAGE 3 : MARQUER LA PRÉSENCE ====================
elif page == "✅ Marquer la Présence":
    st.markdown('<h1 class="main-header">✅ Marquer la Présence</h1>', unsafe_allow_html=True)
    
    # Vérifier si un modèle existe
    recognizer, label_map = load_model(MODEL_FILE, LABELS_FILE)
    
    if recognizer is None or not label_map:
        st.error("⚠️ Aucun modèle entraîné trouvé ! Enregistrez d'abord au moins un employé.")
        st.stop()
    
    st.info(f"👥 **{len(label_map)} employé(s) enregistré(s)** : {', '.join(label_map.values())}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📷 Capture Photo")
        st.markdown("Prenez une photo pour marquer votre présence.")
        
        photo = st.camera_input("Prendre une photo pour pointer")
    
    with col2:
        st.markdown("### 🔍 Résultat de la Reconnaissance")
        
        if photo is not None:
            image = Image.open(photo)
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            detector = get_face_detector()
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
            
            if len(faces) == 0:
                st.error("❌ Aucun visage détecté dans la photo.")
            else:
                img_display = img_bgr.copy()
                recognized_names = []
                
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    label_id, confidence = recognizer.predict(face_roi)
                    
                    if confidence < CONFIDENCE_THRESHOLD:
                        name = label_map.get(label_id, "Inconnu")
                        color = (0, 255, 0)
                        recognized_names.append((name, confidence))
                    else:
                        name = "Inconnu"
                        color = (0, 0, 255)
                    
                    cv2.rectangle(img_display, (x, y), (x+w, y+h), color, 3)
                    cv2.rectangle(img_display, (x, y-40), (x+w, y), color, cv2.FILLED)
                    cv2.putText(img_display, f"{name}", (x+5, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                img_display_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
                st.image(img_display_rgb, use_container_width=True)
                
                # Marquer la présence
                for name, confidence in recognized_names:
                    if name != "Inconnu":
                        was_marked = mark_attendance(name, ATTENDANCE_FILE)
                        if was_marked:
                            st.success(f"🎉 Présence enregistrée pour **{name}** à {datetime.now().strftime('%H:%M:%S')}")
                            st.balloons()
                        else:
                            st.warning(f"⚠️ **{name}** a déjà pointé aujourd'hui.")
                        st.info(f"🎯 Confiance : **{100 - int(confidence)}%**")
                
                if not recognized_names:
                    st.error("❌ Visage non reconnu. Vérifiez que vous êtes bien enregistré ou ajustez la sensibilité.")
    
    # Présents aujourd'hui
    st.markdown("---")
    st.markdown("### 📋 Présents Aujourd'hui")
    
    df = pd.read_csv(ATTENDANCE_FILE)
    today = datetime.now().strftime("%Y-%m-%d")
    today_df = df[df["Date"] == today]
    
    if not today_df.empty:
        st.dataframe(today_df, use_container_width=True, hide_index=True)
    else:
        st.info("Aucun pointage aujourd'hui pour le moment.")


# ==================== PAGE 4 : HISTORIQUE ====================
elif page == "📊 Historique":
    st.markdown('<h1 class="main-header">📊 Historique des Présences</h1>', unsafe_allow_html=True)
    
    if not os.path.exists(ATTENDANCE_FILE):
        st.warning("Aucun historique disponible.")
        st.stop()
    
    df = pd.read_csv(ATTENDANCE_FILE)
    
    if df.empty:
        st.info("Aucune présence enregistrée pour le moment.")
        st.stop()
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dates_dispo = sorted(df["Date"].unique(), reverse=True)
        selected_date = st.selectbox("📅 Filtrer par date", ["Toutes"] + list(dates_dispo))
    
    with col2:
        noms_dispo = sorted(df["Nom"].unique())
        selected_name = st.selectbox("👤 Filtrer par employé", ["Tous"] + list(noms_dispo))
    
    with col3:
        st.markdown("### 📥 Export")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Télécharger CSV",
            csv,
            "presences.csv",
            "text/csv",
            use_container_width=True
        )
    
    # Application des filtres
    filtered_df = df.copy()
    if selected_date != "Toutes":
        filtered_df = filtered_df[filtered_df["Date"] == selected_date]
    if selected_name != "Tous":
        filtered_df = filtered_df[filtered_df["Nom"] == selected_name]
    
    # Statistiques
    st.markdown("### 📈 Statistiques")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total pointages", len(filtered_df))
    with col2:
        st.metric("Employés uniques", filtered_df["Nom"].nunique())
    with col3:
        st.metric("Jours d'activité", filtered_df["Date"].nunique())
    with col4:
        if not filtered_df.empty:
            top_employee = filtered_df["Nom"].value_counts().idxmax()
            st.metric("Top employé", top_employee)
    
    # Tableau
    st.markdown("### 📋 Détail des Présences")
    st.dataframe(filtered_df.sort_values(by=["Date", "Heure"], ascending=[False, False]),
                 use_container_width=True, hide_index=True)
    
    # Graphique
    if not filtered_df.empty and selected_date == "Toutes":
        st.markdown("### 📊 Pointages par Jour")
        chart_data = filtered_df.groupby("Date").size().reset_index(name="Nombre")
        st.bar_chart(chart_data.set_index("Date"))


# ==================== PAGE 5 : PARAMÈTRES ====================
elif page == "⚙️ Paramètres":
    st.markdown('<h1 class="main-header">⚙️ Paramètres</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Sensibilité", "👥 Gestion Employés", "🗑️ Réinitialisation"])
    
    with tab1:
        st.markdown("### 🎯 Ajuster la Sensibilité de Reconnaissance")
        st.info(f"**Seuil actuel** : {CONFIDENCE_THRESHOLD}\n\n"
                "🔒 Bas (50) = strict | ⚖️ Moyen (70) = équilibré | 🔓 Haut (90) = permissif")
        
        new_threshold = st.slider("Nouveau seuil", 30, 100, CONFIDENCE_THRESHOLD)
        
        if st.button("💾 Sauvegarder", type="primary"):
            with open("config.py", "r") as f:
                content = f.read()
            content = content.replace(
                f"CONFIDENCE_THRESHOLD = {CONFIDENCE_THRESHOLD}",
                f"CONFIDENCE_THRESHOLD = {new_threshold}"
            )
            with open("config.py", "w") as f:
                f.write(content)
            st.success(f"✅ Seuil mis à jour : {new_threshold}. Redémarrez l'application.")
    
    with tab2:
        st.markdown("### 👥 Supprimer un Employé")
        employees_files = [f for f in os.listdir(EMPLOYEES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if employees_files:
            employee_to_delete = st.selectbox("Sélectionner un employé", 
                                              [os.path.splitext(f)[0] for f in employees_files])
            
            if st.button("🗑️ Supprimer", type="secondary"):
                filepath = os.path.join(EMPLOYEES_DIR, f"{employee_to_delete}.jpg")
                if os.path.exists(filepath):
                    os.remove(filepath)
                    with st.spinner("Réentraînement du modèle..."):
                        train_model(EMPLOYEES_DIR, MODEL_FILE, LABELS_FILE)
                    st.success(f"✅ Employé **{employee_to_delete}** supprimé.")
                    st.rerun()
        else:
            st.info("Aucun employé à supprimer.")
    
    with tab3:
        st.markdown("### 🗑️ Réinitialiser l'Historique")
        st.warning("⚠️ Cette action supprimera **tous les pointages**. Cette action est irréversible !")
        
        confirm = st.checkbox("Je confirme vouloir tout supprimer")
        
        if st.button("🗑️ Effacer tout l'historique", disabled=not confirm, type="secondary"):
            df_empty = pd.DataFrame(columns=["Nom", "Date", "Heure"])
            df_empty.to_csv(ATTENDANCE_FILE, index=False)
            st.success("✅ Historique effacé.")
            st.rerun()


# ==================== FOOTER ====================
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Développé par Mouad**")
st.sidebar.caption("© 2026 - Système de Pointage Facial")