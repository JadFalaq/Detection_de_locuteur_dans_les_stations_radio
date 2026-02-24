import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import time
from pathlib import Path
from scipy import signal
import soundfile as sf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from config import DATASET_ROOT, OUTPUT_ROOT, SAMPLE_RATE

# ========================================
# CONFIGURATION PAGE
# ========================================

st.set_page_config(
    page_title=" Système d'Identification Police Radio",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CSS PERSONNALISÉ
# ========================================

st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-correct {
        background-color: #28a745;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-size: 1.5rem;
        text-align: center;
        font-weight: bold;
    }
    .prediction-wrong {
        background-color: #dc3545;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-size: 1.5rem;
        text-align: center;
        font-weight: bold;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# CHARGEMENT MODÈLE
# ========================================

@st.cache_resource
def load_model():
    """Charge le modèle et les encodeurs"""
    models_dir = OUTPUT_ROOT / 'models'
    
    model = joblib.load(models_dir / 'svm_police_radio_model.pkl')
    scaler = joblib.load(models_dir / 'scaler.pkl')
    label_encoder = joblib.load(models_dir / 'label_encoder.pkl')
    
    return model, scaler, label_encoder

model, scaler, label_encoder = load_model()

# ========================================
# FONCTIONS AUDIO
# ========================================

def simulate_police_radio(audio, sr=16000, snr_db=15):
    sos = signal.butter(8, [300, 3000], 'bandpass', fs=sr, output='sos')
    audio_filtered = signal.sosfilt(sos, audio)
    
    noise = np.random.normal(0, 1, len(audio_filtered))
    signal_power = np.mean(audio_filtered ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    noise_factor = np.sqrt(signal_power / (10 ** (snr_db/10) * noise_power))
    audio_noisy = audio_filtered + noise_factor * noise
    
    audio_compressed = np.tanh(audio_noisy * 2.5) / 2.5
    
    return librosa.util.normalize(audio_compressed)

def extract_robust_features(audio, sr=16000):
    frame_length = int(0.025 * sr)
    hop_length = frame_length // 2
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20,
                                n_fft=frame_length, hop_length=hop_length)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    spectral_contrast = librosa.feature.spectral_contrast(
        y=audio, sr=sr, hop_length=hop_length, n_bands=6
    )
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, hop_length=hop_length
    )
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sr, hop_length=hop_length
    )
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length)
    
    features = np.concatenate([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(delta_mfcc, axis=1), np.std(delta_mfcc, axis=1),
        np.mean(delta2_mfcc, axis=1), np.std(delta2_mfcc, axis=1),
        np.mean(spectral_contrast, axis=1), np.std(spectral_contrast, axis=1),
        [np.mean(spectral_centroid), np.std(spectral_centroid)],
        [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],
        [np.mean(zcr), np.std(zcr)]
    ])
    
    return features

def predict_realtime(audio_chunk, sr, snr_db):
    try:
        audio_radio = simulate_police_radio(audio_chunk, sr, snr_db)
        features = extract_robust_features(audio_radio, sr)
        features_scaled = scaler.transform([features])
        
        agent_encoded = model.predict(features_scaled)[0]
        agent_id = label_encoder.inverse_transform([agent_encoded])[0]
        
        decision_scores = model.decision_function(features_scaled)[0]
        top5_indices = np.argsort(decision_scores)[-5:][::-1]
        top5_agents = label_encoder.inverse_transform(top5_indices)
        top5_scores = decision_scores[top5_indices]
        
        top5_scores_norm = np.exp(top5_scores) / np.exp(top5_scores).sum()
        
        top5_predictions = [
            (agent, score*100) 
            for agent, score in zip(top5_agents, top5_scores_norm)
        ]
        
        confidence = top5_scores_norm[0] * 100
        
        return agent_id, confidence, top5_predictions
    except:
        return None, 0, []

# ========================================
# INTERFACE PRINCIPALE
# ========================================

st.markdown('<div class="main-title"> Système d\'Identification Police Radio </div>', 
            unsafe_allow_html=True)

# ========================================
# SIDEBAR - CONFIGURATION
# ========================================

with st.sidebar:
    st.header(" Configuration")
    
    # Liste des agents
    agents_list = sorted(label_encoder.classes_)
    
    st.subheader(" Sélection Agent")
    selected_agent = st.selectbox(
        "Choisir un agent:",
        agents_list,
        index=0
    )
    
    # Liste des fichiers audio pour cet agent
    agent_path = DATASET_ROOT / selected_agent
    
    audio_files_list = []
    if agent_path.exists():
        for video_dir in agent_path.iterdir():
            if video_dir.is_dir():
                for audio_file in video_dir.glob('*.wav'):
                    audio_files_list.append({
                        'path': audio_file,
                        'display': f"{video_dir.name}/{audio_file.name}"
                    })
    
    if audio_files_list:
        st.subheader("🎵 Fichier Audio")
        selected_audio_idx = st.selectbox(
            "Choisir un fichier:",
            range(len(audio_files_list)),
            format_func=lambda x: audio_files_list[x]['display']
        )
        selected_audio_path = audio_files_list[selected_audio_idx]['path']
    else:
        st.error("Aucun fichier audio trouvé pour cet agent")
        st.stop()
    
    st.divider()
    
    st.subheader("🔊 Conditions Radio")
    snr_db = st.slider(
        "Niveau de bruit (SNR)",
        min_value=5,
        max_value=25,
        value=15,
        step=1,
        help="Plus le SNR est élevé, meilleure est la qualité audio"
    )
    
    st.info(f"**SNR {snr_db} dB**\n\n" + 
            ("🟢 Excellente qualité" if snr_db >= 20 else
             "🟡 Qualité moyenne" if snr_db >= 12 else
             "🔴 Mauvaise qualité"))
    
    st.divider()
    
    st.subheader(" Analyse Temps Réel")
    window_size = st.slider(
        "Fenêtre d'analyse (secondes)",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Durée du segment audio pour chaque prédiction"
    )
    
    st.divider()
    
    st.subheader("📊 Informations Modèle")
    st.metric("Agents entraînés", len(label_encoder.classes_))
    st.metric("Features extraites", model.n_features_in_)
    st.metric("Type de modèle", "SVM RBF")

# ========================================
# ZONE PRINCIPALE
# ========================================

# Affichage agent sélectionné
st.markdown(f"""
<div class="agent-card">
    <h2> Agent Sélectionné: {selected_agent}</h2>
    <p> Fichier: {audio_files_list[selected_audio_idx]['display']}</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs([" Prédiction Temps Réel", " Analyse Complète", " Statistiques"])

# ========================================
# TAB 1: PRÉDICTION TEMPS RÉEL
# ========================================

with tab1:
    st.header(" Simulation Police Radio - Temps Réel")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎵 Lecture Audio")
        
        # Charger audio
        audio_full, sr = librosa.load(str(selected_audio_path), sr=SAMPLE_RATE)
        duration = len(audio_full) / sr
        
        st.audio(str(selected_audio_path))
        st.info(f"⏱️ Durée: {duration:.2f}s | 🔊 Sample Rate: {sr} Hz")
        
        # Bouton de démarrage
        if st.button("🚀 LANCER L'IDENTIFICATION", type="primary", use_container_width=True):
            
            # Conteneurs pour affichage dynamique
            status_container = st.empty()
            progress_bar = st.progress(0)
            prediction_container = st.empty()
            chart_container = st.empty()
            
            # Segmentation de l'audio
            window_samples = int(window_size * sr)
            hop_samples = window_samples // 2
            
            num_segments = max(1, int((len(audio_full) - window_samples) / hop_samples) + 1)
            
            predictions_history = []
            confidence_history = []
            time_stamps = []
            
            for i in range(num_segments):
                start = i * hop_samples
                end = start + window_samples
                
                if end > len(audio_full):
                    end = len(audio_full)
                    start = max(0, end - window_samples)
                
                audio_chunk = audio_full[start:end]
                
                # Prédiction
                pred_agent, confidence, top5 = predict_realtime(audio_chunk, sr, snr_db)
                
                if pred_agent:
                    predictions_history.append(pred_agent)
                    confidence_history.append(confidence)
                    time_stamps.append(start / sr)
                    
                    # Mise à jour barre de progression
                    progress = (i + 1) / num_segments
                    progress_bar.progress(progress)
                    
                    # Status
                    status_container.markdown(f"""
                    **⏱️ Temps: {start/sr:.2f}s / {duration:.2f}s**  
                    **🎯 Segment {i+1}/{num_segments}**
                    """)
                    
                    # Résultat prédiction
                    is_correct = pred_agent == selected_agent
                    status_class = "prediction-correct" if is_correct else "prediction-wrong"
                    status_text = "✅ CORRECT" if is_correct else "❌ ERREUR"
                    
                    confidence_class = ("confidence-high" if confidence >= 80 else
                                      "confidence-medium" if confidence >= 60 else
                                      "confidence-low")
                    
                    prediction_container.markdown(f"""
                    <div class="{status_class}">
                        {status_text}<br>
                        Prédit: {pred_agent}<br>
                        <span class="{confidence_class}">Confiance: {confidence:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Graphique temps réel
                    fig = go.Figure()
                    
                    # Ligne de confiance
                    fig.add_trace(go.Scatter(
                        x=time_stamps,
                        y=confidence_history,
                        mode='lines+markers',
                        name='Confiance',
                        line=dict(color='blue', width=3),
                        marker=dict(size=8)
                    ))
                    
                    fig.update_layout(
                        title="📈 Évolution Confiance en Temps Réel",
                        xaxis_title="Temps (s)",
                        yaxis_title="Confiance (%)",
                        yaxis=dict(range=[0, 100]),
                        height=300
                    )
                    
                    chart_container.plotly_chart(fig, use_container_width=True)
                    
                    time.sleep(0.1)  # Simulation temps réel
            
            # Résumé final
            st.success("✅ Analyse terminée!")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                accuracy = sum([1 for p in predictions_history if p == selected_agent]) / len(predictions_history) * 100
                st.metric("🎯 Accuracy", f"{accuracy:.1f}%")
            
            with col_b:
                avg_confidence = np.mean(confidence_history)
                st.metric("📊 Confiance Moyenne", f"{avg_confidence:.1f}%")
            
            with col_c:
                st.metric("🔢 Segments Analysés", len(predictions_history))
    
    with col2:
        st.subheader("📊 Top 5 Prédictions (Temps Réel)")
        
        if 'top5' in locals() and top5:
            # Graphique barres
            agents = [a for a, _ in top5]
            scores = [s for _, s in top5]
            colors = ['green' if agents[0] == selected_agent else 'red'] + ['steelblue']*4
            
            fig_bar = go.Figure(go.Bar(
                y=agents,
                x=scores,
                orientation='h',
                marker=dict(color=colors),
                text=[f"{s:.1f}%" for s in scores],
                textposition='auto'
            ))
            
            fig_bar.update_layout(
                title="Top 5 Agents",
                xaxis_title="Confiance (%)",
                yaxis_title="Agent ID",
                height=400
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)

# ========================================
# TAB 2: ANALYSE COMPLÈTE
# ========================================

with tab2:
    st.header("🔬 Analyse Complète du Signal")
    
    if st.button("📊 Générer Analyse Complète", use_container_width=True):
        
        with st.spinner("Analyse en cours..."):
            
            # Charger audio
            audio, sr = librosa.load(str(selected_audio_path), sr=SAMPLE_RATE, duration=5)
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
            
            # Simuler radio
            audio_radio = simulate_police_radio(audio_trimmed, sr, snr_db)
            
            # Prédiction
            pred_agent, confidence, top5 = predict_realtime(audio_radio, sr, snr_db)
            
            # Affichage résultat
            col1, col2 = st.columns(2)
            
            with col1:
                is_correct = pred_agent == selected_agent
                if is_correct:
                    st.success(f"✅ **CORRECT**\n\nAgent prédit: **{pred_agent}**\n\nConfiance: **{confidence:.1f}%**")
                else:
                    st.error(f"❌ **ERREUR**\n\nAgent réel: **{selected_agent}**\n\nAgent prédit: **{pred_agent}**\n\nConfiance: **{confidence:.1f}%**")
            
            with col2:
                st.info(f"📊 **Top 5 Prédictions:**\n\n" + 
                       "\n\n".join([f"{i+1}. **{a}** ({s:.1f}%)" for i, (a, s) in enumerate(top5)]))
            
            # Visualisations
            st.subheader("📈 Visualisations")
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Signal temporel original
            axes[0, 0].plot(audio_trimmed[:sr*3], linewidth=0.5, color='blue')
            axes[0, 0].set_title('Signal Original', fontweight='bold')
            axes[0, 0].set_xlabel('Échantillons')
            axes[0, 0].set_ylabel('Amplitude')
            axes[0, 0].grid(alpha=0.3)
            
            # Signal radio simulé
            axes[0, 1].plot(audio_radio[:sr*3], linewidth=0.5, color='red')
            axes[0, 1].set_title(f'Signal Radio (SNR {snr_db} dB)', fontweight='bold')
            axes[0, 1].set_xlabel('Échantillons')
            axes[0, 1].set_ylabel('Amplitude')
            axes[0, 1].grid(alpha=0.3)
            
            # Spectrogramme
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_radio)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', 
                                    ax=axes[1, 0], cmap='viridis')
            axes[1, 0].set_title('Spectrogramme', fontweight='bold')
            
            # MFCC
            mfcc = librosa.feature.mfcc(y=audio_radio, sr=sr, n_mfcc=20)
            librosa.display.specshow(mfcc, sr=sr, x_axis='time', 
                                    ax=axes[1, 1], cmap='coolwarm')
            axes[1, 1].set_title('MFCC Features', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)

# ========================================
# TAB 3: STATISTIQUES
# ========================================

with tab3:
    st.header("📊 Statistiques du Modèle")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Agents Total", len(label_encoder.classes_))
        st.metric("📁 Agent Actuel", selected_agent)
    
    with col2:
        st.metric("🔢 Features", model.n_features_in_)
        st.metric("🔊 SNR Actuel", f"{snr_db} dB")
    
    with col3:
        st.metric("⏱️ Fenêtre Analyse", f"{window_size}s")
        st.metric("📊 Type Modèle", "SVM RBF")
    
    st.divider()
    
    st.subheader("📋 Liste des Agents")
    agents_df = pd.DataFrame({
        'Agent ID': label_encoder.classes_
    })
    st.dataframe(agents_df, use_container_width=True)

# ========================================
# FOOTER
# ========================================

st.divider()
st.markdown("""
<div style="text-align: center; color: gray;">
     Système d'Identification Police Radio | Powered by Jad Falaq
</div>
""", unsafe_allow_html=True)


