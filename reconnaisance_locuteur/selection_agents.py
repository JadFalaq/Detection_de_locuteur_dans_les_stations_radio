import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config import DATASET_ROOT, OUTPUT_ROOT, SAMPLE_RATE, LOW_CUTOFF, HIGH_CUTOFF

print(" TEST SIMULATION RADIO")

def simulate_police_radio(audio, sr=16000, snr_db=10):

    # Filtrage bande passante 300-3000 Hz
    sos = signal.butter(8, [300, 3000], 'bandpass', fs=sr, output='sos')
    audio_filtered = signal.sosfilt(sos, audio)
    
    # Ajout de bruit (interférences radio)
    noise = np.random.normal(0, 1, len(audio_filtered))
    signal_power = np.mean(audio_filtered ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    noise_factor = np.sqrt(signal_power / (10 ** (snr_db/10) * noise_power))
    audio_noisy = audio_filtered + noise_factor * noise
    
    # Compression dynamique (AGC)
    audio_compressed = np.tanh(audio_noisy * 2.5) / 2.5
    
    # Normalisation
    return librosa.util.normalize(audio_compressed)

# Charger agents sélectionnés
csv_path = OUTPUT_ROOT / 'selected_agents' / 'agents_list.csv'
agents_df = pd.read_csv(csv_path)
selected_agents = agents_df['speaker_id'].tolist()

print(f"\n🎵 Test de simulation radio sur 3 agents...\n")

fig, axes = plt.subplots(3, 2, figsize=(15, 10))

for idx, agent_id in enumerate(selected_agents[:3]):
    agent_path = DATASET_ROOT / agent_id
    
    # Trouver premier fichier audio
    audio_file = None
    for video_dir in agent_path.iterdir():
        if video_dir.is_dir():
            wav_files = list(video_dir.glob('*.wav'))
            if wav_files:
                audio_file = wav_files[0]
                break
    
    # Charger audio
    audio, sr = librosa.load(str(audio_file), sr=16000, duration=3)
    audio_radio = simulate_police_radio(audio, sr)
    
    # Signal temporel original
    axes[idx, 0].plot(audio, linewidth=0.5, color='blue')
    axes[idx, 0].set_title(f'Agent {agent_id} - Original', fontsize=11)
    axes[idx, 0].set_ylabel('Amplitude')
    axes[idx, 0].grid(alpha=0.3)
    
    # Signal radio
    axes[idx, 1].plot(audio_radio, linewidth=0.5, color='red')
    axes[idx, 1].set_title(f'Agent {agent_id} - Radio P25', fontsize=11)
    axes[idx, 1].set_ylabel('Amplitude')
    axes[idx, 1].grid(alpha=0.3)

axes[-1, 0].set_xlabel('Échantillons')
axes[-1, 1].set_xlabel('Échantillons')
plt.tight_layout()

output_path = OUTPUT_ROOT / 'radio_simulation_test_agents.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f" Graphique sauvegardé: {output_path}")
plt.show()

print(" Simulation radio validée!")
