import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

from config import DATASET_ROOT, OUTPUT_ROOT, SAMPLE_RATE

print(" CONSTRUCTION DU DATASET COMPLET")

def simulate_police_radio(audio, sr=16000, snr_db=10):

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

def build_complete_dataset(dataset_path, speaker_ids, 
                          max_samples_per_speaker=150,
                          snr_range=[10, 15]):

    print(f"\n Agents: {len(speaker_ids)}")
    print(f" Max échantillons/agent: {max_samples_per_speaker}")
    print(f" Niveaux de bruit (SNR): {snr_range} dB\n")
    
    all_features = []
    all_labels = []
    all_metadata = []
    
    total_processed = 0
    total_skipped = 0
    
    for speaker_id in tqdm(speaker_ids, desc=" Traitement"):
        speaker_path = dataset_path / speaker_id
        sample_count = 0
        
        for video_dir in speaker_path.iterdir():
            if sample_count >= max_samples_per_speaker:
                break
            
            if not video_dir.is_dir():
                continue
            
            for audio_file in video_dir.glob('*.wav'):
                if sample_count >= max_samples_per_speaker:
                    break
                
                try:
                    audio, sr = librosa.load(str(audio_file), sr=16000, duration=5)
                    
                    if len(audio) < sr * 2:
                        total_skipped += 1
                        continue
                    
                    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
                    if len(audio_trimmed) < sr * 1.5:
                        total_skipped += 1
                        continue
                    
                    for snr_db in snr_range:
                        audio_radio = simulate_police_radio(audio_trimmed, sr, snr_db)
                        features = extract_robust_features(audio_radio, sr)
                        
                        all_features.append(features)
                        all_labels.append(speaker_id)
                        all_metadata.append({
                            'speaker_id': speaker_id,
                            'video_id': video_dir.name,
                            'file': audio_file.name,
                            'snr_db': snr_db
                        })
                    
                    sample_count += 1
                    total_processed += 1
                    
                except Exception as e:
                    total_skipped += 1
                    continue
    
    df_features = pd.DataFrame(all_features)
    df_features['speaker_id'] = all_labels
    df_metadata = pd.DataFrame(all_metadata)
    
    print(f"\n DATASET CONSTRUIT!")
    print(f"   Échantillons totaux: {len(df_features)}")
    print(f"   Agents: {df_features['speaker_id'].nunique()}")
    print(f"   Features/échantillon: {df_features.shape[1] - 1}")
    print(f"   Fichiers traités: {total_processed}")
    print(f"   Fichiers ignorés: {total_skipped}")
    
    print(f"\n Top 10 agents:")
    print(df_features['speaker_id'].value_counts().head(10).to_string())
    
    print(f"\n Distribution SNR:")
    print(df_metadata['snr_db'].value_counts().sort_index().to_string())
    
    return df_features, df_metadata

# Charger agents sélectionnés
csv_path = OUTPUT_ROOT / 'selected_agents' / 'agents_list.csv'
agents_df = pd.read_csv(csv_path)
selected_agents = agents_df['speaker_id'].tolist()

print("\n Construction du dataset (15-20 min)...\n")

dataset_df, metadata_df = build_complete_dataset(
    DATASET_ROOT, 
    selected_agents,
    max_samples_per_speaker=150,
    snr_range=[10, 15]
)

# Sauvegarder
print("\n Sauvegarde...")
output_features = OUTPUT_ROOT / 'features'
output_features.mkdir(exist_ok=True)

dataset_path = output_features / 'police_radio_dataset.csv'
metadata_path = output_features / 'dataset_metadata.csv'

dataset_df.to_csv(dataset_path, index=False)
metadata_df.to_csv(metadata_path, index=False)

print("\n Fichiers CSV créés:")
print(f"  - {dataset_path}")
print(f"  - {metadata_path}")

print(f"\n Tailles:")
print(f"  Dataset: {dataset_path.stat().st_size / 1024 / 1024:.1f} MB")
print(f"  Metadata: {metadata_path.stat().st_size / 1024 / 1024:.1f} MB")

print(" CONSTRUCTION TERMINÉE! Passez à l'entraînement")
