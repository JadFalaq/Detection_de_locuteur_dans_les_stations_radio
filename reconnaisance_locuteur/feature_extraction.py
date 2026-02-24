import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')

from config import DATASET_ROOT, OUTPUT_ROOT, SAMPLE_RATE

print(" TEST EXTRACTION DE FEATURES")

def extract_robust_features(audio, sr=16000):

    frame_length = int(0.025 * sr)
    hop_length = frame_length // 2
    
    # MFCC (20 coefficients)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20,
                                n_fft=frame_length, hop_length=hop_length)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    # Features spectrales
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
    
    # Agrégation (mean + std)
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

# Test sur un fichier audio
csv_path = OUTPUT_ROOT / 'selected_agents' / 'agents_list.csv'
import pandas as pd
agents_df = pd.read_csv(csv_path)
agent_id = agents_df['speaker_id'].iloc[0]

agent_path = DATASET_ROOT / agent_id
for video_dir in agent_path.iterdir():
    if video_dir.is_dir():
        wav_files = list(video_dir.glob('*.wav'))
        if wav_files:
            audio_file = wav_files[0]
            break

audio, sr = librosa.load(str(audio_file), sr=SAMPLE_RATE, duration=3)
test_features = extract_robust_features(audio, sr)

print(f" Extraction validée: {len(test_features)} features par échantillon")
