import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from scipy import signal

# Import de la configuration
from config import (
    DATASET_ROOT, OUTPUT_ROOT, SAMPLE_RATE, DURATION,
    LOW_CUTOFF, HIGH_CUTOFF, SNR_DB, 
    THRESHOLD_DB, RATIO, ATTACK_MS, RELEASE_MS
)


print(" SIMULATION CANAL RADIO P25")

def simulate_police_radio(audio, sr=16000, 
                         low_cutoff=300, high_cutoff=3000,
                         snr_db=15, 
                         threshold_db=-20, ratio=4):

    # 1. Filtrage bande passante (300-3000 Hz pour P25)
    sos = signal.butter(8, [low_cutoff, high_cutoff], 'bandpass', 
                       fs=sr, output='sos')
    audio_filtered = signal.sosfilt(sos, audio)
    
    # 2. Ajout de bruit (interférences radio)
    noise = np.random.normal(0, 1, len(audio_filtered))
    signal_power = np.mean(audio_filtered ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    noise_factor = np.sqrt(signal_power / (10 ** (snr_db/10) * noise_power))
    audio_noisy = audio_filtered + noise_factor * noise
    
    # 3. Compression dynamique (AGC - Automatic Gain Control)
    audio_compressed = np.tanh(audio_noisy * 2.5) / 2.5
    
    # 4. Normalisation finale
    audio_radio = librosa.util.normalize(audio_compressed)
    
    return audio_radio


def load_selected_agents():

    csv_path = OUTPUT_ROOT / 'selected_agents' / 'agents_list.csv'
    
    if not csv_path.exists():
        print(f" ERREUR: Fichier agents non trouvé: {csv_path}")
        print("   → Exécutez d'abord: python 02_selection_agents.py")
        exit(1)
    
    agents_df = pd.read_csv(csv_path)
    selected_agents = agents_df['speaker_id'].tolist()
    
    print(f"{len(selected_agents)} agents chargés depuis {csv_path.name}\n")
    return selected_agents, agents_df


def test_radio_simulation(selected_agents, num_test=3):

    print(f" Test de simulation radio sur {num_test} agents...\n")
    
    fig, axes = plt.subplots(num_test, 2, figsize=(16, 3*num_test))
    
    # Si un seul agent, axes n'est pas un array 2D
    if num_test == 1:
        axes = axes.reshape(1, -1)
    
    for idx, agent_id in enumerate(selected_agents[:num_test]):
        agent_path = DATASET_ROOT / agent_id
        
        # Trouver le premier fichier audio
        audio_file = None
        for video_dir in agent_path.iterdir():
            if video_dir.is_dir():
                wav_files = list(video_dir.glob('*.wav'))
                if wav_files:
                    audio_file = wav_files[0]
                    break
        
        if audio_file is None:
            print(f" Aucun fichier audio pour {agent_id}")
            continue
        
        # Charger audio
        audio, sr = librosa.load(str(audio_file), sr=SAMPLE_RATE, duration=DURATION)
        
        # Appliquer simulation radio
        audio_radio = simulate_police_radio(
            audio, sr=sr,
            low_cutoff=LOW_CUTOFF,
            high_cutoff=HIGH_CUTOFF,
            snr_db=SNR_DB
        )
        
        # Visualisation - Signal original
        axes[idx, 0].plot(audio, linewidth=0.5, color='steelblue', alpha=0.8)
        axes[idx, 0].set_title(f'Agent {agent_id} - Signal Original', 
                              fontsize=12, fontweight='bold')
        axes[idx, 0].set_ylabel('Amplitude', fontsize=10)
        axes[idx, 0].grid(alpha=0.3, linestyle='--')
        axes[idx, 0].set_ylim(-1, 1)
        
        # Visualisation - Signal radio
        axes[idx, 1].plot(audio_radio, linewidth=0.5, color='orangered', alpha=0.8)
        axes[idx, 1].set_title(f'Agent {agent_id} - Canal Radio P25', 
                              fontsize=12, fontweight='bold')
        axes[idx, 1].set_ylabel('Amplitude', fontsize=10)
        axes[idx, 1].grid(alpha=0.3, linestyle='--')
        axes[idx, 1].set_ylim(-1, 1)
        
        print(f"    Agent {agent_id}: Simulation OK")
    
    axes[-1, 0].set_xlabel('Échantillons', fontsize=10, fontweight='bold')
    axes[-1, 1].set_xlabel('Échantillons', fontsize=10, fontweight='bold')
    
    plt.suptitle('Comparaison Signal Original vs Canal Radio P25', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Sauvegarder
    output_path = OUTPUT_ROOT / 'radio_simulation_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n Graphique sauvegardé: {output_path}")
    
    plt.show()
    plt.close()


def simulate_all_agents(selected_agents, agents_df, 
                       samples_per_agent=300,
                       snr_values=[10, 15]):  # ← AJOUT MULTIPLE SNR
    """
    Applique simulation avec PLUSIEURS SNR (augmentation données)
    """
    print(" SIMULATION RADIO - AUGMENTATION DONNÉES")
    print(f"   SNR utilisés: {snr_values} dB")
    print(f"   → {len(snr_values)}× augmentation\n")
    
    output_dir = OUTPUT_ROOT / 'radio_simulated'
    
    total_processed = 0
    total_failed = 0
    
    for agent_id in tqdm(selected_agents, desc="Agents"):
        agent_path = DATASET_ROOT / agent_id
        agent_output = output_dir / agent_id
        agent_output.mkdir(exist_ok=True)
        
        # Collecter fichiers
        all_audio_files = []
        for video_dir in agent_path.iterdir():
            if video_dir.is_dir():
                all_audio_files.extend(list(video_dir.glob('*.wav')))
        
        all_audio_files = all_audio_files[:samples_per_agent]
        
        # Traiter avec PLUSIEURS SNR
        for idx, audio_file in enumerate(all_audio_files):
            try:
                audio, sr = librosa.load(str(audio_file), 
                                        sr=SAMPLE_RATE, 
                                        duration=DURATION)
                
                # CRÉER UNE VERSION PAR SNR ← MODIFICATION CLÉE
                for snr_db in snr_values:
                    audio_radio = simulate_police_radio(
                        audio, sr=sr,
                        low_cutoff=LOW_CUTOFF,
                        high_cutoff=HIGH_CUTOFF,
                        snr_db=snr_db  # ← Variable
                    )
                    
                    # Sauvegarder avec SNR dans le nom
                    output_file = agent_output / f"{agent_id}_radio_snr{snr_db}_{idx:04d}.wav"
                    sf.write(output_file, audio_radio, SAMPLE_RATE)
                    
                    total_processed += 1
                
            except Exception as e:
                total_failed += 1
                continue
    
    print(f"\n Simulation terminée:")
    print(f"   ├─ Fichiers créés: {total_processed}")
    print(f"   ├─ Échecs: {total_failed}")
    print(f"   └─ Augmentation: {len(snr_values)}×")



# ========================================
# EXÉCUTION PRINCIPALE
# ========================================

if __name__ == "__main__":
    # Charger agents sélectionnés
    selected_agents, agents_df = load_selected_agents()
    
    # Test sur quelques agents
    test_radio_simulation(selected_agents, num_test=3)
    

    response = input("Continuer avec la simulation complète ? (o/n): ")
    
    if response.lower() in ['o', 'oui', 'y', 'yes']:
        # Simulation complète
        simulate_all_agents(selected_agents, agents_df, 
                       samples_per_agent=150,  
                       snr_values=[10, 15])    
        

        print(" SIMULATION RADIO TERMINÉE")

        print(f"Fichiers dans: {OUTPUT_ROOT / 'radio_simulated'}")
        print("\n Prochaine étape: 04_feature_extraction.py")

    else:
        print("\n Simulation complète annulée")
