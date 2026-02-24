# Système d’identification de locuteur en conditions radio (Police Radio)

## Description du projet

Ce projet met en place un pipeline complet de reconnaissance de locuteur pour des communications radio. Il simule un canal P25 (filtrage bande‑passante, bruit contrôlé par SNR, compression dynamique), construit un dataset d’apprentissage enrichi, entraîne un modèle SVM pour l’identification des agents, puis fournit une interface Streamlit pour la démonstration en temps réel et l’analyse détaillée des signaux.

L’objectif principal est d’évaluer la robustesse de l’identification de locuteur dans un contexte dégradé proche des communications radio réelles, tout en offrant une visualisation claire et interactive des résultats.

## Stack et outils utilisés

- Langage : Python
- Audio & DSP : librosa, scipy, soundfile
- Data : numpy, pandas
- Modélisation : scikit‑learn (SVM), joblib
- Visualisation : matplotlib, seaborn, plotly
- Interface : Streamlit
- Utilitaires : tqdm
- GPU (optionnel) : torch pour détection et configuration

## Structure du projet

- reconnaisance_locuteur/
  - explore_dataset.py
  - selection_agents.py
  - radio_simulation.py
  - dataset_construction.py
  - feature_extraction.py
  - train_model.py
  - train_model_gridsearch.py
  - visualisation_streamlit.py
  - config.py
- wav/ (dataset VoxCeleb organisé par locuteurs et vidéos)
- output/ (artefacts et résultats)
- Documentation/ (captures, démo, rapport)

## Étapes de réalisation du projet

1. Exploration et validation du dataset VoxCeleb.
2. Sélection d’un sous‑ensemble d’agents et vérification de la simulation radio.
3. Simulation du canal radio P25 avec ajout de bruit (SNR) et compression dynamique.
4. Construction d’un dataset enrichi par augmentation SNR.
5. Extraction de features robustes (MFCC, deltas, contrastes spectrals, ZCR, etc.).
6. Entraînement du modèle SVM et évaluation (accuracy, matrice de confusion, rapport).
7. Déploiement d’une interface Streamlit pour la démonstration et l’analyse.

## Exécution rapide

1. Vérifier la structure du dataset dans wav/

```
wav/
├── id10001/
│   ├── video1/
│   │   ├── 00001.wav
│   │   └── 00002.wav
│   └── video2/
└── id10002/
```

2. Explorer le dataset

```
python reconnaisance_locuteur/explore_dataset.py
```

3. Tester la simulation radio sur quelques agents

```
python reconnaisance_locuteur/selection_agents.py
```

4. Simuler le canal radio pour tous les agents

```
python reconnaisance_locuteur/radio_simulation.py
```

5. Construire le dataset complet

```
python reconnaisance_locuteur/dataset_construction.py
```

6. Entraîner le modèle

```
python reconnaisance_locuteur/train_model.py
```

7. Lancer l’interface Streamlit

```
streamlit run reconnaisance_locuteur/visualisation_streamlit.py
```

## Résultats et artefacts générés

- output/selected_agents/ : liste des agents sélectionnés
- output/radio_simulated/ : audios simulés (canal radio)
- output/features/ : dataset final (CSV) et métadonnées
- output/models/ : modèle SVM, scaler, label encoder, figures d’évaluation
- Documentation/ : captures d’écran, démo vidéo, rapport PDF

## Notes

- Les paramètres audio, SNR, sélection d’agents et chemins sont centralisés dans config.py.
- Le script train_model_gridsearch.py permet d’optimiser les hyperparamètres du SVM.
- La démonstration Streamlit propose une identification segmentée en temps réel et une analyse détaillée (spectrogramme, MFCC, etc.).
