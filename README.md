# 🫁 Détection de pneumonie sur radiographies thoraciques

Projet académique de bout en bout (conception → déploiement) réalisé dans le cadre du master en Intelligenza Artificiale e Innovazione Digitale (curriculum biomedicale), UPO Vercelli.

**🔗 Démo live** : [https://pneumonia-xray-detection-dtabmfpomwwcjg26cbbpft.streamlit.app/#detection-de-pneumonie-sur-radiographie-thoracique]
*(Outil pédagogique — ne constitue PAS un diagnostic médical)*

---

##  Objectif

Classifier une radiographie thoracique en `NORMAL` ou `PNEUMONIA`, comme outil d'aide au pré-tri (pas de remplacement du diagnostic médical). Le projet inclut explicabilité (Grad-CAM) et évaluation clinique complète (pas seulement l'accuracy), conditions non négociables pour tout système d'IA médicale.

##  Données

- **Source** : [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kaggle)
- **Total** : 5 856 images (NORMAL / PNEUMONIA)
- **Corrections apportées au dataset original** :
  - Exclusion de l'artefact `__MACOSX/` et de la duplication `chest_xray/chest_xray/`
  - Le `val/` fourni (8 images/classe) était statistiquement inutilisable → re-split stratifié 90/10 sur train+val (4 708 train / 524 val)
  - `test/` (624 images) conservé intact, jamais utilisé avant l'évaluation finale

##  Méthodologie

1. **EDA** : tailles très variables (400–2148 × 138–2376 px), formats mixtes (L/RGB) → normalisation systématique 224×224, conversion RGB
2. **Modèle** : DenseNet121 pré-entraîné (ImageNet), choisi pour sa légèreté et sa littérature spécifique (CheXNet)
3. **Transfer learning en 2 temps** :
   - Phase 1 : classifieur seul entraîné (couches gelées), 10 epochs
   - Phase 2 : fine-tuning du dernier dense block (learning rates différenciés 1e-4 / 1e-5), 5 epochs
4. **Loss pondérée** (`CrossEntropyLoss` avec poids de classe) pour compenser le déséquilibre NORMAL/PNEUMONIA (1341 vs 3875 en train)

##  Résultats (test set, jamais vu pendant l'entraînement)

| Métrique | Valeur |
|---|---|
| Sensibilité (PNEUMONIA) | 97.18% |
| Spécificité (NORMAL) | 73.93% |
| Accuracy globale | 88% |

**Matrice de confusion** :
            Prédit NORMAL   Prédit PNEUMONIA
            Vrai NORMAL 173 61
            Vrai PNEUMONIA 11 379


**Choix du seuil de décision** : maintenu à 0.50 (défaut), pour prioriser la sensibilité — dans ce contexte de pré-tri, rater un vrai cas de pneumonie est jugé plus grave qu'une fausse alerte. Une analyse de seuils alternatifs (0.55 à 0.75) est documentée dans le notebook, montrant un compromis possible sensibilité/spécificité si un usage différent l'exigeait.

##  Explicabilité (Grad-CAM)

Chaque prédiction est accompagnée d'une carte de chaleur montrant les zones de l'image ayant motivé la décision — non négociable pour la confiance clinique dans un modèle.

- Sur les vrais positifs : le modèle cible correctement les champs pulmonaires
- Sur les faux positifs (NORMAL→PNEUMONIA) : le modèle se concentre sur les bases pulmonaires et la zone rétro-cardiaque — des zones reconnues comme les plus ambiguës en radiologie thoracique, y compris pour des lecteurs humains. Les erreurs du modèle sont donc cliniquement interprétables, pas aléatoires.

##  Limites connues

- **Domain shift train/test** : le `test/` provient probablement d'une distribution légèrement différente (autre source hospitalière), ce qui explique en partie l'écart sensibilité/spécificité observé
- **Détection hors distribution (OOD) absente** : le modèle classe n'importe quelle image en NORMAL/PNEUMONIA, y compris des images qui n'ont rien à voir avec une radiographie thoracique (testé avec une IRM cérébrale, classée PNEUMONIA à 86%). Un système clinique réel nécessiterait une étape de validation d'entrée en amont — hors scope de ce projet académique, mais identifiée comme amélioration future
- **Cadrages hétérogènes** dans le dataset source, non corrigés (limite du dataset public utilisé)

##  Stack technique (100% gratuit)

- **Entraînement** : Google Colab (GPU T4 gratuit), PyTorch, torchvision
- **Suivi/versioning** : GitHub
- **Déploiement** : Streamlit Community Cloud (gratuit, lien permanent)
- **Explicabilité** : `pytorch-grad-cam`

##  Structure du dépôt
├── app.py # Application Streamlit (démo)
├── requirements.txt # Dépendances de déploiement
├── best_model_finetuned.pth # Poids du modèle entraîné
└── README.md

##  Auteur

Projet réalisé par michel temukam, étudiant en master IA & Innovation Digitale (curriculum biomedicale), UPO Vercelli — dans une démarche d'apprentissage pratique par projet réel, encadré comme un stage.
