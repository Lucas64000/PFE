# PFE - Reconnaissance d'entités nommées dans le biomédical français

Pipeline complet pour l'entraînement et l'évaluation de modèles de reconnaissance d'entités nommées (NER) dans le domaine biomédical, basé sur le jeu de données **QUAERO**.

## Fonctionnalités principales

- Entraînement d’un modèle Hugging Face ou depuis un répertoire local.
- Gestion flexible des paramètres via un fichier `config.yaml` ainsi que des arguments prioritaires sur le fichier config passables en paramètres directement.
- Support de différents jeux de données (`emea`, `medline`).
- Évaluation des modèles avec matrice de confusion et rapport sklearn.

## Arborescence du code 

```
├── train.py       # Script pour l'entraînement du modèle
├── eval.py        # Script pour l'évaluation du modèle
├── config.yaml    # Fichier de configuration 
├── data/          # Prétraitement des datasets
├── datasets/      # Contient les datasets enregistrés pour éviter de les recharger depuis HF
├── model/         # Fonctions pour charger les modèles
├── models/        # Contient les modèles enregistrés
├── plots/         # Plot la répartition des classes des données et les résultats des modèles
├── tokenizer/     # Gestion des tokenizers
├── training/      # Entraînement et évaluation du modèle
└── logs/          # Logs tensorboard contenant les métriques des modèles lors de l'entraînement 
```

## Utilisation

### Entraînement

```bash
python train.py \
  --config config.yaml \
  --split_names emea medline \
  --epoch 3 \
  --evaluate
```

- `--config` : chemin vers le fichier de configuration YAML.
- `--split_names` : jeux de données à inclure (`emea`, `medline`, etc.).
- `--epoch` : nombre d’époques.
- `--evaluate` : déclenche l’évaluation après l’entraînement.
- `--from_dir` : charger le checkpoint du modèle (indiqué dans le fichier config) depuis un répertoire local.

### Évaluation seule

```bash
python eval.py \
  --config config.yaml \
  --normalize
```

- `--normalize` : normalise la matrice de confusion.

Les arguments ont déjà été prédéfinis par défaut, ils sont tous optionnels. 

## A propos du jeu de données

Le script s’appuie exclusivement sur le corpus **DrBenchmark QUAERO**.
Il est possible de charger son propre corpus en créant sa propre classe héritée de BaseDataset.

## Informations supplémentaires

Les fonctions pour GliNER ont été implémentées mais elles ne sont pas utilisées. GliNER n'est donc pas directement utilisable.
Vous trouverez cependant les démos d'utilisation dans les notebooks. 
