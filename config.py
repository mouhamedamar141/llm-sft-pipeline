import yaml  # Pour lire les fichiers de configuration au format YAML
from pydantic import BaseModel  # Pour la validation automatique des types et valeurs par défaut


class Config(BaseModel):
    """Configuration complète d'un entraînement SFT.

    Attributs :
        model_name : Identifiant HuggingFace du modèle de base à fine-tuner.
            On veut un modèle de base SANS chat template instruction-tuned,
            par exemple OLMo-2-0425-1B.
        chat_template_source : Tokenizer depuis lequel copier le ``chat_template``
            quand le tokenizer de base n'en a pas. Mettre ``null`` pour désactiver.
        dataset_name, dataset_split : Identifiant HuggingFace du dataset et split.
        max_samples : Limite optionnelle du nombre de lignes d'entraînement
            (utile pour des expériences rapides).
        max_length : Longueur totale maximale d'une séquence (prompt + réponse).

        lr, num_epochs, batch_size, gradient_accumulation_steps, warmup_ratio,
        weight_decay, max_grad_norm : hyperparamètres standards d'AdamW pour le SFT.

        bf16, gradient_checkpointing, model_device_id : matériel / mémoire.

        sample_* : paramètres de génération en boucle pour le logging.
            ``sample_every`` se déclenche au step 0 (modèle de base) puis tous
            les ``sample_every`` steps d'optimiseur, ce qui permet à W&B
            d'afficher le modèle de base à côté du modèle SFT en cours.

        wandb_project, wandb_run_name : logging Weights & Biases.
    """

    # ------------------------------------------------------------------
    # Modèle
    # ------------------------------------------------------------------
    # Identifiant HuggingFace du modèle de base à fine-tuner.
    # OLMo-2-0425-1B est un modèle open-source d'AI2 (1 milliard de paramètres),
    # version "base" (non instruct), idéal pour être fine-tuné en SFT.
    model_name: str = "allenai/OLMo-2-0425-1B"

    # Tokenizer depuis lequel copier le chat_template.
    # Le modèle de base n'a pas de chat_template (format de conversation).
    # On le copie donc depuis la version déjà SFT du même modèle.
    # Mettre None pour désactiver cette copie.
    chat_template_source: str | None = "allenai/OLMo-2-0425-1B-SFT"

    # ------------------------------------------------------------------
    # Données
    # ------------------------------------------------------------------
    # Dataset HuggingFace contenant des paires (instruction, réponse).
    # "no_robots" est un dataset de démonstrations humaines de haute qualité,
    # populaire pour le SFT.
    dataset_name: str = "HuggingFaceH4/no_robots"

    # Split du dataset à utiliser : "train", "validation" ou "test".
    dataset_split: str = "train"

    # Limite optionnelle du nombre d'exemples d'entraînement.
    # None = utiliser tout le dataset. Utile pour des tests rapides.
    max_samples: int | None = None

    # Longueur maximale (en tokens) d'une séquence prompt + réponse.
    # Au-delà, la séquence est tronquée ou filtrée.
    max_length: int = 2048

    # ------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------
    # Learning rate. Très bas (5e-6) car le modèle est déjà pré-entraîné :
    # un LR trop élevé casserait ses connaissances générales
    # (catastrophic forgetting).
    lr: float = 5.0e-6

    # Nombre de passages complets sur le dataset.
    # 1 à 3 suffisent généralement en SFT ; au-delà, risque d'overfitting.
    num_epochs: int = 3

    # Nombre d'exemples traités par GPU à chaque forward/backward.
    # Petit (4) pour tenir en mémoire.
    batch_size: int = 4

    # Nombre de batches accumulés avant de faire un optimizer.step().
    # Simule un batch effectif de batch_size × gradient_accumulation_steps
    # = 4 × 8 = 32, sans avoir à tout charger en mémoire en même temps.
    gradient_accumulation_steps: int = 8

    # Proportion des steps totaux consacrés au warmup.
    # Pendant les 10% premiers steps, le LR monte linéairement de 0 à `lr`.
    # Stabilise le début de l'entraînement.
    warmup_ratio: float = 0.1

    # Régularisation L2 (weight decay). Souvent 0 en SFT.
    weight_decay: float = 0.0

    # Norme maximale des gradients (gradient clipping).
    # Évite les explosions de gradient en bornant leur norme à 1.0.
    max_grad_norm: float = 1.0

    # Seed pour la reproductibilité (poids initiaux, shuffling, etc.).
    seed: int = 42

    # ------------------------------------------------------------------
    # Matériel
    # ------------------------------------------------------------------
    # Utiliser bfloat16 (précision réduite, ~2× plus rapide et moins de mémoire).
    # bf16 est préféré à fp16 car il a la même plage dynamique que fp32.
    bf16: bool = True

    # Activer le gradient checkpointing : recalcule les activations au backward
    # au lieu de les stocker, économisant ~50% de mémoire (~20% plus lent).
    # Indispensable pour fine-tuner un 1B sur un GPU consumer.
    gradient_checkpointing: bool = True

    # Identifiant du GPU à utiliser (0 = premier GPU).
    model_device_id: int = 0

    # ------------------------------------------------------------------
    # Génération en boucle (logging)
    # ------------------------------------------------------------------
    # Fréquence de génération d'échantillons (en steps d'optimiseur).
    # Se déclenche aussi au step 0 pour capturer le modèle de base.
    sample_every: int = 50

    # Nombre maximum de tokens générés pour chaque échantillon.
    sample_max_tokens: int = 128

    # Nombre maximum de tokens en entrée (prompt) pour la génération.
    sample_max_input_tokens: int = 512

    # Température d'échantillonnage : 0.7 = un peu de créativité.
    # (1.0 = neutre, 0.0 = déterministe)
    sample_temperature: float = 0.7

    # Nucleus sampling : ne garde que les tokens dont la probabilité
    # cumulée atteint 0.9, éliminant les tokens trop improbables.
    sample_top_p: float = 0.9

    # True = échantillonner (avec température et top-p) ;
    # False = décodage greedy (toujours le token le plus probable).
    sample_do_sample: bool = True

    # ------------------------------------------------------------------
    # Logging Weights & Biases
    # ------------------------------------------------------------------
    # Nom du projet W&B (regroupe plusieurs runs).
    # None = pas de logging W&B (ou logging local uniquement).
    wandb_project: str | None = None

    # Nom du run W&B. None = W&B en génère un automatiquement.
    wandb_run_name: str | None = None


def load_config(config_path: str) -> Config:
    """Charge la configuration depuis un fichier YAML.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)