# SFT — Instruction Tuning d'un LLM

Pipeline de **Supervised Fine-Tuning (SFT)** pour fine-tuner un modèle de base (ex : OLMo-2-1B) sur des paires (instruction, réponse).

---

## 📋 Description

Ce projet implémente l'entraînement SFT d'un LLM de base pour le transformer en modèle capable de suivre des instructions. Il s'inspire des implémentations de référence (CleanRL, HuggingFace).

**Points clés :**
- Masquage du prompt : seuls les tokens de la réponse contribuent à la loss.
- Gradient accumulation : batch effectif élevé sans saturer la mémoire GPU.
- Warmup + décroissance linéaire du learning rate.
- Génération périodique d'échantillons pour visualiser les progrès.
- Logging Weights & Biases (loss, grad_norm, learning rate).

---

## ⚙️ Installation

```bash
pip install uv
uv pip install --system torch datasets transformers pydantic rich pyyaml wandb numpy
```


###  Se logger à W&B (optionnel)

```bash
export WANDB_API_KEY="ta_cle_api"
export WANDB_PROJECT="sft_pipeline"
```

###  Lancer l'entraînement

```bash
python -m sft.train --config configs/config.yaml
```

---

## 📈 Suivi avec W&B

| Métrique | Description |
|---|---|
| `loss` | Loss moyenne sur le step |
| `grad_norm` | Norme des gradients avant clipping |
| `learning_rate` | LR actuel |
| `epoch` | Position continue dans l'époque |
| `hours` | Temps écoulé |


## 📚 Références

- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [OLMo-2](https://huggingface.co/allenai/OLMo-2-0425-1B)
- [no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
