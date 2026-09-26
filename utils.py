# Utilities for Instruction Tuning (SFT).

import os
import platform
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from .config import Config


console = Console()

IGNORE_INDEX = -100

DEFAULT_SAMPLE_PROMPTS: list[str] = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about programming.",
    "How does photosynthesis work?",
]

TOKEN_COLORS = {
    "<|endoftext|>": "bold red",
    "<|pad|>": "bold magenta",
    "<|user|>": "bold blue",
    "<|assistant|>": "bold green",
    "<|system|>": "bold yellow",
}


def _colorize_tokens(text: str) -> str:
    "colorer les tokens spéciaux pour l'affichage dans la console"
    text = escape (text)
    for token, color in TOKEN_COLORS.items():
        text = text.replace(token, f"[{color}]{token}[/{color}]")
    return text



def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_attn_implementation() -> str:
    if platform.machine() != "x86_64":
        return "sdpa"
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"
    

def load_model(config : Config , device : torch.device) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    """Charge le modèle et le tokenizer depuis HuggingFace Hub.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name , trust_remote_code=False)
    if tokenizer.chat_tempalte is None and config.chat_template_source :
        donor = AutoTokenizer.from_pretrained(config.chat_template_source , trust_remote_code=False)
        if donor.chat_template is  None:
            raise ValueError(
                f"Le tokenizer source {config.chat_template_source} n'a pas de chat_template défini.")
        tokenizer.chat_template = donor.chat_template

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        attn_implementation=get_attn_implementation(),
        dtype=torch.bfloat16 if config.bf16 else torch.float32,
        trust_remote_code=False
    )
    model.to(device)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    return model, tokenizer

@dataclass
class SFTBatch :
    """Un batch de données pour l'entraînement SFT.

    Attributes:
        input_ids (torch.Tensor): Les IDs des tokens d'entrée.
        attention_mask (torch.Tensor): Le masque d'attention pour les tokens d'entrée.
        labels (torch.Tensor): Les labels pour le calcul de la perte.
    """
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device) -> "SFTBatch":
        """Déplace le batch vers le périphérique spécifié (CPU ou GPU).

        Args:
            device (torch.device): Le périphérique cible.

        Returns:
            SFTBatch: Le batch déplacé vers le périphérique spécifié.
        """
        return SFTBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device)
        )

def compute_loss(model, batch: SFTBatch) -> torch.Tensor:
    out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
    shift_logits = out.logits[:, :-1, :].contiguous()
    shift_labels = batch.labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def _encode_row(
    messages: list[dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> dict[str, torch.Tensor] | None:

    if not messages or messages[-1]["role"] != "assistant":
        return None

    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=False
    )
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=False
    )
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(full_ids[len(prompt_ids):])

    if max_length is not None and len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        labels = labels[:max_length]

    if all(label == IGNORE_INDEX for label in labels):
        return None

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class SFTDataset(Dataset):
    """Dataset pour l'entraînement SFT.

    Chaque élément du dataset est un dictionnaire contenant :
        - input_ids : IDs des tokens d'entrée.
        - labels : Labels pour le calcul de la perte.
    """

    def __init__(self, encoded: list[dict[str, torch.Tensor]]):
        self.encoded_data: list[dict[str, torch.Tensor]] = encoded

    def __len__(self) -> int:
        return len(self.encoded_data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.encoded_data[idx]


def _collate(examples: list[dict[str, torch.Tensor]], pad_token_id: int) -> SFTBatch:
    """
    Cette fonction prend une liste d'exemples encodés et les combine en un batch unique,
    en ajoutant un padding pour que toutes les séquences aient la même longueur."""

    max_len = max(len(ex["input_ids"]) for ex in examples)
    input_ids, attention_mask, labels = [], [], []
    for ex in examples :
        ids, lbl = ex["input_ids"], ex["labels"]
        pad = max_len - len(ids)
        input_ids.append(torch.cat([ids, torch.full((pad,), pad_token_id, dtype=torch.long)]))
        attention_mask.append(torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
        labels.append(torch.cat([lbl, torch.full((pad,), IGNORE_INDEX, dtype=torch.long)]))
    return SFTBatch(
        input_ids=torch.stack(input_ids),
        attention_mask=torch.stack(attention_mask),
        labels=torch.stack(labels)
    )

def create_dataloader(cfg: Config, tokenizer: PreTrainedTokenizer) -> DataLoader:


    dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    if cfg.max_samples is not None and len(raw) > cfg.max_samples:
        raw = raw.select(range(cfg.max_samples))
    
    encoded: list[dict[str, torch.Tensor]] = []
    skipped = 0
    for example in raw:
        row = _encode_row(example["messages"], tokenizer, cfg.max_length)
        if row is None:
            skipped += 1
            continue
        encoded.append(row)

    if encoded is None :
        raise RuntimeError("pas de donnees disponibles pour l'entrainement")
    if skipped :
        console.print(f"[dim]{skipped}/{len(raw)} lignes ignorées (aucun token assistant entraînable).[/dim]")

    pad_token_id = tokenizer.pad_token_id 
    return DataLoader(
        SFTDataset(encoded),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda examples: _collate(examples, pad_token_id),
        num_workers=0,
        pin_memory=False,
    )

def generate_samples(
    model,
    tokenizer: PreTrainedTokenizer,
    cfg: Config,
    step: int,
    prompts: list[str] | None = None,
    max_new_tokens: int | None = None,
) -> None:
    was_training = model.training
    model.eval()
    new_tokens = max_new_tokens if max_new_tokens is not None else cfg.sample_max_tokens
    if prompts is None:
        prompts = DEFAULT_SAMPLE_PROMPTS

    console.rule(f"[bold yellow]Samples @ step {step}[/bold yellow]", style="yellow")

    for prompt_id , prompt in enumerate(prompts , start=1):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages , tokenize=False , add_generation_prompt=True )
        inputs = tokenizer(
            formatted ,
            return_tensors="pt",
            truncation=True ,
            max_length=cfg.sample_max_input_tokens 
        ).to(model.device())
        kwargs = dict(
            **inputs,
            max_new_tokens = new_tokens , 
            do_sample = cfg.sample_do_sample , 
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        if cfg.sample_do_sample : 
            kwargs.update (temperature= cfg.sample_temperature , top_p = cfg.sample_top_p)
        with torch.no_grad():
            out = model.generate(**kwargs)
        outputs = tokenizer.decode(out[0] , skip_special_tokens=False )
        console.print(
            Panel(
                _colorize_tokens(outputs),
                title=f"[bold cyan]Prompt {prompt_id}[/bold cyan]",
                title_align="left",
                border_style="cyan",
            )
        )

    if was_training:
        model.train()

def progress_bar() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("*"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )

def print_training_info(model, cfg: Config, total_steps: int, warmup_steps: int) -> None:
    console.print(
        Panel(
            f"[bold magenta]Model:[/bold magenta] {cfg.model_name}\n"
            f"[dim]Parameters:[/dim] {sum(p.numel() for p in model.parameters()):,}\n"
            f"[dim]Device:[/dim] {model.device}\n"
            f"[dim]Dataset:[/dim] {cfg.dataset_name} (split={cfg.dataset_split})\n"
            f"[dim]Effective batch:[/dim] {cfg.batch_size} x {cfg.gradient_accumulation_steps}"
            f" = {cfg.batch_size * cfg.gradient_accumulation_steps}\n"
            f"[dim]Steps:[/dim] {total_steps} total, {warmup_steps} warmup",
            title="[bold magenta]SFT Configuration[/bold magenta]",
            border_style="magenta",
        )
    )
    console.print(
        "[dim yellow]Note : OLMo-2 réutilise [bold]<|endoftext|>[/bold] comme BOS, EOS"
        "et UNK, donc il apparaît au début *et* à la fin de chaque conversation.[/dim yellow]"
    )

def print_epoch_header(epoch_idx: int, total_epochs: int) -> None:
    console.rule(f"[bold cyan]Epoch {epoch_idx + 1}/{total_epochs}[/bold cyan]", style="cyan")

def make_lr_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps + 1))
        remaining = total_steps - step
        return max(0.0, remaining / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)