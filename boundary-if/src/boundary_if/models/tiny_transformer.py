from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from boundary_if.models.m1_tfidf_logreg import binary_classification_metrics

MODEL_ID = "M3"
MODEL_NAME = "M3_prompt_bidirectional_encoder_pooling_mlp"

PREDICTION_METADATA_COLUMNS = [
    "prompt_id",
    "split",
    "label",
    "raw_prompt_bpe_token_count_full",
    "raw_prompt_bpe_token_count",
    "raw_prompt_bpe_truncated",
    "num_constraints",
    "cluster",
    "length_bin",
]


@dataclass(frozen=True)
class M3TinyTransformerConfig:
    vocab_size: int = 8000
    max_length: int = 2048
    hidden_size: int = 128
    layers: int = 4
    heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    pooling: str = "mean"
    classifier_hidden_size: int = 128
    pad_token_id: int | None = None
    threshold: float = 0.5

    @property
    def resolved_pad_token_id(self) -> int:
        return self.vocab_size if self.pad_token_id is None else self.pad_token_id

    @property
    def embedding_vocab_size(self) -> int:
        return max(self.vocab_size, self.resolved_pad_token_id + 1)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resolved_pad_token_id"] = self.resolved_pad_token_id
        payload["embedding_vocab_size"] = self.embedding_vocab_size
        payload["pipeline"] = "prompt->bidirectional_encoder->pooling->MLP->P_T(pass)"
        return payload


class PromptBidirectionalEncoder(nn.Module):
    def __init__(self, config: M3TinyTransformerConfig) -> None:
        super().__init__()
        if config.hidden_size % config.heads != 0:
            raise ValueError("hidden_size must be divisible by heads.")

        self.config = config
        self.token_embedding = nn.Embedding(
            config.embedding_vocab_size,
            config.hidden_size,
            padding_idx=config.resolved_pad_token_id,
        )
        self.position_embedding = nn.Embedding(config.max_length + 1, config.hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids and attention_mask must both be rank-2 tensors.")
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have the same shape.")
        if input_ids.shape[1] > self.config.max_length:
            raise ValueError(
                f"Sequence length {input_ids.shape[1]} exceeds max_length={self.config.max_length}."
            )

        batch_size, seq_len = input_ids.shape
        token_embeddings = self.token_embedding(input_ids)
        cls_tokens = self.cls_token.expand(batch_size, 1, -1)
        hidden_states = torch.cat([cls_tokens, token_embeddings], dim=1)

        positions = torch.arange(seq_len + 1, device=input_ids.device).unsqueeze(0)
        hidden_states = hidden_states + self.position_embedding(positions)

        cls_mask = torch.ones(
            batch_size,
            1,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        key_padding_mask = ~full_attention_mask.bool()
        encoded = self.encoder(hidden_states, src_key_padding_mask=key_padding_mask)
        return encoded[:, 0], encoded[:, 1:], attention_mask


class PromptPooling(nn.Module):
    def __init__(self, config: M3TinyTransformerConfig) -> None:
        super().__init__()
        if config.pooling not in {"cls", "mean"}:
            raise ValueError(f"Unsupported pooling mode: {config.pooling!r}")
        self.config = config
        self.output_size = config.hidden_size

    def forward(
        self,
        cls_state: torch.Tensor,
        token_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_mask = attention_mask.unsqueeze(-1).to(token_states.dtype)
        mean_state = (token_states * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0)

        if self.config.pooling == "cls":
            return cls_state
        return mean_state


class PassProbabilityMLP(nn.Module):
    def __init__(self, input_size: int, config: M3TinyTransformerConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, config.classifier_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_size, 1),
        )

    def forward(self, pooled_prompt: torch.Tensor) -> torch.Tensor:
        return self.layers(pooled_prompt).squeeze(-1)


class M3TinyTransformer(nn.Module):
    """prompt -> bidirectional encoder -> pooling -> MLP -> P_T(pass)."""

    def __init__(self, config: M3TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.prompt_encoder = PromptBidirectionalEncoder(config)
        self.pooling = PromptPooling(config)
        self.pass_head = PassProbabilityMLP(self.pooling.output_size, config)

    def encode_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.prompt_encoder(input_ids=input_ids, attention_mask=attention_mask)

    def pool_prompt(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls_state, token_states, token_mask = self.encode_prompt(input_ids, attention_mask)
        return self.pooling(cls_state, token_states, token_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled_prompt = self.pool_prompt(input_ids=input_ids, attention_mask=attention_mask)
        return self.pass_head(pooled_prompt)

    def predict_pass_probability(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(input_ids=input_ids, attention_mask=attention_mask))


def normalize_token_ids(value: Any, *, max_length: int) -> list[int]:
    if isinstance(value, np.ndarray):
        values = value.tolist()
    else:
        values = list(value)
    return [int(token_id) for token_id in values[:max_length]]


class M3PromptDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, *, max_length: int) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        return {
            "input_ids": normalize_token_ids(row["input_ids"], max_length=self.max_length),
            "label": int(row["label"]),
            "metadata": {column: row[column] for column in PREDICTION_METADATA_COLUMNS},
        }


def m3_collate_fn(
    samples: list[dict[str, Any]],
    *,
    pad_token_id: int,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    max_batch_length = max(len(sample["input_ids"]) for sample in samples)
    max_batch_length = max(max_batch_length, 1)
    input_ids = torch.full(
        (len(samples), max_batch_length),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros((len(samples), max_batch_length), dtype=torch.long)
    labels = torch.tensor([sample["label"] for sample in samples], dtype=torch.float32)
    metadata: dict[str, list[Any]] = {column: [] for column in PREDICTION_METADATA_COLUMNS}

    for row_index, sample in enumerate(samples):
        token_ids = sample["input_ids"]
        if token_ids:
            input_ids[row_index, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
            attention_mask[row_index, : len(token_ids)] = 1
        for column in PREDICTION_METADATA_COLUMNS:
            metadata[column].append(sample["metadata"][column])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "metadata": metadata,
    }


def make_m3_dataloader(
    frame: pd.DataFrame,
    config: M3TinyTransformerConfig,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    generator: torch.Generator | None = None,
) -> DataLoader:
    dataset = M3PromptDataset(frame, max_length=config.max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        collate_fn=lambda samples: m3_collate_fn(
            samples,
            pad_token_id=config.resolved_pad_token_id,
        ),
    )


def compute_pos_weight(labels: pd.Series) -> float:
    positive_count = int(labels.astype(int).sum())
    negative_count = int(len(labels) - positive_count)
    if positive_count == 0:
        return 1.0
    return negative_count / positive_count


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_m3_epoch(
    model: M3TinyTransformer,
    dataloader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_norm: float,
    scaler: torch.cuda.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

        if scaler is None:
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        if scheduler is not None:
            scheduler.step()

        batch_size = int(labels.numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
    return {"loss": total_loss / max(total_examples, 1)}


@torch.no_grad()
def predict_m3(
    model: M3TinyTransformer,
    dataloader: DataLoader,
    *,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        probabilities = (
            model.predict_pass_probability(input_ids=input_ids, attention_mask=attention_mask)
            .detach()
            .cpu()
            .numpy()
        )
        labels = batch["labels"].detach().cpu().numpy().astype(int)
        metadata = batch["metadata"]
        for row_index, probability in enumerate(probabilities):
            row = {column: metadata[column][row_index] for column in PREDICTION_METADATA_COLUMNS}
            row["label"] = int(labels[row_index])
            row["m3_pred_proba"] = float(probability)
            row["m3_pred_label"] = int(probability >= model.config.threshold)
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_m3_predictions(predictions: pd.DataFrame, threshold: float) -> dict[str, Any]:
    labels = predictions["label"].astype(int).to_numpy()
    probabilities = predictions["m3_pred_proba"].astype(float).to_numpy()
    return binary_classification_metrics(labels, probabilities, threshold=threshold)


def evaluate_m3_by_split(predictions: pd.DataFrame, threshold: float) -> dict[str, dict[str, Any]]:
    split_metrics: dict[str, dict[str, Any]] = {}
    for split, split_frame in predictions.groupby("split", sort=True):
        split_metrics[str(split)] = evaluate_m3_predictions(split_frame, threshold)
    return split_metrics
