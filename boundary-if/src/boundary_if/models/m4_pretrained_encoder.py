from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from boundary_if.models.m1_tfidf_logreg import binary_classification_metrics
from boundary_if.models.tiny_transformer import (
    M3TinyTransformerConfig,
    PromptBidirectionalEncoder,
    PromptPooling,
    compute_pos_weight,
    set_torch_seed,
)

MODEL_ID = "M4"
MODEL_NAME = "M4_if_domain_pretrained_encoder_frozen_mlp"

STRUCTURAL_LABELS = [
    "has_length_count_constraint",
    "has_keyword_inclusion_constraint",
    "has_keyword_exclusion_constraint",
    "has_punctuation_restriction",
    "has_format_constraint",
    "has_casing_constraint",
    "has_start_end_constraint",
    "has_list_bullet_constraint",
]
NUM_CONSTRAINT_BUCKETS = ["1", "2", "3", "4+"]

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
class M4EncoderConfig:
    vocab_size: int = 8000
    max_length: int = 2048
    hidden_size: int = 128
    layers: int = 2
    heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    pooling: str = "mean"
    classifier_hidden_size: int = 128
    classifier_dropout: float = 0.1
    pad_token_id: int | None = None
    mask_token_id: int = 4
    threshold: float = 0.5

    @property
    def resolved_pad_token_id(self) -> int:
        return self.vocab_size if self.pad_token_id is None else self.pad_token_id

    @property
    def embedding_vocab_size(self) -> int:
        return max(self.vocab_size, self.resolved_pad_token_id + 1)

    def to_m3_config(self) -> M3TinyTransformerConfig:
        return M3TinyTransformerConfig(
            vocab_size=self.vocab_size,
            max_length=self.max_length,
            hidden_size=self.hidden_size,
            layers=self.layers,
            heads=self.heads,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
            pooling=self.pooling,
            classifier_hidden_size=self.classifier_hidden_size,
            pad_token_id=self.pad_token_id,
            threshold=self.threshold,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resolved_pad_token_id"] = self.resolved_pad_token_id
        payload["embedding_vocab_size"] = self.embedding_vocab_size
        payload["structural_labels"] = STRUCTURAL_LABELS
        payload["num_constraint_buckets"] = NUM_CONSTRAINT_BUCKETS
        payload["pipeline"] = (
            "IF-domain MLM+structural pretraining -> frozen mean-pooled encoder -> MLP -> P_T(pass)"
        )
        return payload


class M4PretrainingModel(nn.Module):
    def __init__(self, config: M4EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.prompt_encoder = PromptBidirectionalEncoder(config.to_m3_config())
        self.pooling = PromptPooling(config.to_m3_config())
        self.mlm_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.vocab_size),
        )
        self.structural_binary_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, len(STRUCTURAL_LABELS)),
        )
        self.num_constraints_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, len(NUM_CONSTRAINT_BUCKETS)),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cls_state, token_states, token_mask = self.prompt_encoder(input_ids, attention_mask)
        pooled = self.pooling(cls_state, token_states, token_mask)
        return {
            "mlm_logits": self.mlm_head(token_states),
            "structural_logits": self.structural_binary_head(pooled),
            "num_constraint_logits": self.num_constraints_head(pooled),
        }


class M4ClassifierMLP(nn.Module):
    def __init__(self, config: M4EncoderConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.classifier_hidden_size),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden_size, 1),
        )

    def forward(self, pooled_prompt: torch.Tensor) -> torch.Tensor:
        return self.layers(pooled_prompt).squeeze(-1)


class M4FrozenEncoderClassifier(nn.Module):
    def __init__(self, config: M4EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.prompt_encoder = PromptBidirectionalEncoder(config.to_m3_config())
        self.pooling = PromptPooling(config.to_m3_config())
        self.pass_head = M4ClassifierMLP(config)
        self.freeze_encoder()

    def freeze_encoder(self) -> None:
        for parameter in self.prompt_encoder.parameters():
            parameter.requires_grad = False

    def load_pretrained_encoder(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.prompt_encoder.load_state_dict(state_dict)
        self.freeze_encoder()

    def encode_frozen(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.prompt_encoder.eval()
        with torch.no_grad():
            cls_state, token_states, token_mask = self.prompt_encoder(input_ids, attention_mask)
            return self.pooling(cls_state, token_states, token_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled_prompt = self.encode_frozen(input_ids, attention_mask)
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


def normalize_instruction_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def instruction_family(instruction_id: str) -> str:
    return instruction_id.split(":", 1)[0]


def derive_structural_targets(
    instruction_ids: Any,
    num_constraints: int,
) -> tuple[list[float], int]:
    ids = normalize_instruction_ids(instruction_ids)
    families = {instruction_family(instruction_id) for instruction_id in ids}
    lowered = [instruction_id.lower() for instruction_id in ids]

    has_keyword_exclusion = any(
        family == "keywords" and ("forbidden" in item or "exclude" in item)
        for family, item in zip((instruction_family(item) for item in ids), lowered, strict=True)
    )
    has_keyword_inclusion = any(
        family == "keywords" and "forbidden" not in item and "exclude" not in item
        for family, item in zip((instruction_family(item) for item in ids), lowered, strict=True)
    )
    has_list_bullet = any("bullet" in item or "list" in item for item in lowered)
    has_punctuation = "punctuation" in families or any("hyphen" in item for item in lowered)

    binary_targets = [
        float(bool(families & {"length_constraints", "count", "letters", "paragraphs"})),
        float(has_keyword_inclusion),
        float(has_keyword_exclusion),
        float(has_punctuation),
        float(bool(families & {"detectable_format", "detectable_content"})),
        float("change_case" in families),
        float(bool(families & {"startend", "first_word", "last_word"}) or any("start_end" in item for item in lowered)),
        float(has_list_bullet),
    ]
    if num_constraints <= 1:
        bucket = 0
    elif num_constraints == 2:
        bucket = 1
    elif num_constraints == 3:
        bucket = 2
    else:
        bucket = 3
    return binary_targets, bucket


def quote_inside_mask(text: str) -> list[bool]:
    inside = [False] * len(text)
    pairs = {
        '"': '"',
        "`": "`",
        "“": "”",
        "‘": "’",
        "「": "」",
        "『": "』",
    }
    closing_to_opening = {closing: opening for opening, closing in pairs.items()}
    stack: list[tuple[str, int]] = []

    for index, char in enumerate(text):
        if stack and char == stack[-1][0]:
            _, start = stack.pop()
            for span_index in range(start + 1, index):
                inside[span_index] = True
        elif char in pairs:
            stack.append((pairs[char], index))
        elif char in closing_to_opening:
            for stack_index in range(len(stack) - 1, -1, -1):
                expected, start = stack[stack_index]
                if expected == char:
                    del stack[stack_index:]
                    for span_index in range(start + 1, index):
                        inside[span_index] = True
                    break
    return inside


def compute_token_mask_categories(
    *,
    text: str,
    input_ids: list[int],
    tokenizer: Any,
    max_length: int,
) -> list[int]:
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        verbose=False,
    )
    offsets = list(encoding["offset_mapping"])
    quote_mask = quote_inside_mask(text)
    categories: list[int] = []
    for token_index in range(len(input_ids)):
        if token_index >= len(offsets):
            categories.append(0)
            continue
        start, end = offsets[token_index]
        start = max(0, int(start))
        end = min(len(text), int(end))
        if end <= start:
            categories.append(0)
            continue
        span = text[start:end]
        has_digit = any(char.isdigit() for char in span)
        in_quote = any(quote_mask[start:end])
        categories.append(int(has_digit or in_quote))
    return categories


class M4PretrainingDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        for _, row in frame.reset_index(drop=True).iterrows():
            input_ids = normalize_token_ids(row["input_ids"], max_length=max_length)
            text = "" if pd.isna(row["user_prompt"]) else str(row["user_prompt"])
            structural_targets, num_bucket = derive_structural_targets(
                row["instruction_ids"],
                int(row["num_constraints"]),
            )
            self.rows.append(
                {
                    "input_ids": input_ids,
                    "mask_categories": compute_token_mask_categories(
                        text=text,
                        input_ids=input_ids,
                        tokenizer=tokenizer,
                        max_length=max_length,
                    ),
                    "structural_targets": structural_targets,
                    "num_constraint_bucket": num_bucket,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def m4_pretraining_collate_fn(
    samples: list[dict[str, Any]],
    *,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
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
    mask_categories = torch.zeros((len(samples), max_batch_length), dtype=torch.float32)
    structural_targets = torch.tensor(
        [sample["structural_targets"] for sample in samples],
        dtype=torch.float32,
    )
    num_constraint_buckets = torch.tensor(
        [sample["num_constraint_bucket"] for sample in samples],
        dtype=torch.long,
    )

    for row_index, sample in enumerate(samples):
        token_ids = sample["input_ids"]
        categories = sample["mask_categories"]
        if token_ids:
            token_count = len(token_ids)
            input_ids[row_index, :token_count] = torch.tensor(token_ids, dtype=torch.long)
            attention_mask[row_index, :token_count] = 1
            mask_categories[row_index, :token_count] = torch.tensor(categories, dtype=torch.float32)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mask_categories": mask_categories,
        "structural_targets": structural_targets,
        "num_constraint_buckets": num_constraint_buckets,
    }


class M4ClassifierDataset(Dataset):
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


def m4_classifier_collate_fn(
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


def make_m4_classifier_dataloader(
    frame: pd.DataFrame,
    config: M4EncoderConfig,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    generator: torch.Generator | None = None,
) -> DataLoader:
    dataset = M4ClassifierDataset(frame, max_length=config.max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        collate_fn=lambda samples: m4_classifier_collate_fn(
            samples,
            pad_token_id=config.resolved_pad_token_id,
        ),
    )


def make_m4_pretraining_dataloader(
    frame: pd.DataFrame,
    config: M4EncoderConfig,
    *,
    tokenizer: Any,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    generator: torch.Generator | None = None,
) -> DataLoader:
    dataset = M4PretrainingDataset(frame, tokenizer=tokenizer, max_length=config.max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        collate_fn=lambda samples: m4_pretraining_collate_fn(
            samples,
            pad_token_id=config.resolved_pad_token_id,
        ),
    )


def make_mlm_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_categories: torch.Tensor,
    *,
    base_mask_rate: float,
    high_mask_rate: float,
    mask_token_id: int,
    special_token_ids: set[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.0 < base_mask_rate < 1.0:
        raise ValueError(f"base_mask_rate must be in (0, 1), got {base_mask_rate}.")
    if not 0.0 < high_mask_rate < 1.0:
        raise ValueError(f"high_mask_rate must be in (0, 1), got {high_mask_rate}.")

    eligible = attention_mask.bool()
    for special_token_id in special_token_ids:
        eligible = eligible & input_ids.ne(int(special_token_id))

    rates = torch.full(input_ids.shape, base_mask_rate, dtype=torch.float32, device=input_ids.device)
    rates = torch.where(mask_categories.to(input_ids.device).gt(0), high_mask_rate, rates)
    random_values = torch.rand(input_ids.shape, device=input_ids.device)
    selected = random_values.lt(rates) & eligible

    for row_index in range(selected.shape[0]):
        if not bool(selected[row_index].any()) and bool(eligible[row_index].any()):
            candidate_positions = torch.nonzero(eligible[row_index], as_tuple=False).flatten()
            selected_position = candidate_positions[
                torch.randint(candidate_positions.numel(), (1,), device=input_ids.device)
            ]
            selected[row_index, selected_position] = True

    mlm_labels = input_ids.clone()
    mlm_labels[~selected] = -100
    masked_input_ids = input_ids.clone()
    masked_input_ids[selected] = int(mask_token_id)
    return masked_input_ids, mlm_labels, selected


def train_m4_pretraining_epoch(
    model: M4PretrainingModel,
    dataloader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
    base_mask_rate: float,
    high_mask_rate: float,
    lambda_struct: float,
    special_token_ids: set[int],
    scaler: torch.cuda.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
    model.train()
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    structural_criterion = nn.BCEWithLogitsLoss()
    bucket_criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_mlm_loss = 0.0
    total_struct_loss = 0.0
    total_masked_tokens = 0
    total_examples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        mask_categories = batch["mask_categories"].to(device)
        structural_targets = batch["structural_targets"].to(device)
        num_constraint_buckets = batch["num_constraint_buckets"].to(device)
        masked_input_ids, mlm_labels, selected_mask = make_mlm_batch(
            input_ids,
            attention_mask,
            mask_categories,
            base_mask_rate=base_mask_rate,
            high_mask_rate=high_mask_rate,
            mask_token_id=model.config.mask_token_id,
            special_token_ids=special_token_ids,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(masked_input_ids, attention_mask)
            mlm_loss = mlm_criterion(
                outputs["mlm_logits"].reshape(-1, model.config.vocab_size),
                mlm_labels.reshape(-1),
            )
            binary_loss = structural_criterion(
                outputs["structural_logits"],
                structural_targets,
            )
            bucket_loss = bucket_criterion(
                outputs["num_constraint_logits"],
                num_constraint_buckets,
            )
            structural_loss = 0.5 * (binary_loss + bucket_loss)
            loss = mlm_loss + lambda_struct * structural_loss

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

        batch_size = int(input_ids.shape[0])
        masked_tokens = int(selected_mask.sum().detach().cpu())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_mlm_loss += float(mlm_loss.detach().cpu()) * batch_size
        total_struct_loss += float(structural_loss.detach().cpu()) * batch_size
        total_masked_tokens += masked_tokens
        total_examples += batch_size

    return {
        "loss": total_loss / max(total_examples, 1),
        "mlm_loss": total_mlm_loss / max(total_examples, 1),
        "structural_loss": total_struct_loss / max(total_examples, 1),
        "masked_tokens_per_example": total_masked_tokens / max(total_examples, 1),
    }


def train_m4_classifier_epoch(
    model: M4FrozenEncoderClassifier,
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
    model.prompt_encoder.eval()
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
                torch.nn.utils.clip_grad_norm_(model.pass_head.parameters(), grad_clip_norm)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.pass_head.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        if scheduler is not None:
            scheduler.step()

        batch_size = int(labels.numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
    return {"loss": total_loss / max(total_examples, 1)}


@torch.no_grad()
def predict_m4(
    model: M4FrozenEncoderClassifier,
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
            row["m4_pred_proba"] = float(probability)
            row["m4_pred_label"] = int(probability >= model.config.threshold)
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_m4_predictions(predictions: pd.DataFrame, threshold: float) -> dict[str, Any]:
    labels = predictions["label"].astype(int).to_numpy()
    probabilities = predictions["m4_pred_proba"].astype(float).to_numpy()
    return binary_classification_metrics(labels, probabilities, threshold=threshold)


def evaluate_m4_by_split(predictions: pd.DataFrame, threshold: float) -> dict[str, dict[str, Any]]:
    split_metrics: dict[str, dict[str, Any]] = {}
    for split, split_frame in predictions.groupby("split", sort=True):
        split_metrics[str(split)] = evaluate_m4_predictions(split_frame, threshold)
    return split_metrics
