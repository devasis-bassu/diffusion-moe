"""Tokenizer wrapper around HuggingFace AutoTokenizer."""

from __future__ import annotations

import torch
from transformers import AutoTokenizer


class TokenizerWrapper:
    """Wraps a HuggingFace tokenizer with padding/truncation defaults for causal LM."""

    def __init__(self, pretrained_name: str, padding_side: str = "right") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        if self.tokenizer.pad_token is None:
            # Many causal LM tokenizers (e.g. Mistral, Llama) ship without a pad token.
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = padding_side

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def encode_batch(self, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
        """Tokenize a batch of texts with padding/truncation to max_length.

        Returns a dict with "input_ids" and "attention_mask", each a LongTensor
        of shape (batch, max_length).
        """
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].long(),
            "attention_mask": encoded["attention_mask"].long(),
        }

    def encode(self, text: str, max_length: int, truncation: bool = True) -> list[int]:
        """Tokenize a single string without padding — used by streaming datasets."""
        return self.tokenizer(
            text,
            truncation=truncation,
            max_length=max_length,
        )["input_ids"]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decodes token ids back to text — used to recover the exact UTF-8
        byte length of a (possibly truncated) token sequence for bits-per-byte."""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
