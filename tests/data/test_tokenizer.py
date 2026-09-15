"""Tests for TokenizerWrapper — mocks HuggingFace AutoTokenizer, no network calls."""

import torch

from diffusion_moe.data import tokenizer as tokenizer_module
from diffusion_moe.data.tokenizer import TokenizerWrapper


class FakeHFTokenizer:
    """Minimal stand-in for a HuggingFace tokenizer without a native pad token."""

    def __init__(self) -> None:
        self._pad_token = None
        self.pad_token_id = None
        self.eos_token = "<eos>"
        self.eos_token_id = 2
        self.vocab_size = 100
        self.padding_side = "right"

    @property
    def pad_token(self):
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value
        self.pad_token_id = self.eos_token_id if value == self.eos_token else 0

    def _tokenize(self, text: str) -> list[int]:
        return [3 + (ord(c) % 50) for c in text]

    def __call__(
        self, texts, padding=False, truncation=False, max_length=None, return_tensors=None
    ):
        single = isinstance(texts, str)
        batch = [texts] if single else texts

        all_ids, all_mask = [], []
        for text in batch:
            ids = self._tokenize(text)
            if truncation and max_length is not None:
                ids = ids[:max_length]
            all_ids.append(ids)

        if padding == "max_length":
            target_len = max_length
        elif padding:
            target_len = max(len(ids) for ids in all_ids)
        else:
            target_len = None

        if target_len is not None:
            for ids in all_ids:
                mask = [1] * len(ids) + [0] * (target_len - len(ids))
                ids += [self.pad_token_id] * (target_len - len(ids))
                all_mask.append(mask)
        else:
            all_mask = [[1] * len(ids) for ids in all_ids]

        if return_tensors == "pt":
            result = {
                "input_ids": torch.tensor(all_ids, dtype=torch.long),
                "attention_mask": torch.tensor(all_mask, dtype=torch.long),
            }
        else:
            result = {
                "input_ids": all_ids[0] if single else all_ids,
                "attention_mask": all_mask[0] if single else all_mask,
            }
        return result


def _make_wrapper(monkeypatch) -> tuple[TokenizerWrapper, FakeHFTokenizer]:
    fake = FakeHFTokenizer()
    monkeypatch.setattr(
        tokenizer_module.AutoTokenizer, "from_pretrained", lambda name: fake
    )
    wrapper = TokenizerWrapper("fake/model")
    return wrapper, fake


def test_sets_pad_token_when_missing(monkeypatch):
    wrapper, fake = _make_wrapper(monkeypatch)
    assert fake.pad_token == fake.eos_token
    assert wrapper.pad_token_id == fake.eos_token_id


def test_encode_batch_shapes_and_dtypes(monkeypatch):
    wrapper, _ = _make_wrapper(monkeypatch)
    out = wrapper.encode_batch(["hello", "hi"], max_length=8)

    assert set(out.keys()) == {"input_ids", "attention_mask"}
    assert out["input_ids"].shape == (2, 8)
    assert out["attention_mask"].shape == (2, 8)
    assert out["input_ids"].dtype == torch.long
    assert out["attention_mask"].dtype == torch.long
    # "hi" is shorter than "hello" -> more padding -> fewer attended positions
    assert out["attention_mask"][1].sum() < out["attention_mask"][0].sum()


def test_encode_batch_truncates_to_max_length(monkeypatch):
    wrapper, _ = _make_wrapper(monkeypatch)
    out = wrapper.encode_batch(["a much longer sentence than the limit allows"], max_length=4)
    assert out["input_ids"].shape == (1, 4)


def test_encode_single_returns_list_of_ints(monkeypatch):
    wrapper, _ = _make_wrapper(monkeypatch)
    ids = wrapper.encode("hello world", max_length=5)
    assert isinstance(ids, list)
    assert len(ids) <= 5
    assert all(isinstance(i, int) for i in ids)
