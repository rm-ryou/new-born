from __future__ import annotations

import pickle
from collections import defaultdict
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path

import regex as re
from tqdm import tqdm

DEFAULT_END_TOKEN = "<|endoftext|>"  # noqa: S105

type Pair = tuple[int, int]
type MergeRules = dict[Pair, int]


def pretokenize(text: str) -> list[str]:
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.findall(pattern, text)


def count_pairs(
    ids: list[int], counts: defaultdict[Pair, int] | None = None
) -> defaultdict[Pair, int]:
    if counts is None:
        counts = defaultdict(int)

    for pair in pairwise(ids):
        counts[pair] += 1
    return counts


def merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    merged_ids: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1
    return merged_ids


def train_bpe(input_text: str, vocab_size: int, end_token: str = DEFAULT_END_TOKEN) -> MergeRules:
    min_vocab_size = 256 + 1
    if vocab_size < min_vocab_size:
        msg = f"vocab_size must be at least {min_vocab_size}"
        raise ValueError(msg)

    texts = input_text.split(end_token)

    ids_list: list[list[int]] = []
    for text in texts:
        for pretoken in pretokenize(text):
            ids_list.append(list(pretoken.encode("utf-8")))

    num_merges = vocab_size - min_vocab_size
    merge_rules: MergeRules = {}

    for step in tqdm(range(num_merges), desc="Training BPE"):
        counts: defaultdict[Pair, int] = defaultdict(int)
        for ids in ids_list:
            count_pairs(ids, counts)

        if not counts:
            break

        best_pair = max(counts, key=lambda pair: (counts[pair], pair[0], pair[1]))

        new_id = 256 + step
        merge_rules[best_pair] = new_id

        for i, ids in enumerate(ids_list):
            ids_list[i] = merge(ids, best_pair, new_id)

    return merge_rules


class BPETokenizer:
    def __init__(self, merge_rules: MergeRules, end_token: str = DEFAULT_END_TOKEN) -> None:
        self.merge_rules = merge_rules
        self.end_token = end_token
        self.end_token_id = 256 + len(merge_rules)

        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for (id1, id2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]
        self.id_to_bytes[self.end_token_id] = self.end_token.encode("utf-8")

        self.vocab_size = len(self.id_to_bytes)

    @staticmethod
    def load_from(filepath: str | Path) -> BPETokenizer:
        with Path(filepath).open("rb") as f:
            merge_rules = pickle.load(f)  # noqa: S301
        return BPETokenizer(merge_rules)

    def save_to(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.merge_rules, f)

    def _encode_text(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        for merge_pair, new_id in self.merge_rules.items():
            ids = merge(ids, merge_pair, new_id)
        return ids

    def encode(self, input_text: str, show_progress: bool = False) -> list[int]:
        pattern = f"({re.escape(self.end_token)})"
        texts: Iterable[str] = re.split(pattern, input_text)
        all_ids: list[int] = []

        if show_progress:
            texts = tqdm(texts, desc="Encoding")

        for text in texts:
            if text == self.end_token:
                all_ids.append(self.end_token_id)
            else:
                for pretoken in pretokenize(text):
                    ids = self._encode_text(pretoken)
                    all_ids.extend(ids)

        return all_ids

    def decode(self, ids: Iterable[int]) -> str:
        byte_list = [self.id_to_bytes[i] for i in ids]
        text_bytes = b"".join(byte_list)
        return text_bytes.decode("utf-8", errors="replace")
