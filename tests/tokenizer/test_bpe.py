from collections import defaultdict

import pytest

from new_born.tokenizer import BPETokenizer, train_bpe
from new_born.tokenizer.bpe import DEFAULT_END_TOKEN, count_pairs, merge, pretokenize


def test_pretokenize_keeps_leading_spaces() -> None:
    assert pretokenize("Hello world!") == ["Hello", " world", "!"]


def test_count_pairs_accumulates_adjacent_token_pairs() -> None:
    counts = count_pairs([1, 2, 1, 2, 3])

    assert counts == {
        (1, 2): 2,
        (2, 1): 1,
        (2, 3): 1,
    }


def test_count_pairs_can_reuse_existing_counter() -> None:
    counts: defaultdict[tuple[int, int], int] = defaultdict(int)

    count_pairs([1, 2, 3], counts)
    count_pairs([1, 2], counts)

    assert counts[(1, 2)] == 2
    assert counts[(2, 3)] == 1


def test_merge_replaces_non_overlapping_pairs() -> None:
    assert merge([1, 1, 1], (1, 1), 99) == [99, 1]
    assert merge([1, 2, 1, 2], (1, 2), 99) == [99, 99]


def test_train_bpe_rejects_vocab_size_smaller_than_byte_vocab_plus_end_token() -> None:
    with pytest.raises(ValueError, match="vocab_size must be at least 257"):
        train_bpe("hello", vocab_size=256)


def test_train_bpe_returns_deterministic_merge_rules() -> None:
    merge_rules = train_bpe("abababab", vocab_size=260)

    assert merge_rules == {
        (97, 98): 256,
        (256, 256): 257,
        (257, 257): 258,
    }


def test_bpe_tokenizer_round_trips_text() -> None:
    text = f"Hello world!{DEFAULT_END_TOKEN}Hello again."
    merge_rules = train_bpe(text, vocab_size=280)
    tokenizer = BPETokenizer(merge_rules)

    ids = tokenizer.encode(text)

    assert tokenizer.end_token_id in ids
    assert tokenizer.decode(ids) == text


def test_bpe_tokenizer_can_encode_unknown_text_using_bytes() -> None:
    merge_rules = train_bpe("hello hello", vocab_size=270)
    tokenizer = BPETokenizer(merge_rules)

    text = "unseen text"

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_bpe_tokenizer_save_and_load_round_trips_merge_rules(tmp_path) -> None:
    merge_rules = train_bpe("hello hello", vocab_size=270)
    tokenizer = BPETokenizer(merge_rules)
    filepath = tmp_path / "tokenizer.pkl"

    tokenizer.save_to(filepath)
    loaded = BPETokenizer.load_from(filepath)

    assert loaded.merge_rules == tokenizer.merge_rules
    assert loaded.decode(loaded.encode("hello unseen")) == "hello unseen"
