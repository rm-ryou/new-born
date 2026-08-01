"""Encode raw text with trained BPE merge rules using configs/dataset.yaml.

Input:
    data/raw/tiny_codes.txt
    data/tokenizer/merge_rules.pkl

Output:
    data/processed/tiny_codes.bin
"""

from __future__ import annotations
from new_born.tokenizer.bpe import DEFAULT_END_TOKEN
from new_born.tokenizer import BPETokenizer

import sys
from array import array
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
CONFIG_PATH = PROJECT_ROOT / "configs/dataset.yaml"


def load_config(filepath: Path) -> dict[str, Any]:
    with filepath.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        msg = f"Dataset config must be a mapping: {filepath}"
        raise ValueError(msg)

    return config


def resolve_config() -> tuple[Path, Path, Path, str, str]:
    config = load_config(CONFIG_PATH)

    input_path = PROJECT_ROOT / str(config["input"])
    tokenizer_path = PROJECT_ROOT / str(config["tokenizer"])
    output_path = PROJECT_ROOT / str(config["output"])
    dtype = str(config.get("dtype", "uint16"))
    end_token = str(config.get("end_token", DEFAULT_END_TOKEN))

    return input_path, tokenizer_path, output_path, dtype, end_token


def save_token_ids(token_ids: list[int], output_path: Path, dtype: str) -> None:
    if dtype != "uint16":
        msg = f"Unsupported dtype: {dtype}"
        raise ValueError(msg)

    max_token_id = max(token_ids, default=0)
    if max_token_id > 65535:
        msg = f"uint16 cannot store token id {max_token_id}"
        raise ValueError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_array = array("H", token_ids)
    with output_path.open("wb") as f:
        token_array.tofile(f)


def main() -> None:
    input_path, tokenizer_path, output_path, dtype, end_token = resolve_config()

    input_text = input_path.read_text(encoding="utf-8")
    tokenizer = BPETokenizer.load_from(tokenizer_path, end_token=end_token)
    token_ids = tokenizer.encode(input_text, show_progress=True)
    save_token_ids(token_ids, output_path, dtype)

    print(f"Encoded {len(token_ids)} tokens to {output_path}")


if __name__ == "__main__":
    main()
