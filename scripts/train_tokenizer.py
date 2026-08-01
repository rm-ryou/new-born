"""Train BPE merge rules from raw text using configs/tokenizer.yaml.

Input:
    data/raw/tiny_codes.txt

Output:
    data/tokenizer/merge_rules.pkl
"""

from __future__ import annotations
from new_born.tokenizer.bpe import DEFAULT_END_TOKEN
from new_born.tokenizer import BPETokenizer, train_bpe

import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
CONFIG_PATH = PROJECT_ROOT / "configs/tokenizer.yaml"


def load_config(filepath: Path) -> dict[str, Any]:
    with filepath.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        msg = f"Tokenizer config must be a mapping: {filepath}"
        raise ValueError(msg)

    return config


def resolve_config() -> tuple[Path, Path, int, str]:
    config = load_config(CONFIG_PATH)

    input_path = PROJECT_ROOT / str(config["input"])
    output_path = PROJECT_ROOT / str(config["output"])
    vocab_size = int(config["vocab_size"])
    end_token = str(config.get("end_token", DEFAULT_END_TOKEN))

    return input_path, output_path, vocab_size, end_token


def main() -> None:
    input_path, output_path, vocab_size, end_token = resolve_config()

    input_text = input_path.read_text(encoding="utf-8")
    merge_rules = train_bpe(input_text, vocab_size=vocab_size, end_token=end_token)
    tokenizer = BPETokenizer(merge_rules, end_token=end_token)
    tokenizer.save_to(output_path)
    print(f"Saved tokenizer to {output_path}")


if __name__ == "__main__":
    main()
