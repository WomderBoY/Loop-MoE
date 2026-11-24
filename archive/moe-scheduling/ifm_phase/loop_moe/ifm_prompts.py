"""Author-style base-model prompt formatting, quality checks, and request splits for IFM A1-A3."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import re
import subprocess
import sys
from typing import Any


# Exact GSM8K chain-of-thought exemplars from xllm/paper_part1/tasks/gsm8k.py (first 4 shots).
GSM8K_FEWSHOT: list[tuple[str, str]] = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "Let's think step by step. There are 15 trees originally. Then there were 21 trees after "
        "some more were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total "
        "they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
        "How many lollipops did Jason give to Denny?",
        "Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
]

# Author-style MBPP few-shot exemplars matching xllm/paper_part1/tasks/mbpp.py format.
MBPP_PROMPT_TEMPLATE = (
    "You are an expert Python programmer, and here is your task: {context} "
    "Your code should pass these tests:\n\n{tests}\n[BEGIN]\n"
)

MBPP_FEWSHOT: list[dict[str, Any]] = [
    {
        "text": "Write a function to find the shared elements from the given two lists.",
        "test_list": [
            "assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))",
            "assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))",
        ],
        "code": "def similar_elements(test_tup1, test_tup2):\n    return tuple(set(test_tup1) & set(test_tup2))",
    },
    {
        "text": "Write a python function to identify non-prime numbers.",
        "test_list": [
            "assert is_not_prime(2) == False",
            "assert is_not_prime(10) == True",
            "assert is_not_prime(35) == True",
        ],
        "code": (
            "import math\n"
            "def is_not_prime(n):\n"
            "    if n < 2:\n"
            "        return True\n"
            "    for i in range(2, int(math.sqrt(n)) + 1):\n"
            "        if n % i == 0:\n"
            "            return True\n"
            "    return False"
        ),
    },
]


def format_ifm_prompt(item: dict[str, Any]) -> tuple[str, list[str]]:
    """Return (author_formatted_prompt, stop_strings) for IFM Part I base models."""
    category = item["category"]
    if category == "math" and "question" in item:
        shots = [f"Q: {q}\nA: {a}" for q, a in GSM8K_FEWSHOT]
        prompt = "\n\n".join(shots + [f"Q: {item['question']}\nA:"])
        return prompt, ["\n\nQ:", "\nQ:"]
    if category == "code" and "test_list" in item:
        shots = [
            MBPP_PROMPT_TEMPLATE.format(
                context=ex["text"],
                tests="\n".join(ex["test_list"]),
            )
            + f"{ex['code']}\n[DONE]"
            for ex in MBPP_FEWSHOT
        ]
        target = MBPP_PROMPT_TEMPLATE.format(
            context=item["text"],
            tests="\n".join(item["test_list"]),
        )
        return "\n".join(shots + [target]), ["[DONE]"]
    prompt = item.get("prompt") or item.get("text")
    if not prompt:
        raise ValueError(f"Prompt item {item.get('id')} is missing text/prompt")
    stops = item.get("stop_strings") or ["\n\n\n"]
    return prompt, list(stops)


def extract_number(text: str) -> str | None:
    """Extract final numeric answer matching GSM8K evaluation conventions."""
    cleaned = text.strip()
    m = re.search(r"The answer is\s*\$?(-?\d[\d,]*(?:\.\d+)?)", cleaned, flags=re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", cleaned)
    if not nums:
        return None
    return nums[-1].replace(",", "")


def grade_gsm8k(generated_text: str, gold_answer: str) -> tuple[bool, str | None]:
    pred = extract_number(generated_text)
    if pred is None:
        return False, None
    try:
        return math.isclose(float(pred), float(gold_answer.replace(",", "")), rel_tol=1e-6, abs_tol=1e-6), pred
    except ValueError:
        return pred.strip() == gold_answer.strip(), pred


def evaluate_mbpp_code(generated_text: str, test_list: list[str], timeout_s: float = 3.0) -> tuple[bool, str]:
    """Execute generated MBPP solution against unit tests in an isolated subprocess."""
    code = generated_text.split("[DONE]")[0].strip()
    if not code:
        return False, "empty"
    script = code + "\n\n" + "\n".join(test_list) + "\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode == 0:
            return True, "passed"
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        return False, err[-1][:160] if err else f"exit_{proc.returncode}"
    except subprocess.TimeoutExpired:
        return False, "timeout"


def ngram_repetition_rate(token_ids: list[int], n: int = 4) -> float:
    """Fraction of n-grams in token_ids that are duplicates of earlier n-grams."""
    if len(token_ids) < n:
        return 0.0
    total = len(token_ids) - n + 1
    unique = len({tuple(token_ids[i : i + n]) for i in range(total)})
    return 1.0 - unique / total


def split_requests_three_way(
    requests: list[dict[str, Any]],
    *,
    fit_ratio: float = 1 / 3,
    val_ratio: float = 1 / 6,
    seed: int = 20261121,
) -> dict[str, list[str]]:
    """Deterministic stratified fit / val / eval split by request ID."""
    by_category: dict[str, list[str]] = defaultdict(list)
    for req in requests:
        req_id = req.get("request_id") or req["id"]
        by_category[req["category"]].append(req_id)

    fit_ids: list[str] = []
    val_ids: list[str] = []
    eval_ids: list[str] = []
    for category in sorted(by_category):
        ordered = sorted(
            by_category[category],
            key=lambda rid: hashlib.sha256(f"{seed}:{category}:{rid}".encode()).digest(),
        )
        n = len(ordered)
        n_fit = max(1, int(round(n * fit_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
        if n_fit + n_val >= n:
            n_val = max(0, n - n_fit - 1)
        fit_ids.extend(ordered[:n_fit])
        val_ids.extend(ordered[n_fit : n_fit + n_val])
        eval_ids.extend(ordered[n_fit + n_val :])
    return {
        "fit_request_ids": sorted(fit_ids),
        "val_request_ids": sorted(val_ids),
        "eval_request_ids": sorted(eval_ids),
    }
