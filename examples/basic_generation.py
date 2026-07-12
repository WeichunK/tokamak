"""Minimal usage example: chat with a model via the offline LLM API.

Usage:
    uv run python examples/basic_generation.py
    uv run python examples/basic_generation.py --prompt "Explain KV caching briefly."
"""

import argparse

from tokamak import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="What does an LLM inference engine do?")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    llm = LLM(args.model)

    # The engine consumes raw text; chat formatting is the tokenizer's job.
    chat_prompt = llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    outputs = llm.generate(
        chat_prompt,
        SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        ),
    )

    print(f"\n[prompt]\n{args.prompt}\n")
    print(f"[completion ({outputs[0].finish_reason.value})]\n{outputs[0].output_text}")


if __name__ == "__main__":
    main()
