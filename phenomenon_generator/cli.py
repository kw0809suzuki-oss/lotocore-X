from __future__ import annotations

import argparse
import json
from pathlib import Path

from generator import GenerationError, run_recipe


def main() -> None:
    p = argparse.ArgumentParser(description="LOTO7 局所現象 配置生成機構 v0")
    p.add_argument("recipe", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))

    try:
        result = run_recipe(recipe)
    except GenerationError as exc:
        raise SystemExit(f"generation rejected: {exc}") from exc

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"saved: {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
