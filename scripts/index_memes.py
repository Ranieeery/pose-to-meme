"""Indexes the meme folder with CLIP. Run from the project root: python -m scripts.index_memes

The app does this on its own when the folder changes; this script shows the result, so
a wrong match can be fixed in memes.overrides and a weak one spotted by its margin.
"""

import sys
from collections.abc import Sequence
from pathlib import Path

from pose_meme.config import ConfigError, load_config
from pose_meme.meme_index import BUCKETS_FILE, ClipEncoder, MemeLoadError, MemeMatch, group_by_pose, index_memes

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def format_matches(matches: Sequence[MemeMatch], classes: Sequence[str]) -> list[str]:
    """A table of meme -> class, plus the classes no meme went to."""
    lines = [f"{'meme':30} {'classe':26} {'similaridade':>12} {'margem':>7}  2a colocada / nota"]
    for match in matches:
        margin = "-" if match.margin is None else f"{match.margin:.3f}"
        if match.overridden:
            note = f"override (CLIP: {match.clip_pose})"
        elif match.pose is None:
            note = f"ignorado (abaixo do minimo; CLIP: {match.clip_pose})"
        else:
            note = match.runner_up or ""
        lines.append(f"{match.file:30} {match.pose or '-':26} {match.similarity:12.3f} {margin:>7}  {note}")
    if missing := [name for name in classes if name not in group_by_pose(matches)]:
        lines.append(f"classes sem meme: {', '.join(missing)}")
    return lines


def main() -> None:
    try:
        config = load_config(CONFIG_PATH)
        matches = index_memes(config.memes, config.poses.prompts, lambda: ClipEncoder(config.memes.clip_model))
    except (ConfigError, MemeLoadError) as error:
        sys.exit(f"error: {error}")

    print("\n".join(format_matches(matches, list(config.poses.prompts))))
    print(f"buckets: {config.memes.cache_dir / BUCKETS_FILE}")


if __name__ == "__main__":
    main()
