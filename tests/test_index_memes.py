from pose_meme.meme_index import MemeMatch
from scripts.index_memes import format_matches

CLASSES = ["pose_x", "pose_y", "pose_z"]


def match(file: str, pose: str | None, clip_pose: str = "pose_x", overridden: bool = False) -> MemeMatch:
    return MemeMatch(file, pose, clip_pose, 0.25, "pose_y", 0.01, overridden)


def test_each_meme_gets_a_row_with_its_class_and_margin() -> None:
    lines = format_matches([match("a.png", "pose_x")], CLASSES)

    row = next(line for line in lines if "a.png" in line)
    assert "pose_x" in row
    assert "0.250" in row
    assert "0.010" in row


def test_ignored_and_overridden_memes_are_flagged() -> None:
    lines = "\n".join(
        format_matches([match("a.png", None), match("b.png", "pose_y", overridden=True)], CLASSES)
    )

    assert "ignorado" in lines
    assert "override (CLIP: pose_x)" in lines


def test_classes_without_memes_are_listed() -> None:
    lines = format_matches([match("a.png", "pose_x")], CLASSES)

    assert lines[-1] == "classes sem meme: pose_y, pose_z"


def test_no_missing_classes_line_when_every_class_has_a_meme() -> None:
    lines = format_matches([match("a.png", "pose_x"), match("b.png", "pose_y"), match("c.png", "pose_z")], CLASSES)

    assert not any(line.startswith("classes sem meme") for line in lines)
