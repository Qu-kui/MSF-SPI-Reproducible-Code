from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_release_directories_exist():
    required = (
        "configs",
        "data",
        "docs",
        "results",
        "scripts",
        "src",
        "tests",
        "tools",
    )
    assert all((ROOT / name).is_dir() for name in required)


def test_private_source_roots_are_not_nested_in_release():
    names = {path.name for path in ROOT.rglob("*")}
    assert "\u6700\u7ec8\u7efc\u5408\u6d4b\u8bd5" not in names
    assert "\u5ba1\u7a3f\u610f\u89c1" not in names
