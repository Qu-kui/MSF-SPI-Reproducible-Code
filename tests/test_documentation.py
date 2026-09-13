from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".yaml", ".yml", ".cff", ".txt"}
EXEMPT = {
    ROOT / "docs" / "source_manifest.json",
    ROOT / "docs" / "SOURCE_INTEGRITY.md",
    ROOT / "docs" / "FILE_PROVENANCE.md",
}


def test_public_text_is_english_only():
    violations = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES or ".git" in path.parts:
            continue
        if path in EXEMPT or "superpowers" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if HAN.search(text):
            violations.append(str(path.relative_to(ROOT)))
    assert not violations, violations


def test_executable_and_user_documentation_have_no_private_paths_or_internal_modes():
    roots = [
        ROOT / "src",
        ROOT / "scripts",
        ROOT / "tools",
        ROOT / "configs",
        ROOT / "README.md",
    ]
    forbidden = ("\u6700\u7ec8\u7efc\u5408\u6d4b\u8bd5", "\u5ba1\u7a3f\u610f\u89c1", "Moire_SPI_260603", "Mode 4", "Mode 10")
    violations = []
    for root in roots:
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if any(token in text for token in forbidden):
                    violations.append(str(path.relative_to(ROOT)))
    assert not violations, violations


def test_required_public_documents_exist():
    required = (
        "README.md",
        "data/README.md",
        "docs/MODEL_ASSUMPTIONS.md",
        "docs/CALIBRATION.md",
        "docs/REPRODUCIBILITY.md",
        "docs/FILE_PROVENANCE.md",
        "docs/RESULTS_INDEX.md",
    )
    assert all((ROOT / name).is_file() for name in required)
