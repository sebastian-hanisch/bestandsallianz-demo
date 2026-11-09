"""README und App sagen dasselbe: geteilter Text, Testzahl, genannte Testnamen, keine Platzhalter."""

import re
import subprocess
import sys
from pathlib import Path

from bsa_texts import ASSUMPTIONS_MARKDOWN

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_shared_assumptions_text_is_in_the_readme_verbatim():
    assert ASSUMPTIONS_MARKDOWN.strip() in README


def test_the_test_count_in_the_readme_matches_the_suite():
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True, timeout=300).stdout
    collected = int(re.search(r"(\d+) tests? collected", out).group(1))
    stated = int(re.search(r"\*\*(\d+) Tests\*\*", README).group(1))
    assert stated == collected


def test_every_test_name_the_readme_cites_exists():
    sources = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "tests").glob("test_*.py"))
    for name in set(re.findall(r"`(test_[a-z0-9_]+)(?:\*)?`", README)):
        assert re.search(rf"def {re.escape(name)}", sources), name
    for stem in set(re.findall(r"`(test_[a-z0-9_]+)_\*`", README)):
        assert re.search(rf"def {re.escape(stem)}_", sources), stem


def test_readme_has_no_open_placeholders_and_names_the_live_link():
    assert not [line for line in README.splitlines() if "TODO" in line]
    assert "sebastianhanisch-bestandsallianz-demo.streamlit.app" in README
