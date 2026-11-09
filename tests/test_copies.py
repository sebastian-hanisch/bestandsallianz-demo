"""Der kopierte Baustein: `bsa_rng.py` ist eine wörtliche Kopie von `ab-test-demo/abt_rng.py` (SplitMix64, Portfolio-Konvention).

Solange das Original im Portfolio-Ordner liegt, wird Byte für Byte verglichen; sonst gilt die Prüfsumme des Textes (Zeilenenden normalisiert), die beim Kopieren
bestimmt wurde. Ändert sich das Original, gehört die Änderung dorthin und die Kopie wird neu gezogen, nicht hier ergänzt."""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = ROOT.parent / "ab-test-demo" / "abt_rng.py"
COPY_SHA256 = "7ff588c02bcc96a1798c49ffb4e4b99decf7780eb8f119e8b2ba3af6fc88db48"   # Prüfsumme von abt_rng.py beim Kopieren (2026-11-09)


def _normalised(path: Path) -> bytes:
    return "\n".join(path.read_text(encoding="utf-8").splitlines()).encode("utf-8")


def test_copy_equals_the_original_when_it_is_available():
    if not ORIGINAL.exists():
        pytest.skip("Original liegt nicht im Portfolio-Ordner")
    assert _normalised(ROOT / "bsa_rng.py") == _normalised(ORIGINAL), "die Kopie weicht vom Original ab"


def test_copy_has_the_pinned_checksum():
    assert hashlib.sha256(_normalised(ROOT / "bsa_rng.py")).hexdigest() == COPY_SHA256


def test_copied_generator_has_the_documented_stream():
    """SplitMix64 mit Seed 0 liefert die bekannten ersten Werte der Konvention (Vigna)."""
    from bsa_rng import SplitMix64

    rng = SplitMix64(0)
    assert [rng.next() for _ in range(3)] == [16294208416658607535, 7960286522194355700, 487617019471545679]


def test_the_vectorised_block_reproduces_the_scalar_stream():
    from bsa_rng import SplitMix64, uniform_block

    rng = SplitMix64(12345)
    scalar = [rng.uniform() for _ in range(50)]
    assert uniform_block(12345, 50).tolist() == pytest.approx(scalar, abs=0, rel=0)
