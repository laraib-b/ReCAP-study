"""Repository path helpers after the implementation/ layout change."""

from __future__ import annotations

from pathlib import Path

IMPL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = IMPL_ROOT.parent
