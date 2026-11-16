"""Package metadata for the Derecho IA normative repository."""

from pathlib import Path

from setuptools import find_packages, setup


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"

setup(
    name="derecho-ia-normative-repository",
    version="0.1.0",
    description="Local document processing pipeline for Peruvian legal sources",
    packages=find_packages(include=("normative_repository", "normative_repository.*")),
    python_requires=">=3.12",
    install_requires=[
        line.strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ],
)
