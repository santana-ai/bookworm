"""Confidence signals and accept policies for the top-1 answer of the retrieval harness.

Run as ``python -m experiments.verifier.confidence_policies`` with the ``collect``, ``pairs`` and
``evaluate`` commands. The code lives in the ``experiments.verifier.confidence`` package.
"""

from experiments.verifier.confidence.cli import main
from experiments.verifier.confidence.provenance import SOURCES, code_hashes

__all__ = ["SOURCES", "code_hashes", "main"]

if __name__ == "__main__":
    main()
