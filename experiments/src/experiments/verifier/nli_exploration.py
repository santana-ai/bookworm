"""Exploration of verifier candidates: cross-validation on train, confirmation on validation and
a single final test.

Run as ``python -m experiments.verifier.nli_exploration`` with the ``explore``, ``confirm`` and
``final-test`` commands. The code lives in the ``experiments.verifier.exploration`` package.
"""

from experiments.verifier.exploration.cli import main
from experiments.verifier.exploration.provenance import SOURCES, code_hashes

__all__ = ["SOURCES", "code_hashes", "main"]

if __name__ == "__main__":
    main()
