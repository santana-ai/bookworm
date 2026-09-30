"""The selected verifier applied to every UDV: translate, score and apply.

Run as ``python -m experiments.verifier.udv_verifier`` with the ``translate``, ``score`` and
``apply`` commands. The code lives in the ``experiments.verifier.udv_scores`` package.
"""

from experiments.verifier.udv_scores.cli import main
from experiments.verifier.udv_scores.primary import fit_primary
from experiments.verifier.udv_scores.provenance import SOURCES, code_hashes

__all__ = ["SOURCES", "code_hashes", "fit_primary", "main"]

if __name__ == "__main__":
    main()
