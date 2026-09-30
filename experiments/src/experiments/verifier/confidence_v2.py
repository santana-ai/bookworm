"""Literature grounding scorers against the serafim cosine and the E3x primary as confidence
signals for UDV evidence, on the train and validation opinions of the NLI benchmark (E-A).

Run as ``python -m experiments.verifier.confidence_v2`` with the ``translate``, ``smoke``,
``decide``, ``score``, ``laya-smoke``, ``laya-score`` and ``evaluate-ea`` commands. The code lives
in the ``experiments.verifier.grounding`` package.
"""

from experiments.verifier.grounding.cli import main
from experiments.verifier.grounding.provenance import SOURCES, code_hashes

__all__ = ["SOURCES", "code_hashes", "main"]

if __name__ == "__main__":
    main()
