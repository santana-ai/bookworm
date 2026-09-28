"""NLI verifier experiments on the NLI benchmark and on UDV-style (proposition, evidence) pairs.

Run as ``python -m experiments.verifier.nli_experiments`` with the ``fetch``, ``score``, ``plan``,
``evaluate`` and ``pairs`` commands. The code lives in the ``experiments.verifier.nli`` package.
"""

from experiments.verifier.nli.cli import main
from experiments.verifier.nli.provenance import SOURCE_FILES, code_hashes

SOURCES = SOURCE_FILES

__all__ = ["SOURCES", "code_hashes", "main"]

if __name__ == "__main__":
    main()
