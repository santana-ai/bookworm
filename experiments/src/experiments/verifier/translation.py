"""Translation of the benchmark opinions and NLI chunks with pinned seq2seq models.

Run as ``python -m experiments.verifier.translation`` with the ``fetch``, ``plan``, ``translate``,
``report`` and ``spot-check`` commands. The code lives in the ``experiments.verifier.translate``
package.
"""

from experiments.verifier.translate.cli import main
from experiments.verifier.translate.report import SOURCES, code_hashes

__all__ = ["SOURCES", "code_hashes", "main"]

if __name__ == "__main__":
    main()
