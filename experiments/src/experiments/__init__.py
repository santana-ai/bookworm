"""Experiments of bookworm on PublicHearingBR, one subpackage per stage.

``data`` builds the benchmarks from the dataset, ``udv`` calibrates and measures the evidence
units, ``retrieval`` compares retrievers and unit kinds, ``verifier`` runs the NLI and decision
verifiers, ``validation`` draws and scores the human validation samples, ``actors`` builds speeches,
profiles and simulations, ``mlx`` runs the local MLX backend, and ``common`` holds the helpers
they share. Each command runs as ``python -m experiments.<stage>.<module>`` from ``experiments/``.
"""
