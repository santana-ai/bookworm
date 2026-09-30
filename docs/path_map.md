# Mapa de caminhos

A versão 1.0 do repositório reorganizou as pastas, os documentos e os módulos dos experimentos. Os
artefatos gerados antes dela não foram reescritos e continuam citando os caminhos antigos: os relatórios
JSON (seção `code` e comandos gravados), o manifesto dos arquivos pesados
(`experiments/artifacts/MANIFEST_heavy.tsv`, colunas `path` e `regenerate`), os relatórios das rodadas
MLX, as ADRs e a prosa das declarações nos TOML de `experiments/configs/`, cujo sha256 está gravado nos
relatórios. Esta página dá, para cada caminho antigo, o caminho desta versão.

A tag `research-2026-09-28` guarda o repositório inteiro no layout antigo, com o histórico de todos os
arquivos, inclusive os que saíram desta versão. Um arquivo antigo se lê com
`git show research-2026-09-28:<caminho antigo>`.

## Pastas e documentos

| caminho antigo | nesta versão |
|---|---|
| `challenge/` | `experiments/` |
| `challenge/utils/` | `experiments/src/experiments/` (tabela de módulos abaixo) |
| `challenge/mlx_alternative/*.py` | `experiments/src/experiments/mlx/` (`python -m experiments.mlx.<módulo>`) |
| `challenge/mlx_alternative/config.yaml` | `experiments/configs/mlx.yaml` |
| `challenge/mlx_alternative/config_udv_v2.yaml` | `experiments/configs/mlx_udv_v2.yaml` |
| `challenge/mlx_alternative/runs/` | `experiments/artifacts/mlx_runs/` |
| `challenge/mlx_alternative/README.md` | `docs/methodology/mlx_backend.md` |
| `challenge/RELATORIO_EXPERIMENTOS.md` | `docs/report.md` |
| `challenge/PIPELINE.md` | `docs/pipeline.md` |
| `challenge/UDV.md` | `docs/methodology/udv.md` |
| `challenge/CONFIDENCE_V2.md` | `docs/methodology/confidence.md` |
| `challenge/HEARING_ACTORS.md` | `docs/methodology/hearing_actors.md` |
| `challenge/ACTOR_PROFILES.md` | `docs/methodology/actor_profiles.md` |
| `challenge/ACTOR_SIMULATION.md` | `docs/methodology/actor_simulation.md` |
| `challenge/annotation_guide.md` | `docs/validation/annotation_guide.md` |
| `challenge/desafio_ideias_em_rede.md` | `docs/challenge/brief.md` |
| `challenge/2410.07495v2.pdf` | removido; o artigo do dataset é [arXiv 2410.07495](https://arxiv.org/abs/2410.07495) |
| `CONSTITUTION.md` | `docs/vision.md` |
| `ROADMAP.md` | `docs/roadmap.md` |
| `challenge/eda_v01.ipynb` | `experiments/notebooks/eda.ipynb` |
| `challenge/splits_v00.ipynb` | `experiments/notebooks/splits.ipynb` |
| `challenge/udv_v05.ipynb` | `experiments/notebooks/udv.ipynb` |
| `challenge/actor_profiles_v01.ipynb` | `experiments/notebooks/actor_profiles.ipynb` |
| `challenge/run_actor_profiles.sh`, `challenge/run_simulation_pipeline.sh` | `experiments/scripts/` |
| `challenge/artifacts/`, `challenge/configs/`, `challenge/prompts/`, `challenge/tests/`, `challenge/dataset/` | o mesmo caminho sob `experiments/` |

## Módulos

Um comando antigo `python -m utils.<nome>` passa a ser `python -m experiments.<subpacote>.<módulo>`,
rodado de dentro de `experiments/`; o arquivo fica em `experiments/src/experiments/<subpacote>/<módulo>.py`.

| arquivo antigo em `challenge/` | módulo nesta versão |
|---|---|
| `utils/actor_simulation.py` | `experiments.actors.simulation` |
| `utils/build_actor_speeches.py` | `experiments.actors.build_speeches` |
| `utils/build_nli_benchmark.py` | `experiments.data.nli_benchmark` |
| `utils/build_quote_benchmark.py` | `experiments.data.quote_benchmark` |
| `utils/cache_lock.py` | `experiments.common.cache_lock` |
| `utils/calibrate_threshold.py` | `experiments.udv.calibrate_threshold` |
| `utils/calibrate_udv_v2.py` | `experiments.udv.calibrate_v2` |
| `utils/confidence_policies.py` | `experiments.verifier.confidence_policies` |
| `utils/confidence_v2.py` | `experiments.verifier.confidence_v2` |
| `utils/decision_models.py` | `experiments.verifier.decision_models` |
| `utils/decision_scoring.py` | `experiments.verifier.decision_scoring` |
| `utils/download_dataset.py` | `experiments.data.download` |
| `utils/evaluate_actor_simulation.py` | `experiments.actors.evaluate_simulation` |
| `utils/filter_actor_speeches.py` | `experiments.actors.filter_speeches` |
| `utils/fuzzy_matching_experiments.py` | `experiments.udv.fuzzy_matching` |
| `utils/fuzzy_review_precision.py` | `experiments.validation.fuzzy_review_precision` |
| `utils/generate_actor_profiles.py` | `experiments.actors.generate_profiles` |
| `utils/generate_validation_sample.py` | `experiments.validation.generate_sample` |
| `utils/grounding_models.py` | `experiments.verifier.grounding.models` |
| `utils/grounding_scorers.py` | `experiments.verifier.grounding.scorers` |
| `utils/hub_offline.py` | `experiments.common.hub_offline` |
| `utils/measure_case_insensitive_quotes.py` | `experiments.udv.measure_case_insensitive_quotes` |
| `utils/measure_hearing_actors.py` | `experiments.actors.measure_hearing_actors` |
| `utils/measure_quote_patterns.py` | `experiments.udv.measure_quote_patterns` |
| `utils/measure_turn_segmentation.py` | `experiments.udv.measure_turn_segmentation` |
| `utils/nli_verifier_experiments.py` | `experiments.verifier.nli_experiments` |
| `utils/nli_verifier_exploration.py` | `experiments.verifier.nli_exploration` |
| `utils/precision_report.py` | `experiments.validation.precision_report` |
| `utils/retrieval_data.py` | `experiments.retrieval.data` |
| `utils/retrieval_experiments.py` | `experiments.retrieval.experiments` |
| `utils/retrieval_models.py` | `experiments.retrieval.models` |
| `utils/retrieval_stats.py` | `experiments.common.stats` |
| `utils/retrieval_store.py` | `experiments.retrieval.store` |
| `utils/simulate_actors.py` | `experiments.actors.simulate` |
| `utils/translation.py` | `experiments.verifier.translation` |
| `utils/udv_v2_analysis.py` | `experiments.udv.v2_analysis` |
| `utils/udv_v2_downstream.py` | `experiments.udv.v2_downstream` |
| `utils/udv_verifier.py` | `experiments.verifier.udv_verifier` |

Os módulos abaixo duplicavam regras que a biblioteca já tinha, com paridade testada, e saíram:

| arquivo antigo em `challenge/` | nesta versão |
|---|---|
| `utils/udv_pipeline.py` | as regras estão em `bookworm.transcript` e `bookworm.udv.quotes`; `experiments.common.transcript` dá a mesma interface em dicionários e delega à biblioteca |
| `utils/build_udvs.py` | `uv run bookworm build-udvs`; a configuração de rodada e o cache do encoder que os experimentos reusam estão em `experiments.common.udv_run` |
| `utils/verify_udvs.py` | `uv run bookworm verify-udvs` |
| `utils/build_splits.py` | `uv run bookworm build-splits`; as funções que o notebook de splits e o teste de paridade da biblioteca chamam estão em `experiments.data.legacy_splits` |
| `utils/verify_splits.py` | `uv run bookworm verify-splits`; as funções `check_*` estão em `experiments.data.legacy_splits` |
| `utils/hearing_dates.py` | `bookworm.data.dates`; as funções do notebook estão em `experiments.data.legacy_splits` |
| `utils/dataset_io.py` | `bookworm.data.io` (leitura, sha256 e escrita); os hashes da seção `code` estão em `experiments.common.provenance` |

A equivalência de `experiments.common.transcript` e `experiments.common.udv_run` com os arquivos
antigos foi conferida, antes da remoção, sobre as 206 audiências do LDS: as mesmas saídas em todas as
funções e constantes públicas, nenhuma diferença.

## Seção `code` dos relatórios

Os relatórios gravados antes desta versão registram cada arquivo de código como `utils/<nome>.py`
(relativo a `challenge/`). Os gravados a partir dela registram o caminho a partir da raiz do
repositório, como `experiments/src/experiments/udv/calibrate_threshold.py`, e incluem os arquivos da
biblioteca de que um resultado depende (`bookworm/src/bookworm/transcript/*.py`,
`bookworm/src/bookworm/udv/quotes.py`). Para achar o commit de um hash antigo, o caminho a usar no
histórico da tag é o antigo, `challenge/utils/<nome>.py` (ver `experiments/README.md`, "Proveniência e
caminhos antigos").

## Arquivos que saíram antes desta versão

| caminho citado | onde está |
|---|---|
| `../backup/*.py` (`udv_pipeline_2026-09-*.py`, `build_udvs_2026-09-*.py`, `verify_udvs_2026-09-21.py`) | na tag `research-2026-09-28`, em `backup/` |
| `utils/generate_udv_manual_review.py`, `udv_manual_review.json`, `udv_manual_review_template.json` | na tag; eram a revisão exploratória do protótipo TF-IDF, não cega e sem anotador independente, substituída pela validação humana |
| arquivos listados em `experiments/artifacts/MANIFEST_heavy.tsv` | fora do Git, na tag; `experiments/README.md` mostra como restaurá-los nos caminhos desta versão |
| `challenge/artifacts/...` "no checkout principal" (versões anteriores do relatório e de `docs/pipeline.md`) | o mesmo caminho sob `experiments/artifacts/`; os dois ramos foram unidos |
