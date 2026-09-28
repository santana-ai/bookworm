# experiments

Projeto `uv` onde rodam os experimentos do bookworm sobre o dataset
[PublicHearingBR](https://huggingface.co/datasets/unicamp-dl/PublicHearingBR): construção das UDVs,
benchmarks, calibrações, recuperação, confiança, validação humana e perfis de ator. As regras de
transcrição, citação, UDV, splits e falas por ator vivem só na biblioteca [`bookworm`](../bookworm/README.md);
este projeto as importa e guarda o que é próprio da pesquisa: scripts, configurações, artefatos
versionados e notebooks.

A rodada de referência é `udv_v2`. `udv_v0` e `udv_v1` ficam como histórico e base de comparação, e
toda etapa depois da UDV foi refeita ou conferida sobre `udv_v2`
(`artifacts/udv/udv_v2_downstream_report.json`).

## Instalação

Requer Python 3.12 e [uv](https://docs.astral.sh/uv/). Todos os comandos rodam de dentro de
`experiments/`, porque as configurações usam caminhos relativos a esta pasta.

```bash
cd experiments
uv sync                                   # instala também a biblioteca, editável, com os extras
uv run python -m experiments.data.download
```

O download grava os dois arquivos JSONL em `dataset/` (ignorado pelo Git), numa revisão fixada do
dataset, e as configurações que leem o LDS conferem o sha256 do arquivo antes de rodar. O que cada
arquivo contém está no [relatório, seção 1](../docs/report.md#1-dataset-e-tarefa).

## Comandos principais

```bash
# confere a rodada publicada recalculando cada UDV a partir do LDS (CPU, sem modelo)
uv run bookworm verify-udvs --config configs/udv_v2.toml --run-name udv_v2

# reconstrói udv_v2 com outro nome, sem tocar a rodada publicada
# (precisa do encoder Serafim ou do cache de embeddings)
uv run bookworm build-udvs --config configs/udv_v2.toml --run-name udv_v2_rebuild

# análise de udv_v2 e precisão da validação humana
uv run python -m experiments.udv.v2_analysis analyze
uv run python -m experiments.udv.v2_analysis score-annotation --final-test \
  --annotation artifacts/validation/human_validation_v1_udv_v1/annotation.csv \
  --supplement-dir artifacts/validation/human_validation_v1_udv_v2_supplement \
  --output artifacts/udv/udv_v2_precision_final.json

# qualquer script
uv run python -m experiments.<etapa>.<módulo> --help
```

A sequência completa, com as 25 etapas na ordem em que cada uma lê as saídas da anterior, a
configuração e a saída versionada de cada uma, está no [guia de reprodução](../docs/reproduce.md).

## Organização

| pasta | conteúdo |
|---|---|
| `src/experiments/` | o pacote, um subpacote por etapa (tabela abaixo) |
| `configs/` | um TOML por experimento, com a declaração escrita antes dos resultados |
| `artifacts/` | saídas versionadas; `artifacts/cache/` é ignorado ([artefatos pesados](../docs/reproduce.md#artefatos-pesados)) |
| `prompts/` | prompts dos perfis e da simulação de atores |
| `notebooks/` | análise exploratória (`eda`), splits, UDV e perfis de ator |
| `scripts/` | execução encadeada dos perfis e da simulação |
| `tests/` | testes do projeto, com dados sintéticos |
| `dataset/` | o PublicHearingBR baixado, fora do Git |

| subpacote | scripts (`python -m`) | módulos de apoio |
|---|---|---|
| `data` | `download`, `quote_benchmark`, `nli_benchmark` | `legacy_splits` |
| `udv` | `calibrate_threshold`, `calibrate_v2`, `v2_analysis`, `v2_downstream`, `fuzzy_matching`, `measure_*` | |
| `retrieval` (E1, E2, retrieval_v2) | `experiments` | `data`, `models`, `store` |
| `verifier` (E3, E3x, E5, confidence_v2) | `nli_experiments`, `nli_exploration`, `translation`, `udv_verifier`, `confidence_policies`, `confidence_v2` | `decision_models`, `decision_scoring`, `grounding_scorers`, `grounding_models` |
| `validation` | `generate_sample`, `precision_report`, `fuzzy_review_precision` | |
| `actors` | `measure_hearing_actors`, `build_speeches`, `filter_speeches`, `generate_profiles`, `evaluate_simulation`, `simulate` | `simulation` |
| `mlx` | `run`, `verify_backend`, `benchmark` | `backend`, `settings` |
| `common` | | `transcript`, `udv_run`, `provenance`, `stats`, `cache_lock`, `hub_offline` |

A construção e a verificação de UDVs e de splits são comandos da biblioteca (`uv run bookworm
build-udvs`, `verify-udvs`, `build-splits`, `verify-splits`). `common.transcript` e `common.udv_run`
dão a mesma interface dos scripts antigos em dicionários e delegam à biblioteca; `data.legacy_splits`
guarda as funções de data e de split que o notebook de splits e o teste de paridade chamam.

## Regras que valem para todo experimento

- **O teste não escolhe nada.** Todo comando que lê o split de teste exige `--final-test`; cortes,
  métodos e hiperparâmetros são escolhidos no treino e na validação.
- **Declaração antes do resultado.** Cada TOML de `configs/` registra o experimento antes da rodada;
  mudanças feitas depois de ver resultados ficam no próprio arquivo, com data e motivo.
- **Nenhum script preenche julgamento humano.** As planilhas saem com os campos de julgamento vazios,
  e o preenchimento é manual, seguindo o [guia do anotador](../docs/validation/annotation_guide.md).
- **Proveniência.** Cada relatório JSON grava o sha256 do código, das entradas e da configuração. Como
  achar o commit de um hash antigo está em
  [proveniência e caminhos antigos](../docs/reproduce.md#proveniência-e-caminhos-antigos).

## Testes e qualidade de código

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Os testes usam dados sintéticos pequenos, não leem o dataset nem os arquivos pesados, e uma fixture
bloqueia qualquer conexão de rede. O ruff usa as regras E, F, I, B, UP e SIM, e o mypy confere `src/`
com `disallow_untyped_defs` (configuração em `pyproject.toml`).

## Onde ler mais

| documento | para quê |
|---|---|
| [`docs/reproduce.md`](../docs/reproduce.md) | todas as etapas, ambiente, hardware, modelos fixados e artefatos pesados |
| [`docs/pipeline.md`](../docs/pipeline.md) | o que cada experimento decidiu e o pipeline `udv_v2` |
| [`docs/report.md`](../docs/report.md) | todos os números, com a origem de cada um, limitações e referências |
| [`docs/methodology/`](../docs/README.md#metodologia) | UDV, confiança, atores, perfis, simulação e backend MLX |
| [`docs/path_map.md`](../docs/path_map.md) | caminhos da versão de pesquisa e os desta versão |
