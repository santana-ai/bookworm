# experiments

Projeto `uv` onde rodam os experimentos do bookworm sobre o dataset **PublicHearingBR**: construção das
Unidades Deliberativas Verificáveis (UDV), benchmarks, calibrações, experimentos de recuperação e de
confiança, validação humana e perfis de ator. A biblioteca reutilizável fica em `../bookworm/`; este
diretório guarda o pacote `experiments` (`src/experiments/`), as configurações, os artefatos versionados e os notebooks exploratórios.

Este README é o guia de reprodução. A rodada de UDV de referência é `udv_v2` (janelas de duas sentenças,
citação inteira, corte de cosseno 0,50); `udv_v0` e `udv_v1` ficam como histórico e como base de
comparação, e toda etapa depois da UDV foi refeita ou conferida sobre `udv_v2`
(`artifacts/udv/udv_v2_downstream_report.json`). Os resultados e a origem de cada número estão nos
documentos abaixo.

| documento | conteúdo |
|---|---|
| [`docs/report.md`](../docs/report.md) | relatório completo: todos os experimentos, métricas, resultados, limitações e referências |
| [`docs/pipeline.md`](../docs/pipeline.md) | o que cada experimento decidiu e o pipeline recomendado, `udv_v2` |
| [`docs/methodology/udv.md`](../docs/methodology/udv.md) | o registro UDV e as regras de construção (resolução de pessoa, sentenças, citação, similaridade, offsets) |
| [`docs/methodology/confidence.md`](../docs/methodology/confidence.md) | levantamento da literatura de grounding e o experimento confidence_v2 |
| [`docs/methodology/hearing_actors.md`](../docs/methodology/hearing_actors.md) | falas por ator entre audiências |
| [`docs/methodology/actor_profiles.md`](../docs/methodology/actor_profiles.md), [`docs/methodology/actor_simulation.md`](../docs/methodology/actor_simulation.md) | perfis de ator e simulação a partir do perfil |
| [`docs/methodology/mlx_backend.md`](../docs/methodology/mlx_backend.md) | backend MLX usado na rodada de perfis e simulação |
| [`docs/validation/annotation_guide.md`](../docs/validation/annotation_guide.md) | guia do anotador da validação humana |
| [`docs/challenge/brief.md`](../docs/challenge/brief.md), [arXiv 2410.07495](https://arxiv.org/abs/2410.07495) | brief do desafio e artigo do dataset |
| [`docs/path_map.md`](../docs/path_map.md) | caminhos da versão de pesquisa e os desta versão |

## Dataset

O PublicHearingBR (Fernandes et al., 2024, arXiv 2410.07495) reúne 206 audiências públicas da Câmara
dos Deputados, publicado em
[unicamp-dl/PublicHearingBR](https://huggingface.co/datasets/unicamp-dl/PublicHearingBR). São dois
arquivos JSONL, com 206 registros cada, alinhados pelo campo `id`, vindos de extrações diferentes:

- **`PublicHearingBR_LDS.jsonl`** (Long Document Summarization): `id`, `materia` (a matéria da Agência
  Câmara), `transcricao` (a transcrição completa) e `metadados` (`assunto` e `envolvidos`, cada pessoa
  com `nome`, `cargo` e `opinioes`). As 2.203 opiniões foram extraídas **da matéria** por um LLM e
  corrigidas à mão. É o arquivo de onde saem as UDVs: registra o que a matéria atribui a cada pessoa.
- **`PublicHearingBR_NLI.jsonl`** (Natural Language Inference): `id` e `metadados_extraidos`, sem o
  texto da matéria nem da transcrição. As 4.238 opiniões foram geradas por um LLM lendo **a
  transcrição**, no experimento de sumarização do artigo, e não são as opiniões do LDS. Cada uma traz
  até quatro `chunks_proximos` (trechos da fala recuperados por similaridade) e
  `verificacao_alucinacao`, com o rótulo manual de um especialista e 12 julgamentos automáticos. O
  rótulo diz se a opinião é inferível daqueles quatro trechos (`retrieved_context_entailment`), não se
  ela é verdadeira ou se está na transcrição inteira. O arquivo é usado como benchmark do verificador,
  não como fonte de ligação.

A análise que sustenta essa distinção está em `notebooks/eda.ipynb` e resumida na seção 1 do relatório.

Download, de dentro de `experiments/`:

```bash
uv run python -m experiments.data.download
```

O script baixa para `dataset/` (ignorado pelo Git) a revisão do dataset usada em todas as rodadas,
`2f84a44bc34df483e25c987f0ff86caad0ab3433` (`--revision` troca a revisão, `--target-dir` o destino). Os
sha256 esperados são:

| arquivo | sha256 |
|---|---|
| `PublicHearingBR_LDS.jsonl` | `c4e392ab95ce22f6228eace16f15e9efe1c846724846b873dec420aec3d872c0` |
| `PublicHearingBR_NLI.jsonl` | `13408024c4776eff24f05ee6a56f9b1d33614524d19e90c0cce2fa310487a34e` |

As configurações que leem o LDS guardam esse hash e abortam se o arquivo for outro.

## Ambiente

- Python 3.12 e [`uv`](https://docs.astral.sh/uv/). `uv sync` instala o ambiente fixado em `uv.lock`.
  Todos os comandos rodam de dentro de `experiments/`, porque as configurações usam caminhos relativos a
  este diretório.
- Os scripts rodam como módulos do pacote: `uv run python -m experiments.<etapa>.<módulo> --help`
  lista as opções de cada um.
- A biblioteca é dependência editável deste projeto (`[tool.uv.sources]` aponta para `../bookworm`,
  com os extras `embeddings` e `profiles`), e a CLI roda no mesmo ambiente: `uv run bookworm --help`.
  O lock deste projeto fixa torch 2.13.0, sentence-transformers 5.6.1 e transformers 5.14.1, as
  versões de todas as rodadas de experimento. `udv_v2` foi construída no ambiente da biblioteca, com
  torch 2.14.0 e sentence-transformers 6.1.0 (relatório, seção 9.1); reconstruída neste ambiente a
  partir do cache de embeddings, `artifacts/udv/udv_v2.jsonl` sai igual byte a byte.
- O backend MLX dos perfis de ator entra só no ambiente da execução:
  `uv run --with "mlx-lm==0.31.3" python -m experiments.mlx.run --help`.

Hardware usado: os encoders, verificadores e tradutores rodaram num Apple M5 com 24 GB, em MPS com
float32, um processo com modelo por vez, com `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` e
`PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4` nas rodadas do confidence_v2. Os rerankers Laya do retrieval_v2
rodaram em CPU. A rodada completa de perfis e simulação rodou com MLX num Apple M4 Max com 128 GB
(`artifacts/mlx_runs/qwen38_27b/relatorio_rodada_completa.md`). O dispositivo é escolhido
automaticamente (`mps`, `cuda` ou `cpu`) e fica gravado no relatório de cada rodada; em outro
hardware os números podem diferir na última casa decimal.

As rodadas com modelo usaram `HF_HUB_OFFLINE=1` depois do primeiro download. Os subcomandos `fetch` de
`experiments.verifier.translation` e `experiments.verifier.nli_experiments` baixam os modelos na revisão fixada; os testes
nunca acessam a rede.

### Modelos fixados

Cada configuração grava o id do modelo no Hugging Face e a revisão (commit). Os principais:

| papel | modelo | revisão |
|---|---|---|
| encoder de produção (UDV, E1) | `PORTULAN/serafim-335m-portuguese-pt-sentence-encoder` | `a01887015444f7599669c509447c5bdbce958916` |
| verificador primário (E3x) | `convaiinnovations/laya`, pasta `multilingual` e raiz | `aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8` |
| tradução pt para en | `facebook/nllb-200-distilled-600M` | `f8d333a098d19b4fd9a8b18f94170487ad3f821d` |
| tradução, segunda condição | `facebook/m2m100_418M` | `55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636` |
| NLI de comparação | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | `b5113eb38ab63efdd7f280f8c144ea8b13f978ce` |
| perfis e simulação (MLX) | `mlx-community/Qwen3.8-27B-8bit` | sem revisão fixada em `configs/mlx.yaml` |

A lista completa, com licenças e referências, está no relatório, seções 11.2 e 11.3.

## Organização do pacote

Os módulos ficam em `src/experiments/`, um subpacote por etapa. As regras de transcrição, citação,
UDV, splits e falas por ator vivem só na biblioteca; os módulos daqui as importam.

| subpacote | scripts (rodam com `python -m`) | módulos de apoio |
|---|---|---|
| `data` | `download`, `quote_benchmark`, `nli_benchmark` | `legacy_splits` (funções de data e de split do notebook de splits e do teste de paridade da biblioteca) |
| `udv` | `calibrate_threshold`, `calibrate_v2`, `v2_analysis`, `v2_downstream`, `fuzzy_matching`, `measure_turn_segmentation`, `measure_quote_patterns`, `measure_case_insensitive_quotes` | |
| `retrieval` (E1, E2, retrieval_v2) | `experiments` | `data`, `models`, `store` |
| `verifier` (E3, E3x, E5, confidence_v2) | `nli_experiments`, `nli_exploration`, `translation`, `udv_verifier`, `confidence_policies`, `confidence_v2` | `decision_models`, `decision_scoring`, `grounding_scorers`, `grounding_models` |
| `validation` | `generate_sample`, `precision_report`, `fuzzy_review_precision` | |
| `actors` | `measure_hearing_actors`, `build_speeches`, `filter_speeches`, `generate_profiles`, `evaluate_simulation`, `simulate` | `simulation` |
| `mlx` | `run`, `verify_backend`, `benchmark` | `backend`, `settings` |
| `common` | | `transcript` (visões em dicionário das regras de transcrição e citação da biblioteca), `udv_run` (configuração de rodada e cache do encoder), `provenance` (hashes do código nos relatórios), `stats`, `cache_lock`, `hub_offline` |

A construção e a verificação de UDVs e de splits são comandos da biblioteca: `uv run bookworm
build-udvs`, `verify-udvs`, `build-splits` e `verify-splits`.

`configs/` tem um TOML por etapa (a declaração de cada experimento, escrita antes dos resultados, fica
no próprio TOML), `prompts/` os prompts dos perfis e da simulação, `notebooks/` os notebooks
exploratórios, `scripts/` a execução encadeada dos perfis e da simulação, e `tests/` os testes do
projeto.

## Etapas, em ordem

Cada etapa lê as saídas das anteriores. A coluna "saída versionada" lista o que está no Git; as saídas
pesadas por item estão na seção seguinte.

| # | etapa | comando | config | saída versionada |
|---|---|---|---|---|
| 1 | download | `uv run python -m experiments.data.download` | revisão fixada no script | `dataset/` (não versionado) |
| 2 | splits temporais | `uv run bookworm build-splits` e `uv run bookworm verify-splits` | `configs/splits.toml` | `artifacts/splits/temporal_v1.json`, `temporal_v1_report.json` |
| 3 | benchmarks B1 e B2 | `uv run python -m experiments.data.quote_benchmark` e `uv run python -m experiments.data.nli_benchmark` | `configs/quote_benchmark.toml`, `configs/nli_benchmark.toml` | `artifacts/benchmarks/masked_quotes_v1*`, `nli_v1*` |
| 4 | corte de sentença (`threshold_v1`) | `uv run python -m experiments.udv.calibrate_threshold` | `configs/udv.toml` | `artifacts/calibration/threshold_v1.json`, `threshold_v1_queries.jsonl` |
| 5 | UDV v1 (histórico e base de comparação) | `uv run bookworm build-udvs --run-name udv_v1` e `uv run bookworm verify-udvs --run-name udv_v1` | `configs/udv.toml` | `artifacts/udv/udv_v1.jsonl`, `udv_v1_coverage.json` |
| 6 | E1 e E2 (`retrieval_v1`) | `uv run python -m experiments.retrieval.experiments queue --run-name retrieval_v1` (`--dry-run` lista os passos) | `configs/retrieval_experiments.toml` | `artifacts/experiments/retrieval/retrieval_v1/retrieval_v1_report.json` e `runs/` |
| 7 | retrieval_v2 (Laya como reranqueador) | `uv run python -m experiments.retrieval.experiments run --run-name retrieval_v2 --retrievers serafim_335m,rerank_bge,rerank_mmarco,rerank_laya_p4,rerank_laya_p3 --units sentence --device cpu`, depois `summarize --run-name retrieval_v2` | `configs/retrieval_experiments.toml`, `[declarations.retrieval_v2]` | `artifacts/experiments/retrieval/retrieval_v2/retrieval_v2_report.json` e `runs/` |
| 8 | tradução | `uv run python -m experiments.verifier.translation fetch`, `translate --run-name translation_v1`, `report --run-name translation_v1_report`; o mesmo com `--model m2m100` e os nomes `translation_v1_m2m100` e `translation_v1_m2m100_report` | `configs/translation.toml` | `artifacts/experiments/translation/*/report.json`, `plan.json` |
| 9 | E3 v1 e v2 (verificadores NLI) | `uv run python -m experiments.verifier.nli_experiments fetch`, `score --run-name nli_verifier_v1 --scorers <...>`, `evaluate --run-name nli_verifier_v1`; o mesmo para `nli_verifier_v2` | `configs/nli_verifier.toml` (`[declarations.<rodada>]` lista os avaliadores) | `artifacts/experiments/nli_verifier/nli_verifier_v1/`, `nli_verifier_v2/` (relatórios e `scores/*_report.json`) |
| 10 | E3x (verificador aprendido) | `uv run python -m experiments.verifier.nli_exploration --config configs/nli_verifier_exploration_v2.toml explore`, depois `confirm`; teste final: `verifier.translation translate --final-test --splits test` nos dois tradutores, `verifier.nli_experiments score --run-name final_test_e3x --final-test --scorers <...>` e `verifier.nli_exploration --config configs/nli_verifier_exploration_v2.toml final-test` | `configs/nli_verifier_exploration_v2.toml` | `artifacts/experiments/nli_verifier_exploration/e3x_v2/` (`selection.json`, `confirmation.*`, `final_test.json`) |
| 11 | E5 (`confidence_v1`) | `uv run python -m experiments.verifier.confidence_policies collect --run-name confidence_v1`, `pairs --run-name confidence_v1`, `verifier.nli_experiments pairs --run-name confidence_v1 --input artifacts/experiments/confidence/confidence_v1/nli/pairs_input.jsonl --output-dir artifacts/experiments/confidence/confidence_v1/nli`, `verifier.confidence_policies evaluate --run-name confidence_v1` | `configs/confidence_policies.toml` | `artifacts/experiments/confidence/confidence_v1/confidence_v1_report.json` |
| 12 | E6 (`fuzzy_v1`) | `uv run python -m experiments.udv.fuzzy_matching`; precisão, depois de julgar as planilhas: `uv run python -m experiments.validation.fuzzy_review_precision` | `configs/fuzzy_matching.toml` | `artifacts/experiments/fuzzy/` |
| 13 | camada do verificador sobre `udv_v1` (histórico) | `uv run python -m experiments.verifier.udv_verifier --config configs/udv_verifier.toml translate`, `score`, `apply` | `configs/udv_verifier.toml` | `artifacts/udv/udv_v1_verifier.jsonl`, `udv_v1_verifier_report.json` |
| 14 | cortes de `udv_v2` | `uv run python -m experiments.udv.calibrate_v2 cosine`; `uv run python -m experiments.verifier.udv_verifier --config configs/udv_v2_calibration_verifier.toml translate`, `score`, `apply`; `uv run python -m experiments.udv.calibrate_v2 verifier` | `configs/udv_v2.toml`, `configs/udv_v2_calibration_verifier.toml` | `artifacts/calibration/threshold_v2*`, `udv_verifier_threshold_v1.json` |
| 15 | UDV v2 | `uv run bookworm build-udvs --config configs/udv_v2.toml --run-name udv_v2` e `uv run bookworm verify-udvs --config configs/udv_v2.toml --run-name udv_v2 --baseline artifacts/udv/udv_v1.jsonl` | `configs/udv_v2.toml` | `artifacts/udv/udv_v2.jsonl`, `udv_v2_coverage.json`, `udv_v2_verify.json` |
| 16 | verificador sobre `udv_v2` | `uv run python -m experiments.verifier.udv_verifier --config configs/udv_v2_verifier.toml translate`, `score`, `apply` | `configs/udv_v2_verifier.toml` | `artifacts/udv/udv_v2_verifier.jsonl`, `udv_v2_verifier_report.json` |
| 17 | análise de `udv_v2` | `uv run python -m experiments.udv.v2_analysis analyze` | caminhos de entrada em `DEFAULT_PATHS` do script | `artifacts/udv/udv_v2_analysis.json`, `udv_v2_annotation_plan.json` |
| 18 | confidence_v2 | `uv run python -m experiments.verifier.confidence_v2 translate`, `smoke`, `laya-smoke`, `decide`, `score`, `laya-score`, `evaluate-ea` | `configs/confidence_v2.toml` | `artifacts/experiments/confidence_v2/` (`ea_report.json`, `smoke/`, `scores/*_report.json`) |
| 19 | amostra de validação humana | `uv run python -m experiments.validation.generate_sample --run-name udv_v1 --final-test`; a planilha de reanotação sai com `--stage repeat` depois da primeira rodada completa (`--dry-run` grava fora do repositório) | `configs/validation_sample.toml` | `artifacts/validation/human_validation_v1_udv_v1/` |
| 20 | pontuação da validação humana | `uv run python -m experiments.udv.v2_analysis score-annotation --final-test --annotation artifacts/validation/human_validation_v1_udv_v1/annotation.csv --supplement-dir artifacts/validation/human_validation_v1_udv_v2_supplement --output artifacts/udv/udv_v2_precision_final.json`; a planilha suplementar de `udv_v2` (26 linhas, só os itens `moved`; os `same` e `superset` herdam o rótulo pela seção `[udv_v2_supplement]`) foi gerada por `uv run python -m experiments.udv.v2_analysis supplement-sheet --final-test`. As duas planilhas estão julgadas; `existe_trecho_melhor` ficou vazio, e por isso `experiments.validation.precision_report` recusa a planilha e os números finais vêm de `score-annotation`. Resultado: os dois critérios declarados falham em `udv_v1` e em `udv_v2` (`docs/report.md`, seções 7.2 e 7.4) | `configs/validation_sample.toml` | `artifacts/udv/udv_v2_precision_final.json` (final); `udv_v2_precision_interim_*.json` são leituras intermediárias de planilhas incompletas, guardadas como histórico |
| 21 | falas por ator | `uv run python -m experiments.actors.measure_hearing_actors`, `uv run python -m experiments.actors.build_speeches`, `uv run python -m experiments.actors.filter_speeches --output artifacts/cache/hearing_actors/actors_multi_hearing_train.jsonl` | `configs/hearing_actors.toml`, `configs/actor_profiles.toml` | `artifacts/hearing_actors/`, `artifacts/actor_profiles/train_speeches_stats.json` |
| 22 | perfis de ator e simulação (MLX) | `uv run --with "mlx-lm==0.31.3" python -m experiments.mlx.run all --model qwen38_27b --all-actors` (`smoke`, `profiles`, `evaluate` e `simulate` rodam uma etapa); sobre `udv_v2`, o mesmo com `--settings configs/mlx_udv_v2.yaml` (não rodado nesta versão) | `configs/mlx.yaml`, `configs/mlx_udv_v2.yaml`, `configs/actor_profiles.toml`, `configs/actor_simulation.toml` | `artifacts/mlx_runs/qwen38_27b/` (rodada `udv_v1`) |
| 23 | conferência de perfis contra UDVs | `uv run bookworm validate-profiles --config configs/profile_validation.toml`; para `udv_v1`, `--config configs/profile_validation_udv_v1.toml` | `configs/profile_validation.toml`, `configs/profile_validation_udv_v1.toml` | `artifacts/profile_validation/` |
| 24 | etapas depois da UDV sobre `udv_v2`, lado a lado com `udv_v1` | `uv run python -m experiments.udv.v2_downstream` | caminhos no próprio script | `artifacts/udv/udv_v2_downstream_report.json` |
| 25 | exportação da demo web | `uv run bookworm export-site --config configs/udv_v2.toml --run-name udv_v2 ...` (argumentos em `../bookworm/web/README.md`) | `configs/udv_v2.toml`, `configs/hearing_actors.toml` | `artifacts/web/export_site_udv_v2.json` (resumo); os dados vão para `../bookworm/web/app/data/`, fora do Git |

Notas sobre as etapas:

- As etapas 1 a 5 e 21 rodam em minutos em CPU ou MPS (a rodada `udv_v1` leva 23,5 s com o cache de
  embeddings pronto). As etapas com modelo levam horas: a calibração `threshold_v2` levou 990 s, a
  construção de `udv_v2` 662 s, e cada avaliador do confidence_v2 entre 20 min e 4 h
  (`docs/methodology/confidence.md`, "O que rodou"); a rodada de perfis e simulação levou cerca de 13 h.
- `--final-test` é obrigatório em todo comando que lê o split de teste. Nenhuma escolha de método usa
  o teste; o gerador da amostra humana recusa uma rodada cujo corte tenha sido calibrado com audiências
  de teste.
- As medições históricas das ADRs 0002 e 0003 (`measure_turn_segmentation`, `measure_quote_patterns`,
  `measure_case_insensitive_quotes`) comparam com rodadas antigas; `measure_turn_segmentation` exige
  `--legacy-pipeline` apontando para o `udv_pipeline.py` antigo, que está na tag `research-2026-09-28`
  (`git show research-2026-09-28:backup/udv_pipeline_2026-09-21.py`).
- Nenhum script preenche julgamento humano. `validation.generate_sample` e `udv.fuzzy_matching`
  gravam planilhas com os campos de julgamento vazios; o preenchimento é manual, seguindo
  `docs/validation/annotation_guide.md`.
- Resultado da validação humana (etapa 20, `artifacts/udv/udv_v2_precision_final.json`, status
  `final`): em `quote_found`, 25 `correta`, 6 `parcial` e 4 `incorreta` em 35, precisão estrita 0,7143
  [0,5495; 0,8367], igual em `udv_v1` e `udv_v2`; o critério (limite inferior de 0,90) falha. Em
  `semantic_match_high`, a precisão tolerante é 0,7385 [0,6205; 0,8298] em `udv_v1` (65 itens) e 0,8254
  [0,7138; 0,8996] em `udv_v2` (63 itens); o critério (limite inferior de 0,75) falha nas duas. Os
  números de `udv_v2` supõem que os 78 itens `superset` mantêm o rótulo de `udv_v1`. No `speaker_check`,
  5 das 6 pessoas sem evidência falaram na transcrição.

## Artefatos versionados e artefatos pesados

Tudo em `artifacts/` é versionado, com duas exceções:

- `artifacts/cache/` (ignorado): embeddings, traduções, respostas de modelo e escores por par. Cada
  comando refaz o que falta no cache e reutiliza o resto.
- As saídas por item das rodadas de experimento (escores, consultas, features, predições e pares, 369
  arquivos) saíram do Git. Estão listadas em [`artifacts/MANIFEST_heavy.tsv`](artifacts/MANIFEST_heavy.tsv),
  com caminho, tamanho, sha256 e o comando que regenera cada arquivo. Os relatórios agregados que
  citam esses arquivos continuam versionados.

Os comandos abaixo leem arquivos pesados de etapas anteriores e só rodam depois que eles existirem,
regenerados pelo comando da coluna `regenerate` do manifesto ou restaurados da tag:

| comando | lê |
|---|---|
| `experiments.verifier.nli_exploration` (`explore`, `confirm`, `final-test`) | `scores/<avaliador>_<split>.jsonl` de `nli_verifier_v1`, `nli_verifier_v2` e `final_test_e3x` |
| `experiments.verifier.udv_verifier apply` (as três configurações) | os arquivos de escore de treino de `nli_verifier_v1` e `nli_verifier_v2` que o E3x leu, usados no reajuste do primário, e os de `artifacts/experiments/nli_verifier/<rodada>/scores/` gravados por `score` |
| `experiments.udv.calibrate_v2 verifier` | `artifacts/calibration/threshold_v2_verifier_scores.jsonl`, gravado por `udv_verifier apply` |
| `experiments.verifier.confidence_v2` (`evaluate-ea`, reaproveitamento das respostas do Laya) | `scores/` de `nli_verifier_v1`, `nli_verifier_v2` e `udv_v1_verifier`, e `artifacts/experiments/confidence_v2/scores/` |
| `experiments.verifier.confidence_policies evaluate` | `features/` e `nli/pairs/` de `artifacts/experiments/confidence/confidence_v1/` |

Para restaurar todos os arquivos pesados sem recalcular, a partir da raiz do repositório:

```bash
tail -n +2 experiments/artifacts/MANIFEST_heavy.tsv | cut -f1 | while read -r p; do
  mkdir -p "$(dirname "experiments/${p#challenge/}")"
  git show "research-2026-09-28:$p" > "experiments/${p#challenge/}"
done
```

O manifesto registra os caminhos e os comandos da versão de pesquisa (`challenge/...`,
`python -m utils.<nome>`); [`docs/path_map.md`](../docs/path_map.md) dá o caminho e o módulo de cada
um nesta versão.

Os arquivos restaurados ficam ignorados pelo Git e conferem com o sha256 do manifesto.

## Proveniência e caminhos antigos

Os relatórios JSON gravam o sha256 do código (`code`), das entradas (`inputs`) e da configuração
(`config`) de cada rodada, e não a linha de comando. Os módulos desta versão receberam
formatação e anotações de tipo depois das rodadas, sem mudança de comportamento, e por isso os hashes
gravados não coincidem com os arquivos atuais. Os relatórios gravados antes desta versão registram
os caminhos `utils/<nome>.py`; os gravados a partir dela, o caminho a partir da raiz do repositório. O código exato de uma rodada está no histórico da tag
`research-2026-09-28`; para achar o commit cujo arquivo tem o hash gravado:

```bash
f=challenge/utils/retrieval_models.py; h=<sha256 gravado>
for c in $(git rev-list research-2026-09-28 -- "$f"); do
  [ "$(git show "${c}:${f}" | shasum -a 256 | cut -d' ' -f1)" = "$h" ] && echo "$c" && break
done
```

Os caminhos citados em artefatos, notebooks, ADRs e no manifesto que mudaram ou saíram desta versão
estão em [`docs/path_map.md`](../docs/path_map.md).

## Testes e qualidade de código

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Os testes usam dados sintéticos pequenos, não leem o dataset nem os arquivos pesados, e uma fixture
bloqueia qualquer conexão de rede. O ruff usa as regras E, F, I, B, UP e SIM, e o mypy confere
`src/` com `disallow_untyped_defs` (configuração em `pyproject.toml`).
