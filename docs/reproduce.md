# Guia de reprodução

Este guia lista, em ordem, os comandos que regeneram cada artefato versionado dos experimentos. Cada
etapa lê as saídas das anteriores. Todos os comandos rodam de dentro de `experiments/`, porque as
configurações usam caminhos relativos a essa pasta; os caminhos abaixo também são relativos a ela.

A descrição de cada experimento, com os números e a origem de cada um, está em
[`report.md`](report.md); o que cada experimento decidiu, em [`pipeline.md`](pipeline.md). Para só
conferir a rodada publicada em poucos minutos, sem modelo, basta o [início rápido do
README](../README.md#rode-em-3-comandos).

## Sumário

- [Ambiente](#ambiente)
- [Etapas em ordem](#etapas-em-ordem)
  - [Dados, splits e benchmarks (1 a 3)](#dados-splits-e-benchmarks-1-a-3)
  - [UDV v1 e o corte de sentença (4 e 5)](#udv-v1-e-o-corte-de-sentença-4-e-5)
  - [Trabalho 1: recuperação (6 e 7)](#trabalho-1-recuperação-6-e-7)
  - [Trabalho 2: verificador e confiança (8 a 13)](#trabalho-2-verificador-e-confiança-8-a-13)
  - [UDV v2 (14 a 18)](#udv-v2-14-a-18)
  - [Validação humana (19 e 20)](#validação-humana-19-e-20)
  - [Atores, perfis e simulação (21 a 23)](#atores-perfis-e-simulação-21-a-23)
  - [Etapas depois da UDV e demo (24 e 25)](#etapas-depois-da-udv-e-demo-24-e-25)
- [Tempo de cada etapa](#tempo-de-cada-etapa)
- [Artefatos pesados](#artefatos-pesados)
- [Proveniência e caminhos antigos](#proveniência-e-caminhos-antigos)

## Ambiente

- Python 3.12 e [`uv`](https://docs.astral.sh/uv/). `uv sync` instala o ambiente fixado em
  `uv.lock`. A biblioteca `bookworm` é dependência editável do projeto `experiments`, com os extras
  `embeddings` e `profiles`, e a CLI `bookworm` roda no mesmo ambiente.
- O lock de `experiments/` fixa torch 2.13.0, sentence-transformers 5.6.1 e transformers 5.14.1, as
  versões de todas as rodadas de experimento. `udv_v2` foi construída no ambiente da biblioteca, com
  torch 2.14.0 e sentence-transformers 6.1.0 ([relatório, seção 9.1](report.md#91-ambiente));
  reconstruída no ambiente de `experiments/` a partir do cache de embeddings,
  `artifacts/udv/udv_v2.jsonl` sai igual byte a byte.
- O backend MLX dos perfis de ator entra só no ambiente da execução:
  `uv run --with "mlx-lm==0.31.3" python -m experiments.mlx.run --help`.
- As rodadas com modelo usaram `HF_HUB_OFFLINE=1` depois do primeiro download. Os subcomandos `fetch`
  de `experiments.verifier.translation` e `experiments.verifier.nli_experiments` baixam os modelos na
  revisão fixada; os testes nunca acessam a rede.

**Hardware.** Encoders, verificadores e tradutores rodaram num Apple M5 com 24 GB, em MPS com float32,
um processo com modelo por vez, com `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` e
`PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4` nas rodadas do confidence_v2. Os rerankers Laya do retrieval_v2
rodaram em CPU. A rodada completa de perfis e simulação rodou com MLX num Apple M4 Max com 128 GB
(`artifacts/mlx_runs/qwen38_27b/relatorio_rodada_completa.md`). O dispositivo é escolhido
automaticamente (`mps`, `cuda` ou `cpu`) e fica gravado no relatório de cada rodada; em outro hardware
os números podem diferir na última casa decimal.

**Modelos fixados.** Cada configuração grava o id do modelo no Hugging Face e a revisão (commit). Os
principais:

| papel | modelo | revisão |
|---|---|---|
| encoder de produção (UDV, E1) | `PORTULAN/serafim-335m-portuguese-pt-sentence-encoder` | `a01887015444f7599669c509447c5bdbce958916` |
| verificador primário (E3x) | `convaiinnovations/laya`, pasta `multilingual` e raiz | `aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8` |
| tradução pt para en | `facebook/nllb-200-distilled-600M` | `f8d333a098d19b4fd9a8b18f94170487ad3f821d` |
| tradução, segunda condição | `facebook/m2m100_418M` | `55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636` |
| NLI de comparação | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | `b5113eb38ab63efdd7f280f8c144ea8b13f978ce` |
| perfis e simulação (MLX) | `mlx-community/Qwen3.8-27B-8bit` | sem revisão fixada em `configs/mlx.yaml` |

A lista completa, com licenças e referências, está no [relatório, seções 11.2 e
11.3](report.md#112-modelos-usados).

**Sementes.** Cada experimento declara sua semente e o número de réplicas do bootstrap
([relatório, seção 9.2](report.md#92-sementes-e-réplicas)); os splits, as calibrações e a amostra de
validação usam semente 42.

## Etapas em ordem

Cada etapa lista o comando, a configuração e a saída versionada no Git. Toda configuração de
experimento é um TOML de `configs/`, com a declaração do experimento escrita antes dos resultados.
`uv run python -m <módulo> --help` lista as opções de cada script.

> [!IMPORTANT]
> `--final-test` é obrigatório em todo comando que lê o split de teste. Nenhuma escolha de método usa
> o teste, e o gerador da amostra humana recusa uma rodada cujo corte tenha sido calibrado com
> audiências de teste.

### Dados, splits e benchmarks (1 a 3)

```bash
# 1. download do dataset para dataset/ (não versionado), na revisão fixada no script
uv run python -m experiments.data.download

# 2. splits temporais (configs/splits.toml)
uv run bookworm build-splits
uv run bookworm verify-splits

# 3. benchmarks B1 (citações mascaradas) e B2 (NLI)
uv run python -m experiments.data.quote_benchmark
uv run python -m experiments.data.nli_benchmark
```

| etapa | config | saída versionada |
|---|---|---|
| 1 download | revisão `2f84a44bc34df483e25c987f0ff86caad0ab3433` no script | `dataset/` (não versionado) |
| 2 splits | `configs/splits.toml` | `artifacts/splits/temporal_v1.json`, `temporal_v1_report.json` |
| 3 benchmarks | `configs/quote_benchmark.toml`, `configs/nli_benchmark.toml` | `artifacts/benchmarks/masked_quotes_v1*`, `nli_v1*` |

O download aceita `--revision` e `--target-dir`. Os sha256 esperados:

| arquivo | sha256 |
|---|---|
| `PublicHearingBR_LDS.jsonl` | `c4e392ab95ce22f6228eace16f15e9efe1c846724846b873dec420aec3d872c0` |
| `PublicHearingBR_NLI.jsonl` | `13408024c4776eff24f05ee6a56f9b1d33614524d19e90c0cce2fa310487a34e` |

As configurações que leem o LDS guardam esse hash e abortam se o arquivo for outro. O que cada arquivo
contém, e por que o NLI não serve de fonte de ligação, está no [relatório, seção
1](report.md#1-dataset-e-tarefa).

### UDV v1 e o corte de sentença (4 e 5)

```bash
# 4. corte de cosseno para sentenças (threshold_v1)
uv run python -m experiments.udv.calibrate_threshold

# 5. UDV v1 (histórico e base de comparação)
uv run bookworm build-udvs --run-name udv_v1
uv run bookworm verify-udvs --run-name udv_v1
```

| etapa | config | saída versionada |
|---|---|---|
| 4 `threshold_v1` | `configs/udv.toml` | `artifacts/calibration/threshold_v1.json`, `threshold_v1_queries.jsonl` |
| 5 `udv_v1` | `configs/udv.toml` | `artifacts/udv/udv_v1.jsonl`, `udv_v1_coverage.json` |

As medições históricas das ADRs 0002 e 0003 (`experiments.udv.measure_turn_segmentation`,
`measure_quote_patterns`, `measure_case_insensitive_quotes`) comparam com rodadas antigas;
`measure_turn_segmentation` exige `--legacy-pipeline` apontando para o `udv_pipeline.py` antigo, que
está na tag `research-2026-09-28`
(`git show research-2026-09-28:backup/udv_pipeline_2026-09-21.py`).

### Trabalho 1: recuperação (6 e 7)

```bash
# 6. E1 e E2 (retrieval_v1); --dry-run lista os passos
uv run python -m experiments.retrieval.experiments queue --run-name retrieval_v1

# 7. retrieval_v2 (Laya como reranqueador)
uv run python -m experiments.retrieval.experiments run --run-name retrieval_v2 \
  --retrievers serafim_335m,rerank_bge,rerank_mmarco,rerank_laya_p4,rerank_laya_p3 \
  --units sentence --device cpu
uv run python -m experiments.retrieval.experiments summarize --run-name retrieval_v2
```

| etapa | config | saída versionada |
|---|---|---|
| 6 `retrieval_v1` | `configs/retrieval_experiments.toml` | `artifacts/experiments/retrieval/retrieval_v1/retrieval_v1_report.json` e `runs/` |
| 7 `retrieval_v2` | idem, `[declarations.retrieval_v2]` | `artifacts/experiments/retrieval/retrieval_v2/retrieval_v2_report.json` e `runs/` |

### Trabalho 2: verificador e confiança (8 a 13)

```bash
# 8. tradução pt para en, NLLB e M2M100
uv run python -m experiments.verifier.translation fetch
uv run python -m experiments.verifier.translation translate --run-name translation_v1
uv run python -m experiments.verifier.translation report --run-name translation_v1_report
#    o mesmo com --model m2m100 e os nomes translation_v1_m2m100 e translation_v1_m2m100_report

# 9. E3 v1 e v2 (verificadores NLI); [declarations.<rodada>] lista os avaliadores
uv run python -m experiments.verifier.nli_experiments fetch
uv run python -m experiments.verifier.nli_experiments score --run-name nli_verifier_v1 --scorers <...>
uv run python -m experiments.verifier.nli_experiments evaluate --run-name nli_verifier_v1
#    o mesmo para nli_verifier_v2

# 10. E3x (verificador aprendido): seleção e confirmação
uv run python -m experiments.verifier.nli_exploration --config configs/nli_verifier_exploration_v2.toml explore
uv run python -m experiments.verifier.nli_exploration --config configs/nli_verifier_exploration_v2.toml confirm
#     teste final
uv run python -m experiments.verifier.translation translate --final-test --splits test   # nos dois tradutores
uv run python -m experiments.verifier.nli_experiments score --run-name final_test_e3x --final-test --scorers <...>
uv run python -m experiments.verifier.nli_exploration --config configs/nli_verifier_exploration_v2.toml final-test

# 11. E5 (confidence_v1)
uv run python -m experiments.verifier.confidence_policies collect --run-name confidence_v1
uv run python -m experiments.verifier.confidence_policies pairs --run-name confidence_v1
uv run python -m experiments.verifier.nli_experiments pairs --run-name confidence_v1 \
  --input artifacts/experiments/confidence/confidence_v1/nli/pairs_input.jsonl \
  --output-dir artifacts/experiments/confidence/confidence_v1/nli
uv run python -m experiments.verifier.confidence_policies evaluate --run-name confidence_v1

# 12. E6 (fuzzy_v1); a precisão só depois de julgar as planilhas
uv run python -m experiments.udv.fuzzy_matching
uv run python -m experiments.validation.fuzzy_review_precision

# 13. camada do verificador sobre udv_v1 (histórico)
uv run python -m experiments.verifier.udv_verifier --config configs/udv_verifier.toml translate
uv run python -m experiments.verifier.udv_verifier --config configs/udv_verifier.toml score
uv run python -m experiments.verifier.udv_verifier --config configs/udv_verifier.toml apply
```

| etapa | config | saída versionada |
|---|---|---|
| 8 tradução | `configs/translation.toml` | `artifacts/experiments/translation/*/report.json`, `plan.json` |
| 9 E3 | `configs/nli_verifier.toml` | `artifacts/experiments/nli_verifier/nli_verifier_v1/`, `nli_verifier_v2/` |
| 10 E3x | `configs/nli_verifier_exploration_v2.toml` | `artifacts/experiments/nli_verifier_exploration/e3x_v2/` (`selection.json`, `confirmation.*`, `final_test.json`) |
| 11 E5 | `configs/confidence_policies.toml` | `artifacts/experiments/confidence/confidence_v1/confidence_v1_report.json` |
| 12 E6 | `configs/fuzzy_matching.toml` | `artifacts/experiments/fuzzy/` |
| 13 verificador sobre `udv_v1` | `configs/udv_verifier.toml` | `artifacts/udv/udv_v1_verifier.jsonl`, `udv_v1_verifier_report.json` |

### UDV v2 (14 a 18)

```bash
# 14. cortes de udv_v2: cosseno de janela e verificador sobre premissa de UDV
uv run python -m experiments.udv.calibrate_v2 cosine
uv run python -m experiments.verifier.udv_verifier --config configs/udv_v2_calibration_verifier.toml translate
uv run python -m experiments.verifier.udv_verifier --config configs/udv_v2_calibration_verifier.toml score
uv run python -m experiments.verifier.udv_verifier --config configs/udv_v2_calibration_verifier.toml apply
uv run python -m experiments.udv.calibrate_v2 verifier

# 15. UDV v2
uv run bookworm build-udvs --config configs/udv_v2.toml --run-name udv_v2
uv run bookworm verify-udvs --config configs/udv_v2.toml --run-name udv_v2 --baseline artifacts/udv/udv_v1.jsonl

# 16. verificador sobre udv_v2
uv run python -m experiments.verifier.udv_verifier --config configs/udv_v2_verifier.toml translate
uv run python -m experiments.verifier.udv_verifier --config configs/udv_v2_verifier.toml score
uv run python -m experiments.verifier.udv_verifier --config configs/udv_v2_verifier.toml apply

# 17. análise de udv_v2
uv run python -m experiments.udv.v2_analysis analyze

# 18. confidence_v2
uv run python -m experiments.verifier.confidence_v2 translate   # depois smoke, laya-smoke, decide, score, laya-score, evaluate-ea
```

| etapa | config | saída versionada |
|---|---|---|
| 14 cortes | `configs/udv_v2.toml`, `configs/udv_v2_calibration_verifier.toml` | `artifacts/calibration/threshold_v2*`, `udv_verifier_threshold_v1.json` |
| 15 `udv_v2` | `configs/udv_v2.toml` | `artifacts/udv/udv_v2.jsonl`, `udv_v2_coverage.json`, `udv_v2_verify.json` |
| 16 verificador | `configs/udv_v2_verifier.toml` | `artifacts/udv/udv_v2_verifier.jsonl`, `udv_v2_verifier_report.json` |
| 17 análise | caminhos em `DEFAULT_PATHS` do script | `artifacts/udv/udv_v2_analysis.json`, `udv_v2_annotation_plan.json` |
| 18 confidence_v2 | `configs/confidence_v2.toml` | `artifacts/experiments/confidence_v2/` (`ea_report.json`, `smoke/`, `scores/*_report.json`) |

### Validação humana (19 e 20)

Nenhum script preenche julgamento humano. `experiments.validation.generate_sample` e
`experiments.udv.fuzzy_matching` gravam planilhas com os campos de julgamento vazios; o preenchimento é
manual, seguindo o [guia do anotador](validation/annotation_guide.md).

```bash
# 19. amostra cega sorteada de udv_v1 no teste; --dry-run grava fora do repositório
uv run python -m experiments.validation.generate_sample --run-name udv_v1 --final-test
#     planilha de reanotação, depois da primeira rodada completa (não feita)
uv run python -m experiments.validation.generate_sample --run-name udv_v1 --final-test --stage repeat
#     planilha suplementar de udv_v2 (26 linhas, só os itens moved)
uv run python -m experiments.udv.v2_analysis supplement-sheet --final-test

# 20. precisão final, com as duas planilhas julgadas
uv run python -m experiments.udv.v2_analysis score-annotation --final-test \
  --annotation artifacts/validation/human_validation_v1_udv_v1/annotation.csv \
  --supplement-dir artifacts/validation/human_validation_v1_udv_v2_supplement \
  --output artifacts/udv/udv_v2_precision_final.json
```

| etapa | config | saída versionada |
|---|---|---|
| 19 amostra | `configs/validation_sample.toml` | `artifacts/validation/human_validation_v1_udv_v1/`, `human_validation_v1_udv_v2_supplement/` |
| 20 precisão | `configs/validation_sample.toml` (`[udv_v2_supplement]`) | `artifacts/udv/udv_v2_precision_final.json` |

Os itens `same` e `superset` herdam o rótulo de `udv_v1` pela seção `[udv_v2_supplement]`. O campo
`existe_trecho_melhor` ficou vazio, e por isso `experiments.validation.precision_report` recusa a
planilha; os números finais vêm de `score-annotation`. Os arquivos `udv_v2_precision_interim_*.json` são
leituras intermediárias de planilhas incompletas, guardadas como histórico. Resultado:
[relatório, seções 7.2 e 7.4](report.md#7-validação-humana).

### Atores, perfis e simulação (21 a 23)

```bash
# 21. falas por ator
uv run python -m experiments.actors.measure_hearing_actors
uv run python -m experiments.actors.build_speeches
uv run python -m experiments.actors.filter_speeches --output artifacts/cache/hearing_actors/actors_multi_hearing_train.jsonl

# 22. perfis de ator e simulação com MLX (rodada udv_v1); smoke, profiles, evaluate e simulate rodam uma etapa
uv run --with "mlx-lm==0.31.3" python -m experiments.mlx.run all --model qwen38_27b --all-actors
#     sobre udv_v2 (não rodado nesta versão)
uv run --with "mlx-lm==0.31.3" python -m experiments.mlx.run all --model qwen38_27b --all-actors --settings configs/mlx_udv_v2.yaml

# 23. conferência de perfis contra UDVs (udv_v2; para udv_v1, configs/profile_validation_udv_v1.toml)
uv run bookworm validate-profiles --config configs/profile_validation.toml
```

| etapa | config | saída versionada |
|---|---|---|
| 21 falas por ator | `configs/hearing_actors.toml`, `configs/actor_profiles.toml` | `artifacts/hearing_actors/`, `artifacts/actor_profiles/train_speeches_stats.json` |
| 22 perfis e simulação | `configs/mlx.yaml`, `configs/mlx_udv_v2.yaml`, `configs/actor_profiles.toml`, `configs/actor_simulation.toml` | `artifacts/mlx_runs/qwen38_27b/` (rodada `udv_v1`) |
| 23 conferência | `configs/profile_validation.toml`, `configs/profile_validation_udv_v1.toml` | `artifacts/profile_validation/` |

O backend MLX está descrito em [`methodology/mlx_backend.md`](methodology/mlx_backend.md).

### Etapas depois da UDV e demo (24 e 25)

```bash
# 24. etapas depois da UDV sobre udv_v2, lado a lado com udv_v1
uv run python -m experiments.udv.v2_downstream

# 25. exportação da demo web (argumentos completos no README da demo)
uv run bookworm export-site --config configs/udv_v2.toml --run-name udv_v2 ...
```

| etapa | config | saída versionada |
|---|---|---|
| 24 downstream | caminhos no próprio script | `artifacts/udv/udv_v2_downstream_report.json` |
| 25 demo | `configs/udv_v2.toml`, `configs/hearing_actors.toml` | `artifacts/web/export_site_udv_v2.json` (resumo); os dados vão para `../bookworm/web/app/data/`, fora do Git |

Os argumentos da etapa 25 estão no [README da demo](../bookworm/web/README.md).

## Tempo de cada etapa

- As etapas 1 a 5 e 21 rodam em minutos em CPU ou MPS; a rodada `udv_v1` leva 23,5 s com o cache de
  embeddings pronto.
- As etapas com modelo levam horas: a calibração `threshold_v2` levou 990 s, a construção de `udv_v2`
  662 s, e cada avaliador do confidence_v2 entre 20 min e 4 h
  ([`methodology/confidence.md`](methodology/confidence.md#o-que-rodou-e-o-que-foi-descartado)); a
  rodada de perfis e simulação levou cerca de 13 h.

## Artefatos pesados

Tudo em `artifacts/` é versionado, com duas exceções:

- `artifacts/cache/` (ignorado): embeddings, traduções, respostas de modelo e escores por par. Cada
  comando refaz o que falta no cache e reutiliza o resto.
- As saídas por item das rodadas de experimento (escores, consultas, features, predições e pares, 369
  arquivos) saíram do Git. [`artifacts/MANIFEST_heavy.tsv`](../experiments/artifacts/MANIFEST_heavy.tsv)
  lista cada uma com caminho, tamanho, sha256 e o comando que a regenera. Os relatórios agregados que
  citam esses arquivos continuam versionados. Um pacote com esses arquivos será anexado a um GitHub
  Release do repositório (pendente); até lá, eles se restauram da tag `research-2026-09-28`.

Os comandos abaixo leem arquivos pesados de etapas anteriores e só rodam depois que eles existirem,
regenerados pelo comando da coluna `regenerate` do manifesto ou restaurados da tag:

| comando | lê |
|---|---|
| `experiments.verifier.nli_exploration` (`explore`, `confirm`, `final-test`) | `scores/<avaliador>_<split>.jsonl` de `nli_verifier_v1`, `nli_verifier_v2` e `final_test_e3x` |
| `experiments.verifier.udv_verifier apply` (as três configurações) | os escores de treino de `nli_verifier_v1` e `nli_verifier_v2` que o E3x leu, usados no reajuste do primário, e os de `artifacts/experiments/nli_verifier/<rodada>/scores/` gravados por `score` |
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

Os arquivos restaurados ficam ignorados pelo Git e conferem com o sha256 do manifesto. O manifesto
registra os caminhos e os comandos da versão de pesquisa (`challenge/...`, `python -m utils.<nome>`);
o [mapa de caminhos](path_map.md) dá o caminho e o módulo de cada um nesta versão.

## Proveniência e caminhos antigos

Os relatórios JSON gravam o sha256 do código (`code`), das entradas (`inputs`) e da configuração
(`config`) de cada rodada, e não a linha de comando. Os módulos desta versão receberam formatação e
anotações de tipo depois das rodadas, sem mudança de comportamento, e por isso os hashes gravados não
coincidem com os arquivos atuais. Os relatórios gravados antes desta versão registram os caminhos
`utils/<nome>.py`; os gravados a partir dela, o caminho a partir da raiz do repositório.

O código exato de uma rodada está no histórico da tag `research-2026-09-28`. Para achar o commit cujo
arquivo tem o hash gravado:

```bash
f=challenge/utils/retrieval_models.py; h=<sha256 gravado>
for c in $(git rev-list research-2026-09-28 -- "$f"); do
  [ "$(git show "${c}:${f}" | shasum -a 256 | cut -d' ' -f1)" = "$h" ] && echo "$c" && break
done
```

Os caminhos citados em artefatos, notebooks, ADRs e no manifesto que mudaram ou saíram desta versão
estão no [mapa de caminhos](path_map.md).
