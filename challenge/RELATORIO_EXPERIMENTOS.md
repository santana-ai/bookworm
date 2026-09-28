# Relatório de experimentos do bookworm sobre o PublicHearingBR

Este relatório reúne, num único lugar, todos os experimentos, testes, métricas e resultados produzidos
até 28/09/2026 no repositório. Ele foi escrito para quem vai redigir o artigo do desafio Ideias em Rede
e precisa de cada número com a sua origem. Cada número aparece com o arquivo de onde foi lido e, quando
útil, a chave JSON. A web demo e os mockups ficam fora do escopo.

Convenções usadas no texto:

- Números com ponto decimal, exatamente como estão nos artefatos JSON, para que possam ser buscados
  diretamente nos arquivos. Contagens inteiras também seguem os artefatos.
- Caminhos relativos a `challenge/`, salvo indicação. A primeira versão deste relatório leu dois ramos
  de trabalho (`bookworm_germano`, commit `fbf7a87`, com `udv_v2`, e `udv_germano`, commit `b593be4`, com
  `confidence_v2`); os dois foram unidos, e todos os artefatos citados estão nesta versão nos caminhos
  indicados. Os comandos de cada etapa, em ordem, estão em `README.md`.
- IC 95% é intervalo de confiança de 95%. "Holm" é o p-valor ajustado pelo método de Holm dentro da
  família declarada no próprio experimento.
- "Planejado e não rodado" significa que existe código ou declaração, mas nenhum artefato de resultado
  versionado.

## Sumário

- [Resumo executivo](#resumo-executivo)
- [1. Dataset e tarefa](#1-dataset-e-tarefa)
- [2. Splits](#2-splits)
- [3. UDV: versões v0, v1 e v2](#3-udv-versões-v0-v1-e-v2)
- [4. Trabalho 1: achar a evidência](#4-trabalho-1-achar-a-evidência)
- [5. Trabalho 2: dizer quanto confiar na evidência](#5-trabalho-2-dizer-quanto-confiar-na-evidência)
- [6. Atores e regras de ligação](#6-atores-e-regras-de-ligação)
- [7. Validação humana (parcial)](#7-validação-humana-parcial)
- [8. Glossário de métricas](#8-glossário-de-métricas)
- [9. Software e reprodutibilidade](#9-software-e-reprodutibilidade)
- [10. Limitações](#10-limitações)
- [11. Referências](#11-referências)
- [Apêndice A. Inconsistências entre artefatos](#apêndice-a-inconsistências-entre-artefatos)
- [Apêndice B. O que foi planejado e não rodado](#apêndice-b-o-que-foi-planejado-e-não-rodado)

## Resumo executivo

O problema: cada audiência do PublicHearingBR tem uma transcrição longa (média de 18.102 palavras) e uma
matéria curta (média de 627 palavras) que atribui opiniões a pessoas, mas nada no dataset diz em que ponto
da transcrição cada opinião foi dita. O projeto constrói, para cada uma das 2.203 opiniões publicadas, uma
Unidade Deliberativa Verificável (UDV): a opinião ligada a um trecho da fala da própria pessoa, com
posição exata, tipo de ligação e um sinal de confiança. O trabalho se divide em duas tarefas, avaliadas
separadamente:

1. **Trabalho 1, achar a evidência.** Dado uma opinião e a fala da pessoa, qual trecho a sustenta.
2. **Trabalho 2, dizer quanto confiar.** Dado o trecho escolhido, qual a chance de ele de fato sustentar a
   opinião.

Resultados principais, todos rastreados nas seções seguintes:

| achado | número | fonte |
|---|---|---|
| Opiniões com evidência localizada em `udv_v1` e `udv_v2` | 2.105 de 2.203 (90 de pessoas não resolvidas, 8 sem sentença candidata) | `artifacts/udv/udv_v2_analysis.json` |
| Citações literais localizadas (`quote_found`) | 277 | idem |
| Recuperação no benchmark NLI (validação, sentença, `serafim_335m`) | acc@1 0.8774, MRR 0.9295 | `artifacts/experiments/retrieval/retrieval_v1/retrieval_v1_report.json` |
| Nenhum dos 15 recuperadores alternativos supera o `serafim_335m` na sentença após Holm | E1, validação `nli` | idem |
| Janelas de 2 sentenças contra sentença, mesmo recuperador (`nli`, validação, `serafim_335m`) | lift +0.0143 [-0.0091; 0.0409], Holm 0.4104 | idem |
| Laya como reranqueador do top 20 (resultado negativo) | MRR 0.8690 contra 0.9295, Holm 0.0004 | `artifacts/experiments/retrieval/retrieval_v2/retrieval_v2_report.json` |
| Sinais derivados do cosseno não melhoram a confiança (E5) | nenhum IC de diferença de AURC exclui zero no `nli` | `artifacts/experiments/confidence/confidence_v1/confidence_v1_report.json` |
| Verificador primário (E3x), teste final, 559 opiniões | ROC AUC 0.9122 [0.8698; 0.9454], kappa 0.5924 | `artifacts/experiments/nli_verifier_exploration/e3x_v2/final_test.json` |
| Primário contra o cosseno Serafim na validação (confidence_v2 E-A) | ROC AUC 0.8758 contra 0.7302, Δ +0.1457 [0.0852; 0.2159], Holm 0.012 | `artifacts/experiments/confidence_v2/ea_report.json` |
| Validação humana parcial, `quote_found` (precisão estrita) | 17 de 18, 0.9444 [0.7424; 0.9901] | recálculo desta data, seção 7 |
| Validação humana parcial, `semantic_match_high` (precisão tolerante) | 25 de 32, 0.7812 [0.6125; 0.8898] | idem |

O que ainda não está resolvido: a validação humana cobre 65 de 127 linhas e nenhum critério declarado
pode ser decidido; o critério de `quote_found` já não pode mais ser atingido quando a planilha for
completada, porque exige zero erros em 35 e um erro já foi anotado (seção 7). O verificador foi ajustado
com premissas de quatro trechos e aplicado a UDVs com premissa de uma sentença ou janela, uma mudança de
domínio que não foi medida com rótulo humano.

## 1. Dataset e tarefa

### 1.1 O que é o PublicHearingBR

O PublicHearingBR (Fernandes et al., arXiv 2410.07495) reúne 206 audiências públicas da Câmara dos
Deputados. O repositório lê dois arquivos baixados do Hugging Face por `utils/download_dataset.py` e nunca
versionados (`dataset/`).

| arquivo | registros | chaves de topo | fonte |
|---|---|---|---|
| LDS (`PublicHearingBR_LDS.jsonl`) | 206 | `id`, `materia`, `metadados`, `transcricao` | `eda_v01.ipynb` |
| NLI | 206 | `id`, `metadados_extraidos` (`assunto`, `envolvidos`, `tl_dr`) | `eda_v01.ipynb` |

O sha256 do LDS usado em todas as rodadas é `c4e392ab95ce22f6228eace16f15e9efe1c846724846b873dec420aec3d872c0`
(`configs/retrieval_experiments.toml`, `[dataset].lds_sha256`, e demais configs).

Medidas do EDA (`eda_v01.ipynb`):

| medida | valor |
|---|---|
| palavras por transcrição, média | 18102.09 (mediana 16424.5, máximo 147728) |
| palavras por matéria, média | 627.72 |
| turnos por transcrição | média 50.5, mínimo 5, máximo 2604; nenhuma transcrição sem turnos |
| pessoas em `metadados.envolvidos` | 1065 (5.17 por audiência) |
| opiniões do LDS | 2203 (10.69 por audiência) |
| opiniões do NLI | 4238 |
| opiniões NLI marcadas como não inferíveis pelo especialista | 504 de 4238 (11.89%) |
| `assunto` idêntico entre LDS e NLI | 0 de 206 audiências |
| audiência 0: opiniões LDS contra NLI | 26 contra 18, 0 coincidências exatas; 64 de 72 trechos NLI achados na transcrição |

### 1.2 Três leituras parciais que o dataset contém

O documento de visão do projeto (`CONSTITUTION.md`, seção 3.6) distingue três verdades parciais, e todo
o relatório as mantém separadas:

- **Documental**: o que a transcrição sustenta. É o que a UDV tenta ancorar.
- **Editorial**: o que a matéria selecionou e redigiu. As 2.203 opiniões do LDS são essa leitura, com a
  redação da matéria.
- **Anotada sob recuperação**: o que o especialista validou a partir de apenas quatro trechos
  recuperados. É o rótulo do arquivo NLI (`label_inferable`), com semântica
  `retrieved_context_entailment` (`artifacts/benchmarks/nli_v1_report.json`, `label_semantics`). Ele diz
  se a opinião é inferível daqueles quatro trechos, não se é verdadeira no mundo nem se está na
  transcrição inteira.

Por causa dessa distinção, uma opinião ausente da matéria nunca é tratada como erro, e o rótulo NLI
nunca é chamado de verdade global.

### 1.3 Por que o arquivo NLI não resolve a ligação

As 4.238 opiniões NLI foram geradas por um LLM a partir da transcrição, no experimento de sumarização do
artigo do dataset, e não são as 2.203 do LDS (`UDV.md`, seção "O problema"). Cada opinião NLI vem com
`chunks_proximos` (quatro trechos) e `verificacao_alucinacao`, com a verificação manual e 12 juízes
automáticos: 3 prompts vezes 4 modelos (`gpt-4o-mini-2024-07-18`, `gpt-4o-2024-08-06`, `deepseek-chat`,
`sabia-3.1-2025-05-08`). A concordância desses juízes com o rótulo manual vai de 0.677 a 0.943
(`eda_v01.ipynb`). O arquivo NLI é usado aqui como banco de teste do Trabalho 2 (seção 5), não como fonte
de ligação.

### 1.4 A tarefa que o projeto define

A UDV liga cada opinião do LDS a um trecho da fala da pessoa, com offsets exatos, índice de turno, nível
(`tier`), tipo de suporte (`support_type`) e proveniência (`provenance`). O registro e as regras estão em
`UDV.md` e, para `udv_v2`, em `PIPELINE.md` e no ADR 0006
(`../bookworm/docs/adr/0006-udv-v2-windows-full-quotes-and-verifier.md`).

## 2. Splits

### 2.1 O problema que o split resolve

Opiniões da mesma audiência dividem falantes, tema e transcrição. Um split aleatório por opinião vazaria
informação entre treino e teste. Um split por audiência aleatório ainda deixaria audiências de datas
vizinhas, com os mesmos debates, dos dois lados. O split adotado, `temporal_v1`, corta por data de
publicação da matéria, com audiências inteiras em cada lado.

Comandos: `uv run python -m utils.build_splits` e `uv run python -m utils.verify_splits`. Semente 42
(`configs/splits.toml`). Fonte de todos os números desta seção:
`artifacts/splits/temporal_v1_report.json`.

### 2.2 Datas

| medida | valor |
|---|---|
| audiências com carimbo de publicação | 206 de 206 (13 com carimbo de atualização) |
| matérias que mencionam o dia do evento | 156, das quais 155 confirmadas pelo carimbo |
| defasagem entre evento e publicação | 0 dia em 142 menções, 1 dia em 15 |
| discordâncias de dia da semana | 2 (audiências 111 e 116); só a 111 não tem menção concordante |
| datas distintas | 128, de 2021-11-18 a 2024-05-09; no máximo 5 audiências por data |
| candidatos a ponto de corte | 65 |

### 2.3 Fronteiras e tamanhos

| split | período | audiências | fração | pessoas | UDVs |
|---|---|---|---|---|---|
| train | até 2023-11-13 | 144 | 0.699 | 745 | 1536 |
| validation | 2023-11-21 a 2023-12-20 (8 dias após o fim do train) | 32 | 0.1553 | 161 | 308 |
| test | a partir de 2024-03-05 (76 dias após a validação) | 30 | 0.1456 | 159 | 359 |

O campo `udvs_by_tier` desse relatório foi calculado sobre `udv_v0`.

### 2.4 Checagens de vazamento

| checagem | resultado |
|---|---|
| atores distintos | 879; 55 aparecem em mais de um split; 24 em train e test |
| quase duplicatas entre splits (TF-IDF, limiar 0.5) | 9888 pares comparados, 0 acima do limiar; máximo 0.459 (audiências 68 e 105) |

A presença de 24 atores em train e test é esperada (deputados recorrentes) e é tratada na avaliação de
perfis de ator, que só usa audiências de treino para construir o perfil (seção 6).

## 3. UDV: versões v0, v1 e v2

### 3.1 O que é uma UDV e por que cada campo existe

| campo | problema que resolve |
|---|---|
| `evidence` com offsets de caracteres e `speaker_turn` | permite a qualquer leitor abrir a transcrição no ponto exato |
| `tier` | declara como a ligação foi obtida: `quote_found` (citação literal), `semantic_match_high` e `semantic_match_weak` (similaridade acima ou abaixo de um corte), `no_evidence` (pessoa com turnos, mas sem sentença candidata), `person_not_resolved` (pessoa sem turno) |
| `support_type` | separa `direct_quote`, `semantic_with_short_quote` (a sentença escolhida pelo encoder contém um prefixo curto de citação) e `semantic_similarity` |
| `provenance` | `weak` para regra de texto, `model` para encoder, `null` sem evidência; nunca apresentada como anotação humana |

Fonte: `UDV.md`, seções "O registro" e "Construção".

### 3.2 Construção comum a v0 e v1

Etapas (`UDV.md`, "Construção"; código em `utils/udv_pipeline.py` e `utils/build_udvs.py`):

1. **Turnos.** `split_into_turns` acha os cabeçalhos de fala por expressão regular: 17.264 turnos nas 206
   audiências. O turno resolve a autoria: qualquer trecho dentro dele tem autor conhecido.
2. **Resolução da pessoa.** O nome da matéria casa com o do cabeçalho quando os conjuntos de palavras sem
   acento são iguais ou um contém o outro (ambos com duas palavras ou mais); nome de uma palavra só casa
   com candidato idêntico ou com o único falante que tenha aquela palavra. Resultado: 1.020 de 1.065
   participantes com ao menos um turno; as 45 pessoas restantes somam 90 opiniões.
3. **Sentenças candidatas.** Segmentação dentro de cada turno (ADR 0002), partes com 4 palavras ou mais:
   115.599 sentenças candidatas. A segmentação por turno corrigiu 350 sentenças de 244 pessoas que, na
   versão anterior, atravessavam dois turnos.
4. **Citação literal.** Trechos de 10 caracteres ou mais entre aspas duplas, tipográficas ou simples
   (aspas simples com fronteira de palavra, ADR 0003). Prefixos de 10, 6, 4 e 3 palavras são procurados
   nos turnos da pessoa sem distinção de maiúsculas. Prefixo de 6 palavras ou mais gera `quote_found`.
   O corte de 6 palavras veio de uma leitura exploratória sem protocolo de anotação (`UDV.md`); ele não foi
   validado por anotação humana antes da amostra cega da seção 7.
5. **Similaridade.** Sem citação confiável, opinião e sentenças são codificadas pelo Serafim 335m e vence a
   sentença de maior cosseno.
6. **Nível.** `semantic_match_high` se cosseno maior ou igual ao corte, senão `semantic_match_weak`.
7. **Offsets.** Todas as 2.105 evidências de `udv_v1` são localizadas; em 85 das 115.599 sentenças a busca
   devolve a primeira ocorrência de um texto repetido no turno
   (`artifacts/udv/turn_segmentation_summary.json`, `offset_search`).

Funil das citações nas 2.203 opiniões (`artifacts/benchmarks/masked_quotes_v1_report.json`,
`splits.all.opinions`): 2203 opiniões, 2113 de pessoas resolvidas, 960 com citação extraída, 549 com algum
prefixo achado, 277 com prefixo confiável de 6 palavras ou mais.

### 3.3 Calibração do corte de cosseno para sentenças (`threshold_v1`)

**Problema.** O corte separa `semantic_match_high` de `semantic_match_weak`. Os cortes do protótipo (0.25
com TF-IDF; 0.46, 0.44 e 0.47 com o encoder) foram calibrados nas 20 primeiras audiências, que incluem
audiências de validação e teste. Como a amostra de validação humana é sorteada no teste e depende dessa
fronteira, o corte precisava ser recalculado só no treino.

Comando: `uv run python -m utils.calibrate_threshold`. Saída: `artifacts/calibration/threshold_v1.json`.
Bootstrap de 1000 réplicas por audiência, semente 42. Regra primária declarada em 22/09/2026:
`masked_top1_youden`.

| regra | pares | corte | IC 95% | positivos abaixo | negativos no corte ou acima |
|---|---|---|---|---|---|
| `legacy_random` (adotada) | 183 consultas, 106 audiências; negativo = sentença de outra audiência do train | 0.4543 (arredondado 0.45) | 0.4366 a 0.4687 | 1 | 0 |
| `hard_negative` | mesmos positivos; negativo = outra sentença da mesma pessoa | 0.5019 | 0.5004 a 0.5416 | 7 | 21 |
| `masked_hard_negative` | 142 consultas, 87 audiências | 0.3798 | 0.3623 a 0.4075 | 41 | 38 |
| `masked_top1_youden` (primária, degenerada) | 142 consultas | 0.3928 | 0.3928 a 0.7725 | n/a | n/a |

A regra primária mostrou-se degenerada: só 12 de 142 consultas de treino têm top-1 correto e 130 têm
top-1 errado; acc@1 0.0845 contra 0.0147 ao acaso; o melhor índice J de Youden é 0, e o corte aceita todas
as consultas (`rules.masked_top1_youden.optimum.youden_j`). A mediana do escore top-1 é 0.5543 nos corretos
e 0.6472 nos errados (`rules.masked_top1_youden.top1_score_correct.median` e
`top1_score_incorrect.median`): o encoder dá escore maior aos erros. Sem mascarar a citação, a acc@1 nas
mesmas consultas é 0.5634.

A regra `legacy_random` usa o ponto médio entre o quartil 25 dos positivos e o quartil 75 dos negativos. O
corte adotado é 0.45 (antes 0.47). A troca move 9 registros de `semantic_match_weak` para
`semantic_match_high` e não muda nenhuma evidência (`UDV.md`, "Calibração").

### 3.4 Versões v0, v1_pre e v1

Fonte: `UDV.md`, "Versões", e `artifacts/udv/udv_v0_coverage.json`, `udv_v1_pre_coverage.json`,
`udv_v1_coverage.json`.

| rodada | data | mudança | `quote_found` | `semantic_match_high` | `semantic_match_weak` | `no_evidence` | `person_not_resolved` |
|---|---|---|---|---|---|---|---|
| `udv_v0` | 21/09/2026 | citação com prefixo de 6+ palavras, corte 0.47 | 260 | 1793 | 52 | 8 | 90 |
| `udv_v1_pre` | 23/09/2026 | sentenças por turno e escolha de ocorrência (ADR 0002), aspas simples (ADR 0003) | 277 | 1776 | 52 | 8 | 90 |
| `udv_v1` | 23/09/2026 | corte 0.45 calibrado só no train | 277 | 1785 | 43 | 8 | 90 |

Em `udv_v0`, os tipos de suporte são 260 `direct_quote`, 100 `semantic_with_short_quote` e 1745
`semantic_similarity`, e 2104 de 2105 evidências têm offset (`udv_v0_coverage.json`). De `udv_v0` para
`udv_v1_pre`, 17 opiniões passam de `semantic_match_high` para `quote_found` (14 pelas aspas simples, 3 por
prefixo mais longo em outra citação) e 42 registros mudam a evidência (`UDV.md`). O tempo de construção de
`udv_v1` foi 23.5 s (`udv_v1_coverage.json`).

### 3.5 udv_v2: janelas, citação inteira e verificador

**Problema.** Em `udv_v1` a evidência é uma única sentença. Duas consequências foram medidas: a citação
confiável é cortada no fim da sentença que contém o prefixo, mesmo quando a citação continua; e a camada
do verificador (seção 5.7) reprovou 72% das citações literais com premissa de uma sentença. `udv_v2`
alonga a evidência e recalibra os cortes. Decisões no ADR 0006; configuração em `configs/udv_v2.toml`.

Regras de `udv_v2` (`PIPELINE.md`, "Pipeline recomendado"):

1. Resolução de pessoa igual a `udv_v1`.
2. Citação: mesmo casamento de prefixo; a evidência se estende até o fim da citação, achado por um sufixo
   de 6, 4 ou 3 palavras a no máximo 2 vezes o tamanho da citação; sem sufixo, cobre tantas partes de
   sentença quantas a citação tem. Nunca sai do turno.
3. Busca semântica sobre unidades `window2` (duas sentenças consecutivas do mesmo turno), com o
   `serafim_335m`.
4. Nível: `semantic_match_high` se o cosseno da janela for pelo menos 0.50.
5. Confiança: probabilidade do verificador primário do E3x sobre o texto da evidência, com duas decisões
   gravadas em arquivo lateral (corte do benchmark 0.7478 e corte de premissa UDV 0.2429). A decisão não
   muda o nível.

Por que `window2`: no E2 (seção 4.4), a janela de duas sentenças não perde para a sentença em lift no
`nli` e ganha no benchmark de citações mascaradas, enquanto o turno inteiro perde em lift e entrega
milhares de caracteres.

#### Calibração do corte de janela (`threshold_v2`)

Fonte: `artifacts/calibration/threshold_v2.json`. Regra adotada `legacy_random`, declarada no ADR 0006 em
28/09/2026. Só as 144 audiências de treino; a checagem de vazamento passou. Tempo 990.2 s.

| regra | corte | IC 95% | positivos abaixo | negativos no corte ou acima |
|---|---|---|---|---|
| `legacy_random` (adotada) | 0.5015 (arredondado 0.50) | 0.4858 a 0.5129 | 0 | 0 |
| `hard_negative` | 0.572 | 0.5685 a 0.607 | 5 | 15 |
| `masked_hard_negative` | 0.4432 | 0.4101 a 0.4671 | n/a | n/a |
| `masked_top1_youden` | 0.6878 | 0.5278 a 0.802 | n/a | n/a |

Com janelas, `masked_top1_youden` deixa de ser degenerada (J 0.193), mas continua fraca: 28 de 142
corretos, acc@1 0.1972 contra 0.0303 ao acaso; sem máscara, acc@1 0.7254.

#### Corte do verificador para premissa de UDV (`udv_verifier_threshold_v1`)

Fonte: `artifacts/calibration/udv_verifier_threshold_v1.json` e
`artifacts/calibration/threshold_v2_verifier_scores_report.json`.

**Problema.** O corte 0.7478 do verificador foi escolhido com premissas de quatro trechos. Com premissa
de uma janela, a distribuição da probabilidade muda. O corte de premissa UDV é calibrado nos mesmos 183
pares de treino, com a mesma regra `legacy_random`.

| medida | valor |
|---|---|
| corte `legacy_random` | 0.2429, IC 95% 0.221 a 0.2782; 14 de 183 positivos abaixo, 2 negativos no corte ou acima |
| variante `hard_negative` | 0.2722, IC 95% 0.2485 a 0.319 |
| no corte do benchmark (0.7478) | 86 de 183 positivos passam (0.4699); 0 negativos `legacy_random` e 4 `hard_negative` passam |
| pares pontuados | 549; `refit_check` verdadeiro |
| Spearman entre verificador e cosseno | 0.9324 em todos os pares semânticos; 0.5039 só nos positivos |
| vazamento | 133 audiências, checagem passou |

O bootstrap desse corte mantém os pares fixos e reamostra audiências; os negativos não são ressorteados.

#### Resultado de udv_v2

Fontes: `artifacts/udv/udv_v2_analysis.json`, `artifacts/udv/udv_v2_coverage.json`,
`artifacts/udv/udv_v2_verify.json`, `artifacts/udv/udv_v2_verifier_report.json`.

| nível | `udv_v1` | `udv_v2` |
|---|---|---|
| `quote_found` | 277 | 277 |
| `semantic_match_high` | 1785 | 1744 |
| `semantic_match_weak` | 43 | 84 |
| `no_evidence` | 8 | 8 |
| `person_not_resolved` | 90 | 90 |

| split (`udv_v2`) | `quote_found` | `semantic_match_high` | `semantic_match_weak` | `no_evidence` | `person_not_resolved` |
|---|---|---|---|---|---|
| train | 183 | 1217 | 54 | 2 | 80 |
| validation | 53 | 233 | 19 | 0 | 3 |
| test | 41 | 294 | 11 | 6 | 7 |

| diferença de `udv_v1` para `udv_v2` | valor |
|---|---|
| registros idênticos | 234 |
| registros alterados | 1969 com evidência diferente em texto, `start_char`, `end_char` ou `speaker_turn` (`udv_v2_analysis.json`, `diff.evidence_changed`); 1970 com qualquer mudança, incluindo a de só nível (`diff.records_changed_any`) e, por outro critério, em `udv_v2_verify.json` (`baseline_diff.evidence_changed`, que compara o objeto `evidence` inteiro, `score` incluído); ver Apêndice A |
| citações estendidas | 152 |
| janelas que contêm a sentença de `udv_v1` | 1343 |
| janelas em outro ponto | 474 |
| só mudança de nível | 1 |
| movimentos de nível | 43 de high para weak, 2 de weak para high |
| palavras da evidência de citação, mediana | 27 em `udv_v1`, 45 em `udv_v2` |
| citações com as palavras finais presentes na evidência | 82 em `udv_v1`, 172 em `udv_v2` |
| tipos de suporte em `udv_v2` | 277 `direct_quote`, 149 `semantic_with_short_quote`, 1679 `semantic_similarity` |
| `udv_v2_verify.json`, problemas | lista vazia |
| tempo de construção | 662.3 s |

Distribuição da decisão do verificador em `udv_v2` (`udv_v2_verifier_report.json`,
`supported_at_train_threshold` e `supported_at_udv_threshold`):

| nível | n | passa em 0.7478 | passa em 0.2429 |
|---|---|---|---|
| `quote_found` | 277 | 133 (0.4801) | 256 (0.9242) |
| `semantic_match_high` | 1744 | 691 (0.3962) | 1476 (0.8463) |
| `semantic_match_weak` | 84 | 2 (0.0238) | 18 (0.2143) |
| todas | 2105 | 826 (0.3924) | 1750 (0.8314) |

Com a evidência mais longa, a fração de citações literais que passam no corte do benchmark sobe de 0.282
(`udv_v1`, seção 5.7) para 0.4801. Concordância entre o nível por cosseno e a decisão do verificador
(`udv_v2_analysis.json`): kappa 0.0535 no corte 0.7478 (tabela 691/1053/2/82) e 0.2616 no corte 0.2429
(1476/268/18/66). Spearman entre a probabilidade do verificador e o cosseno: 0.5839 em
`semantic_match_high` e 0.6229 nos dois níveis semânticos (`cosine_relation.spearman`). Em 136 UDVs a
evidência não mudou entre as versões.

Observação técnica: `evidence_score_check` do mesmo relatório compara o cosseno `sentence_max` do
verificador com `evidence.score` gravado pela UDV; só 27 de 1828 ficam dentro de 1e-4, com diferença
máxima 0.2375. Os dois não medem a mesma coisa em `udv_v2` (o escore da UDV é o cosseno da janela inteira;
o do verificador é o máximo por sentença), então a diferença é esperada, mas não foi documentada em
nenhum outro arquivo.

## 4. Trabalho 1: achar a evidência

O Trabalho 1 pergunta: dada a opinião e as unidades de fala da pessoa, qual representação coloca uma
passagem que sustenta a opinião em primeiro lugar. Ele é avaliado sem rótulo humano, com dois benchmarks
construídos a partir do próprio dataset.

### 4.1 B1: citações mascaradas

**Problema.** Uma opinião com citação literal confiável tem alvo conhecido: a sentença da citação. Se a
citação for apagada da opinião, o que sobra obriga o recuperador a achar a sentença pelo sentido, que é o
caso difícil do Trabalho 1.

Comando: `uv run python -m utils.build_quote_benchmark`. Fonte: `artifacts/benchmarks/masked_quotes_v1_report.json`.

| medida | valor |
|---|---|
| opiniões com prefixo confiável | 277 |
| mantidas após mascarar | 210 (67 descartadas por `masked_too_short`) |
| train | 183 consideradas, 142 mantidas |
| validation | 53 consideradas, 40 mantidas |
| test | 41 consideradas, 28 mantidas |
| acc@1 ao acaso | train 0.0147, validation 0.0181, test 0.0167 |
| candidatos por consulta, train | média 128.62 |

Com 40 consultas na validação, os intervalos de B1 são largos e nenhum resultado de B1 decide sozinho.

### 4.2 B2: benchmark NLI

**Problema.** B1 cobre só opiniões com citação. O arquivo NLI traz, para cada opinião gerada, os quatro
trechos que um recuperador achou e o julgamento do especialista. Uma unidade é relevante quando se
sobrepõe a um desses trechos numa opinião julgada inferível.

Comando: `uv run python -m utils.build_nli_benchmark`. Fonte: `artifacts/benchmarks/nli_v1_report.json`.

| split | opiniões | não inferíveis |
|---|---|---|
| train | 2981 | 339 |
| validation | 698 | 76 |
| test | 559 | 89 |
| todos | 4238 | 504 (0.1189) |

Trechos localizados na transcrição: 11920 de 11920 no train; 2228 de 2236 no test. A contagem de rótulos
confere com o artigo do dataset (3734 inferíveis, 504 não inferíveis). Juízes LLM do dataset, train e
validação juntos: melhor kappa `prompt_1_gpt-4o-mini` 0.6733; acurácia de "sempre inferível" 0.8872.

Nos benchmarks de recuperação, só entram opiniões inferíveis, de pessoa resolvida, com trecho localizado
e com ao menos uma sentença que se sobreponha a ele (`configs/retrieval_experiments.toml`,
`[benchmarks.nli].filters`): 2602 consultas no train e 620 na validação. A relevância aqui herda a
semântica do rótulo: o especialista julgou a opinião inferível dos quatro trechos juntos, não que cada
sentença sobreposta a sustente.

### 4.3 E1: comparação de recuperadores (`retrieval_v1`)

**Problema.** Escolher o recuperador da UDV. A comparação é feita dentro de cada tipo de unidade, contra
o recuperador de referência `serafim_335m`, porque unidades maiores acertam mais ao acaso.

Configuração: `configs/retrieval_experiments.toml`. Fonte: `artifacts/experiments/retrieval/retrieval_v1/retrieval_v1_report.json`
(`summaries` e `comparisons.<bench>.<split>.retrievers_within_unit`). Criado em 2026-09-24T12:26:02Z.
Splits lidos: train e validation; o teste não foi lido (`final_test` falso).

Recuperadores:

| nome | tipo | modelo ou parâmetros |
|---|---|---|
| `tfidf_word_speaker`, `tfidf_word_hearing` | TF-IDF de palavras, ajustado na fala da pessoa ou da audiência | unigramas, minúsculas, sem acento |
| `tfidf_char_speaker`, `tfidf_char_hearing` | TF-IDF de n-gramas de caracteres | `char_wb`, 3 a 5 |
| `bm25_speaker`, `bm25_hearing` | BM25 | k1 1.5, b 0.75, epsilon 0.25 |
| `serafim_335m` (referência) | denso | `PORTULAN/serafim-335m-portuguese-pt-sentence-encoder`, 128 tokens |
| `serafim_335m_ir` | denso | variante IR do Serafim 335m |
| `serafim_900m` | denso | Serafim 900m (base em português europeu, segundo o cartão) |
| `e5_large` | denso | `intfloat/multilingual-e5-large`, prefixos `query: ` e `passage: `, 512 tokens |
| `bge_m3` | denso | `BAAI/bge-m3`, só vetor denso, 8192 tokens |
| `mpnet` | denso | `paraphrase-multilingual-mpnet-base-v2` |
| `minilm` | denso | `paraphrase-multilingual-MiniLM-L12-v2` |
| `hybrid_rrf` | fusão por posto recíproco | `bm25_speaker` com `serafim_335m`, k 60, declarado antes de qualquer resultado |
| `rerank_bge` | reranqueador do top 20 do `serafim_335m` | `BAAI/bge-reranker-v2-m3`, 512 tokens, logit bruto |
| `rerank_mmarco` | reranqueador do top 20 | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |

Avaliação: acc@1, acc@3, acc@5 e MRR com IC por bootstrap de 2000 réplicas de audiências; McNemar exato
na acc@1 e teste de sinal por permutação de audiências (10000 amostras) no posto recíproco; Holm numa
família por benchmark, split, unidade e teste; semente 42 (`[evaluation]`). Empates ordenados por índice
de candidato; `rank_optimistic` e `rank_pessimistic` limitam o efeito de empates.

Uma emenda de 23/09/2026, registrada antes de qualquer resumo com recuperador denso, separou as perguntas
E1 (recuperadores) e E2 (unidades), porque a versão anterior misturava as duas e favorecia unidades
grandes (`multiple_comparisons_amendment`).

#### Benchmark `nli`, validação, unidade sentença (620 consultas, 32 audiências; acc@1 ao acaso 0.3044)

| recuperador | acc@1 [IC 95%] | acc@3 | MRR [IC 95%] | acc@1 vs referência, McNemar p (Holm) | MRR vs referência, p (Holm) |
|---|---|---|---|---|---|
| `serafim_335m` (referência) | 0.8774 [0.8396; 0.9095] | 0.9823 | 0.9295 [0.9052; 0.9496] | n/a | n/a |
| `serafim_335m_ir` | 0.8984 [0.8669; 0.9284] | 0.9935 | 0.9435 [0.9255; 0.9601] | 0.1597 (1) | 0.139 (1) |
| `bge_m3` | 0.8839 [0.8567; 0.9102] | 0.9855 | 0.9337 | 0.7163 (1) | 0.556 (1) |
| `serafim_900m` | 0.8823 | 0.9871 | 0.9335 | 0.8151 (1) | 0.6705 (1) |
| `rerank_bge` | 0.8790 | 0.9839 | 0.9316 | 1 (1) | 0.7858 (1) |
| `rerank_mmarco` | 0.8774 | 0.9806 | 0.9300 | 1 (1) | 0.9483 (1) |
| `hybrid_rrf` | 0.8742 | 0.9839 | 0.9294 | 0.8899 (1) | 0.9875 (1) |
| `mpnet` | 0.8710 | 0.9806 | 0.9269 | 0.7465 (1) | 0.7553 (1) |
| `tfidf_char_speaker` | 0.8613 | 0.9661 | 0.9184 | 0.3821 (1) | 0.1517 (1) |
| `minilm` | 0.8581 | 0.9694 | 0.9169 | 0.2461 (1) | 0.2216 (1) |
| `tfidf_char_hearing` | 0.8565 | 0.9661 | 0.9154 | 0.2132 (1) | 0.1113 (1) |
| `bm25_speaker` | 0.8484 | 0.9629 | 0.9078 | 0.08863 (0.9749) | 0.0271 (0.2981) |
| `tfidf_word_speaker` | 0.8419 | 0.9677 | 0.9045 | 0.04281 (0.5137) | 0.015 (0.18) |
| `bm25_hearing` | 0.8371 | 0.9694 | 0.9020 | 0.0124 (0.1624) | 0.003 (0.042) |
| `tfidf_word_hearing` | 0.8339 | 0.9661 | 0.9025 | 0.0116 (0.1624) | 0.007499 (0.09749) |
| `e5_large` | 0.7452 [0.6947; 0.7901] | 0.9339 | 0.8435 | 5.423e-12 (8.135e-11) | 9.999e-05 (0.0015) |

Leitura: depois de Holm, nenhum recuperador supera o `serafim_335m` na sentença. Dois são piores:
`e5_large` (acc@1 -0.1323 [-0.1716; -0.089]) e, só no MRR, `bm25_hearing` (-0.0275 [-0.0436; -0.0111],
Holm 0.042). O `serafim_335m_ir` tem o maior ponto (acc@1 0.8984), sem significância (Holm 1). O
`serafim_335m` continuou como encoder da UDV.

#### Benchmark `masked_quotes`, validação, unidade sentença (40 consultas, 24 audiências; acaso 0.0181)

| recuperador | acc@1 | acc@3 | acc@5 | MRR [IC 95%] | MRR vs referência, p (Holm) |
|---|---|---|---|---|---|
| `serafim_335m` (referência) | 0.1000 | 0.3250 | 0.4250 | 0.2612 [0.1781; 0.3456] | n/a |
| `serafim_900m` | 0.1250 | 0.3000 | 0.4000 | 0.2604 | 0.9775 (1) |
| `minilm` | 0.1250 | 0.2500 | 0.4250 | 0.2470 | 0.7216 (1) |
| `mpnet` | 0.1000 | 0.2750 | 0.4000 | 0.2450 | 0.5342 (1) |
| `bge_m3` | 0.1250 | 0.2750 | 0.3000 | 0.2430 | 0.6264 (1) |
| `serafim_335m_ir` | 0.1000 | 0.2750 | 0.3750 | 0.2423 | 0.6696 (1) |
| `rerank_bge` | 0.0750 | 0.3000 | 0.4250 | 0.2293 | 0.4244 (1) |
| `rerank_mmarco` | 0.1000 | 0.2750 | 0.3500 | 0.2240 | 0.5078 (1) |
| `tfidf_char_speaker` | 0.1000 | 0.2750 | 0.3500 | 0.2232 | 0.4976 (1) |
| `tfidf_char_hearing` | 0.0750 | 0.2750 | 0.3750 | 0.2127 | 0.3907 (1) |
| `e5_large` | 0.1000 | 0.2250 | 0.3250 | 0.2067 | 0.2411 (1) |
| `hybrid_rrf` | 0.0750 | 0.2250 | 0.3750 | 0.2051 | 0.1909 (1) |
| `bm25_hearing` | 0.0750 | 0.1500 | 0.2500 | 0.1688 | 0.1409 (1) |
| `tfidf_word_hearing` | 0.0500 | 0.1500 | 0.2750 | 0.1509 | 0.05539 (0.7755) |
| `bm25_speaker` | 0.0500 | 0.1250 | 0.1750 | 0.1446 | 0.06839 (0.8891) |
| `tfidf_word_speaker` | 0.0250 | 0.1000 | 0.2250 | 0.1296 | 0.0311 (0.4665) |

Nenhuma comparação de acc@1 em B1 validação tem Holm abaixo de 1. Com a citação mascarada, a melhor acc@1
é 0.125 em 40 consultas: achar a sentença exata só pelo resto da opinião é difícil para todos os
recuperadores testados.

#### Treino (descritivo)

No `nli` train (2602 consultas, 144 audiências, acaso 0.2778), a acc@1 na sentença vai de 0.7064
(`e5_large`) a 0.8782 (`serafim_335m_ir`), com `serafim_335m` em 0.8640. No `masked_quotes` train (142
consultas), vai de 0.0493 (`e5_large`) a 0.1268 (`serafim_900m`), com `serafim_335m` em 0.0845. Tabelas
completas em `summaries.<bench>.train.sentence`.

### 4.4 E2: comparação de unidades (`retrieval_v1`)

**Problema.** Uma unidade maior (janela, turno) contém a passagem relevante com mais frequência, mas
entrega mais texto ao leitor e acerta mais ao acaso. A métrica testada é o lift: acerto no top-1 menos a
chance de acerto ao acaso (`n_relevant / n_units`), pareado por consulta, contra a unidade sentença do
mesmo recuperador. acc@1 e MRR brutos são mostrados, mas não testados entre unidades
(`definitions.units_within_retriever`).

Unidades: `sentence`; `window2` e `window3` (N sentenças consecutivas do mesmo turno, passo 1); `turn` (o
turno inteiro).

Fonte: `comparisons.<bench>.validation.units_within_retriever`.

#### `nli`, validação, `serafim_335m`

| unidade | acc@1 | acaso | lift | Δ lift vs sentença [IC 95%] | p (Holm) | caracteres do top-1 |
|---|---|---|---|---|---|---|
| `sentence` | 0.8774 | 0.3044 | 0.5730 | n/a | n/a | 173.4 |
| `window2` | 0.9306 | 0.3433 | 0.5873 | +0.0143 [-0.0091; 0.0409] | 0.2653 (0.4104) | 297.1 |
| `window3` | 0.9355 | 0.3813 | 0.5541 | -0.0189 [-0.0459; 0.0093] | 0.2052 (0.4104) | 433.0 |
| `turn` | 0.7790 | 0.6398 | 0.1392 | -0.4338 [-0.4936; -0.3717] | 9.999e-05 (0.0003) | 4413.4 |

#### `masked_quotes`, validação, `serafim_335m`

| unidade | acc@1 | acaso | lift | Δ lift vs sentença [IC 95%] | p (Holm) | caracteres do top-1 |
|---|---|---|---|---|---|---|
| `sentence` | 0.1000 | 0.0181 | 0.0819 | n/a | n/a | 157.7 |
| `window2` | 0.3000 | 0.0388 | 0.2612 | +0.1793 [0.0652; 0.333] | 0.007899 (0.0165) | 298.7 |
| `window3` | 0.4000 | 0.0641 | 0.3359 | +0.254 [0.1013; 0.4008] | 0.005499 (0.0165) | 435.4 |
| `turn` | 0.4750 | 0.4058 | 0.0692 | -0.0127 [-0.1665; 0.1182] | 0.8672 (0.8672) | 3744.8 |

Leitura: no `nli`, janelas de 2 ou 3 sentenças empatam com a sentença em lift e o turno perde em todos os
recuperadores (Holm até 0.0003). No `masked_quotes`, `window2` e `window3` ganham da sentença no
`serafim_335m` (Holm 0.0165). O turno tem acc@1 bruta alta nos recuperadores lexicais (por exemplo
`tfidf_char_hearing` 0.9774 no `nli`), mas o acaso nessa unidade é 0.6398 e o top-1 tem cerca de 6700
caracteres. A janela de 2 sentenças foi adotada em `udv_v2` como compromisso entre acerto e tamanho do
trecho entregue (ADR 0006).

### 4.5 retrieval_v2: Laya como reranqueador (resultado negativo)

**Problema.** No E3 (seção 5.2), o Laya mostrou bom sinal para julgar se uma opinião é inferível. A
pergunta declarada em 27/09/2026, antes de qualquer escore do Laya em recuperação: reordenar o top 20 do
`serafim_335m` com o Laya coloca mais vezes uma sentença que sustenta a opinião em primeiro lugar?

Declaração: `configs/retrieval_experiments.toml`, `[declarations.retrieval_v2]`. Primário
`rerank_laya_p4` (checkpoint `laya_multi_pt`, pergunta `p4_supports`, escolhida por ter a maior ROC AUC
entre as perguntas em português no E3 v2, 0.793); secundário `rerank_laya_p3` (`p3_inferable`). Sem
tradução, CPU float32, top_k 20. Regra de decisão: adotar só se o MRR na validação `nli` for maior que o do
`serafim_335m` com Holm abaixo de 0.05. A decisão de rodar em CPU e manter top_k 20 foi tomada só pela
vazão (cerca de 46 pares por segundo num smoke de 100 pares), antes de qualquer métrica.

Fonte: `artifacts/experiments/retrieval/retrieval_v2/retrieval_v2_report.json` (criado 2026-09-27T02:48:35Z).
A família de Holm tem 4 recuperadores contra o `serafim_335m`.

| benchmark, validação, sentença | recuperador | acc@1 | MRR [IC 95%] | Δ MRR vs `serafim_335m` [IC 95%] | p (Holm) |
|---|---|---|---|---|---|
| `nli` (620) | `serafim_335m` | 0.8774 | 0.9295 [0.9052; 0.9496] | n/a | n/a |
| `nli` | `rerank_bge` | 0.8790 | 0.9316 | +0.0021 [-0.0117; 0.0194] | 0.7858 (1) |
| `nli` | `rerank_mmarco` | 0.8774 | 0.9300 | +0.0005 [-0.0145; 0.0167] | 0.9483 (1) |
| `nli` | `rerank_laya_p4` | 0.7790 | 0.8690 [0.8405; 0.8922] | -0.0605 [-0.0863; -0.0346] | 9.999e-05 (0.0004) |
| `nli` | `rerank_laya_p3` | 0.7694 | 0.8645 [0.8351; 0.8898] | -0.065 [-0.0905; -0.0384] | 0.0002 (0.0005999) |
| `masked_quotes` (40) | `serafim_335m` | 0.1000 | 0.2612 | n/a | n/a |
| `masked_quotes` | `rerank_laya_p4` | 0.0750 | 0.2160 | -0.0453 [-0.1437; 0.038] | 0.3351 (1) |
| `masked_quotes` | `rerank_laya_p3` | 0.0500 | 0.1999 | -0.0613 [-0.158; 0.029] | 0.2116 (0.8463) |

Na acc@1 do `nli` validação, `rerank_laya_p4` tem 98 consultas em que só o `serafim_335m` acerta contra 37
em que só o Laya acerta (McNemar p 1.502e-07, Holm 4.506e-07). O mesmo sinal aparece no train (MRR 0.8641
contra 0.9204, Holm 0.0004). Contra `rerank_bge` e `rerank_mmarco` como referência (relatórios
`retrieval_v2_report_vs_rerank_bge_sentence.json` e `retrieval_v2_report_vs_rerank_mmarco_sentence.json`,
descritivos e fora da decisão), o Laya também perde no `nli` com Holm 0.0004 a 0.0008.

**Decisão.** Resultado negativo pela regra declarada: o Laya não entra na recuperação da UDV. Ele julga bem
se a opinião é inferível de um conjunto de trechos (seção 5), mas reordena pior que o encoder qual
sentença é a relevante.

## 5. Trabalho 2: dizer quanto confiar na evidência

O Trabalho 2 pergunta: dado o trecho, qual a chance de ele sustentar a opinião. O rótulo disponível em
escala é o do especialista no benchmark NLI (`label_inferable`, sob recuperação). Todo resultado desta
seção herda essa semântica e a mudança de domínio: no benchmark a premissa são quatro trechos de cerca de
630 caracteres cada; na UDV a premissa é uma sentença ou janela.

### 5.1 E3 v1: NLI multilíngue contra o cosseno

**Problema.** Saber se um modelo de inferência textual ordena melhor as opiniões inferíveis do que o
cosseno do encoder que escolheu o trecho.

Declaração do primário em 22/09/2026, antes de qualquer escore: NLI multilíngue de 3 classes por trecho,
máximo da probabilidade de entailment entre os trechos (`configs/nli_verifier.toml`,
`primary_declaration`). Bootstrap de 1000 réplicas por audiência, semente 42. Fonte:
`artifacts/experiments/nli_verifier/nli_verifier_v1/evaluation_report.json`.

| sistema (validação, 698 opiniões) | ROC AUC [IC 95%] | kappa no corte max_f1 |
|---|---|---|
| `cosine_serafim.max` | 0.7302 [0.66; 0.7979] | 0.2181 |
| `cosine_serafim.sentence_max` | 0.7728 | 0.2785 |
| `xnli_mdeberta.max.entailment` (primário) | 0.7698 [0.6962; 0.8325] | 0.236 (corte 0.9068) |
| `xnli_mdeberta.concatenated.entailment` | 0.7285 | n/a |
| `xnli_mdeberta.min.not_contradiction` | 0.4601 | n/a |
| `assin2_mdeberta.max` | 0.6943 (corte 0.0, degenerado) | n/a |
| `assin2_mdeberta.concatenated` | 0.5085 | n/a |

Referência de juiz: `prompt_1_gpt-4o-mini`, escolhido no train (kappa 0.6692), tem kappa 0.6909 na
validação. "Sempre inferível" acerta 0.8911 na validação. Δ ROC AUC do primário contra o cosseno: +0.0396
[-0.0404; 0.1263], intervalo que inclui zero.

**Decisão.** O NLI sozinho não separa do cosseno; nenhum dos dois chega perto do kappa de um juiz LLM.

### 5.2 E3 v2: Laya, bateria de perguntas e tradução

**Problema.** Testar modelos de decisão que respondem perguntas sobre um par premissa e hipótese (Laya) e o
efeito de ler em inglês, com tradução automática.

Bateria (`configs/nli_verifier.toml`, `[decision_battery]`): perguntas p1 a p8 (p1 NLI, p2 NLI com ordem
das opções invertida, p3 inferível, p4 sustenta, p5 posição, p6 posição invertida, p7 cobertura, p8
similaridade). Painel = média de: média(p1, p2), p3, p4, média(p5, p6) e p7. Checkpoints:
`laya_multi_pt` (pasta `multilingual` do repositório, sobre mmBERT-base, lendo português), `laya_multi_en`
(o mesmo lendo inglês) e `laya_en_en` (checkpoint raiz lendo inglês); sufixo `_m2m100` quando a tradução
é do M2M100 em vez do NLLB. `laya==0.3.11`, `max_len` 1024, cabeça 256.

Fonte: `artifacts/experiments/nli_verifier/nli_verifier_v2/evaluation_report.json`. 156 sistemas
avaliados. O Jev (`jev_en`, `jev_pt`) consta em `scorers_missing`: foi declarado e nunca rodou.

| sistema (validação) | ROC AUC [IC 95%] | kappa |
|---|---|---|
| `laya_multi_pt.max.panel` (primário declarado) | 0.7814 [0.7148; 0.8376] | 0.275 |
| `laya_en_en.max.panel` | 0.8059 | n/a |
| `laya_en_en_m2m100.max.panel` | 0.802 | n/a |
| `laya_multi_en_m2m100.max.panel` | 0.7704 | n/a |
| `laya_multi_en.max.panel` | 0.7634 | n/a |
| `laya_en_en.max.p2` (melhor sinal isolado) | 0.8172 | n/a |
| `xnli_en` | 0.7539 | n/a |
| `xnli_en_m2m100` | 0.7441 | n/a |

Comparações declaradas C01 a C14 e T01 a T11: todas as diferenças de ROC AUC têm Holm 1.0. O kappa de todo
sistema contra o do juiz de referência fica de -0.4159 a -0.4853, Holm por volta de 0.02. As comparações
com o Jev não existem. Sensibilidade à ordem das opções (`order change rate`): `laya_multi_pt` 0.041 na
pergunta NLI e 0.254 na de posição; `laya_en_en` 0.010 e 0.069. Na validação, o juiz
`prompt_2_deepseek-chat` tem kappa 0.7114, maior que o do juiz de referência escolhido no train (0.6909).

**Decisão.** Nenhum sinal isolado ou painel separa com significância do cosseno; o sinal está distribuído
entre perguntas e checkpoints, o que motivou combinar os sinais (E3x).

### 5.3 E3x: verificador aprendido

**Problema.** Combinar os sinais do E3 num classificador sem usar a validação para escolher.

Configuração: `configs/nli_verifier_exploration.toml` (e3x_v1) e `configs/nli_verifier_exploration_v2.toml`
(e3x_v2). Seleção por validação cruzada só no train: `StratifiedGroupKFold` de 5 dobras com 5 repetições
(25 dobras), grupos = audiência, estratificado por rótulo, semente 20260925 mais a repetição; C da
regressão logística escolhido por uma validação interna de 3 dobras. Regra de seleção: o melhor candidato
pela média de ROC AUC nas 25 dobras; elegíveis os que ficam a até um erro padrão dele; escolhido o mais
simples entre os elegíveis (primeiro por tipo: sinal, painel, aprendido; depois por número de
componentes). A média de CV do escolhido tem viés para cima pela seleção e não é reportada como desempenho
esperado (`selection.json`, `selection_rule`). Corte: `max_f1_not_inferable`, ajustado no train inteiro;
o sistema prediz inferível quando o escore é maior ou igual ao corte.

| rodada | candidatos | selecionado | CV ROC AUC | observação |
|---|---|---|---|---|
| e3x_v1 | 1709 | `laya_multi_pt:learned_cross_model` (29 variáveis) | 0.8506 | nunca confirmado na validação |
| e3x_v2 | 1712 | `laya_en_en:learned_cross_model_with_pt` (53 variáveis) | 0.8657 (erro padrão 0.011) | 3 elegíveis; melhor e selecionado coincidem |

Referências de CV: painel `laya_multi_pt` 0.7899; `xnli` 0.7621. Divulgação obrigatória: os candidatos
novos do e3x_v2 foram acrescentados depois de ver o resultado do e3x_v1 e resultados de validação do E3;
a seleção dentro do e3x_v2 foi só no train, mas o conjunto de candidatos não é independente da validação.

O primário tem 53 variáveis: sinais do `laya_en_en`, do `laya_multi_pt`, do `xnli_mdeberta` e do cosseno
Serafim, em pools máximo, média e concatenado. C = 0.01 (médias internas: 0.8681 para C 0.01, 0.8674 para
0.1, 0.8618 para 1.0 e 0.8576 para 10.0). Os maiores coeficientes padronizados são do cosseno
`sentence_max` (0.304), do cosseno máximo (0.2045) e do `xnli` máximo (0.1742); a lista completa está em
`final_test.json`.

#### Confirmação na validação

Fonte: `artifacts/experiments/nli_verifier_exploration/e3x_v2/confirmation.json` e `confirmation.csv`
(196 candidatos avaliados). Primário: ROC AUC 0.8758 [0.8218; 0.915], kappa 0.4411, corte 0.7478. Δ contra
o painel `laya_multi_pt` +0.0944 e contra o `xnli` +0.106, ambos Holm 0.7832.

#### Teste final

Declarado em 25/09/2026, depois da confirmação e antes de traduzir ou pontuar qualquer opinião de teste: a
pedido do usuário, os cinco candidatos de maior ROC AUC na validação foram avaliados uma vez no teste, ao
lado das duas referências (`configs/nli_verifier_exploration_v2.toml`, `declared` da seção de teste). Fonte:
`artifacts/experiments/nli_verifier_exploration/e3x_v2/final_test.json`. 559 opiniões, 30 audiências, 89
não inferíveis. Bootstrap pareado de 1000 réplicas de audiências, semente 42.

| sistema (teste) | ROC AUC [IC 95%] | kappa | outras |
|---|---|---|---|
| `laya_en_en:learned_cross_model_with_pt` (primário) | 0.9122 [0.8698; 0.9454] | 0.5924 [0.441; 0.7108] | F1 0.6633, precisão 0.6075, revocação 0.7303 (classe não inferível) |
| `laya_en_en:learned_cross_model` | 0.9074 | 0.6255 | n/a |
| `laya_multi_pt:learned_cross_model` | 0.9069 | 0.6098 | n/a |
| `laya_en_en_m2m100:learned_cross_model` | 0.9021 | 0.5967 | n/a |
| `laya_en_en:learned` | 0.8683 | 0.5271 | n/a |
| `laya_multi_pt:panel:max` (referência) | 0.8571 [0.7899; 0.9071] | 0.4985 | n/a |
| `xnli:max` (referência) | 0.8009 [0.7197; 0.8643] | 0.3481 | n/a |

| comparação no teste | Δ ROC AUC [IC 95%] | Holm |
|---|---|---|
| primário contra painel `laya_multi_pt` | +0.0551 [0.0228; 0.1001] | 0.024 |
| primário contra `xnli` | +0.1113 | 0.020 |
| `laya_en_en:learned` contra painel | +0.0112 | 0.54 |

**Decisão** (`PIPELINE.md`). O primário do E3x é o verificador adotado, com o corte do train 0.7478. O kappa
dele no teste (0.5924) fica abaixo do kappa dos juízes LLM do dataset (0.6909 do juiz de referência na
validação); ele ordena bem, mas não substitui um juiz como classificador binário. O kappa do teste
(0.5924) e o da validação (0.4411) diferem bastante; o teste tem mais negativos (89 de 559 contra 76 de
698).

### 5.4 Tradução

**Problema.** O primário lê premissa e hipótese em inglês. A tradução precisa ser estável e cobrir todos os
trechos.

Fontes: `artifacts/experiments/translation/translation_v1_report/report.json`,
`artifacts/experiments/translation/translation_v1_m2m100_report/report.json` (train e validação),
`translation_final_test_nllb/report.json` e `translation_final_test_m2m100/report.json` (teste), e
`translation_v1/report.json`, `translation_v1_m2m100/report.json` (tempos).

| medida (train e validação) | NLLB-200 distilled 600M | M2M100 418M |
|---|---|---|
| opiniões | 3679 (2981 train, 698 validação), 176 audiências | igual |
| textos de opinião distintos | 3681 | 3681 |
| segmentos distintos de trecho | 41599 | 41599 |
| segmentos degenerados | 56 (44 sem repetição na fonte) | 103 (90) |
| segmentos que atingiram `max_new_tokens` | 6 | 57 |
| segmentos idênticos à fonte | 0 | 2 |
| opiniões degeneradas | 0 | 1 |
| tempo da rodada | 9867.4 s para 45224 textos (4.5832 textos/s) | 4427.33 s para 39876 textos (9.0068 textos/s) |
| ROC AUC no teste de `laya_en_en:learned_cross_model` | 0.9074 | 0.9021 |

No teste: 559 opiniões, 30 audiências, 6332 segmentos; NLLB com 14 segmentos degenerados e M2M100 com 29;
nenhuma opinião degenerada. A segmentação dos trechos junta partes terminadas em abreviatura (`Sr.`, `Dr.`,
`art.` e outras; 1561 partes no train e validação) para não quebrar frases; 45273 textos distintos foram
pedidos na rodada NLLB.

Uma sonda de rótulo mostra a diferença qualitativa: a frase "Os homens estão cuidadosamente colocando as
malas no porta-malas de um carro." vira "The men are carefully putting their bags in the trunk of a car."
no NLLB e "Men are carefully putting their bags in a car’s portfolio." no M2M100 (`label_probes`).

**Decisão.** NLLB, por ter cerca de metade das saídas degeneradas; o efeito no AUC está dentro dos
intervalos. A checagem manual de tradução (`artifacts/validation/translation_spot_check_v2/`) não foi
preenchida. A licença do NLLB é CC-BY-NC-4.0, não comercial.

### 5.5 E5: sinais de confiança derivados do recuperador (`confidence_v1`)

**Problema.** Antes de usar um modelo novo, verificar se sinais baratos do próprio recuperador (escore
top-1, margem para o segundo, z-score) dizem quando o top-1 está certo.

Declaração de 23/09/2026, antes de qualquer resultado: alvo primário `serafim_335m.sentence`, métrica AURC
na validação, população com mais de um candidato; um sinal só é dito melhor que `top1_score` se o IC
pareado de 95% da diferença de AURC excluir zero; `nli` é o benchmark da conclusão (`primary.declaration`).
Regressões e cortes ajustados só no train. 1000 réplicas de audiências. Fonte:
`artifacts/experiments/confidence/confidence_v1/confidence_v1_report.json`, chave
`results.nli.serafim_335m.sentence.evaluate.validation.signal_metrics.multi_candidate`.

| sinal (`nli`, validação, 620 consultas, 544 corretas) | ROC AUC [IC 95%] | AURC [IC 95%] | Δ AURC vs `top1_score` [IC 95%] |
|---|---|---|---|
| `top1_score` (referência) | 0.6288 [0.5776; 0.6836] | 0.0797 [0.0537; 0.1086] | n/a |
| `margin` | 0.6774 [0.6124; 0.7372] | 0.0685 [0.0452; 0.0948] | -0.01111 [-0.028492; 0.004885] |
| `zscore` | 0.6225 [0.5425; 0.7036] | 0.0839 [0.0521; 0.1220] | +0.004225 [-0.014063; 0.025334] |
| `entailment.xnli_mdeberta` | 0.5787 [0.5170; 0.6371] | 0.0939 [0.0649; 0.1261] | +0.014287 [-0.008244; 0.039739] |
| `logistic.scores` | 0.6767 [0.6148; 0.7325] | 0.0685 [0.0447; 0.0941] | -0.011119 [-0.025711; 0.001099] |
| `logistic.scores_entailment` | 0.6806 [0.6186; 0.7363] | 0.0680 [0.0442; 0.0934] | -0.011707 [-0.026432; 0.000828] |

AURC do oráculo: 0.007939. No `masked_quotes` (40 consultas, 4 corretas) os intervalos são largos, como a
declaração previa; a única diferença que exclui zero é desfavorável (`logistic.scores_entailment`, Δ AURC
+0.14686 [0.007215; 0.31139]).

O corte de produção 0.45 aceita 619 de 620 consultas na validação (cobertura 0.998387, precisão 0.877221):
ele separa níveis, mas não filtra erros de top-1 (`production_rule.all_queries.validation`).

**Decisão.** Nenhum sinal supera `top1_score` pela regra declarada. Sinais derivados do cosseno não resolvem
o Trabalho 2.

### 5.6 confidence_v2 E-A: o verificador contra avaliadores da literatura

Fonte: `artifacts/experiments/confidence_v2/ea_report.json` (criado 2026-09-28T09:58:40Z) e
`CONFIDENCE_V2.md`.

**Problema.** Confirmar que o primário do E3x é uma confiança melhor que o cosseno e compará-lo com
avaliadores de consistência publicados, no mesmo rótulo e na mesma premissa.

Desenho: validação do benchmark NLI (698 opiniões, 32 audiências, 76 negativos), premissa de 4 trechos com
máximo por trecho; família de Holm com 30 comparações (15 candidatos contra o cosseno em ROC AUC e AURC);
5000 réplicas de audiências, semente 42; o teste do benchmark não foi lido. O primário foi reajustado no
train e confere com `final_test.json` (`primary.refit_check`: C, médias internas, coeficientes e corte
idênticos). Menor p possível: 0.0004; vezes 30 dá o 0.012 das linhas mais fortes.

| sinal | ROC AUC [IC 95%] | AURC [IC 95%] | Δ AUC vs cosseno [IC 95%], Holm | Δ AURC vs cosseno [IC 95%], Holm |
|---|---|---|---|---|
| `cosine_serafim` | 0.7302 [0.6552; 0.7975] | 0.0544 [0.0358; 0.0741] | referência | referência |
| `e3x_primary` | 0.8758 [0.8231; 0.9149] | 0.0229 [0.0135; 0.0363] | +0.1457 [0.0852; 0.2159], 0.012 | -0.0315 [-0.0448; -0.0176], 0.012 |
| `bge_reranker` | 0.8416 [0.7712; 0.9] | 0.033 [0.0175; 0.054] | +0.1115 [0.0636; 0.1585], 0.012 | -0.0214 [-0.0331; -0.0085], 0.1056 |
| `minicheck_ft5` | 0.8337 [0.7704; 0.8869] | 0.0319 [0.0178; 0.051] | +0.1036 [0.0361; 0.175], 0.1 | -0.0225 [-0.0342; -0.0094], 0.012 |
| `minicheck_deberta` | 0.8203 [0.7645; 0.868] | 0.0341 [0.0211; 0.0501] | +0.0901 [0.0228; 0.1609], 0.2464 | -0.0204 [-0.0339; -0.0062], 0.1104 |
| `factcg_deberta` | 0.8198 [0.7713; 0.8627] | 0.0328 [0.0197; 0.0497] | +0.0896 [0.0183; 0.1662], 0.3695 | -0.0217 [-0.0332; -0.0083], 0.0728 |
| `hhem_open` | 0.8153 [0.7385; 0.8771] | 0.0375 [0.0205; 0.0625] | +0.0851 [0.0112; 0.1552], 0.4799 | -0.0169 [-0.0328; 0.0035], 1.0 |
| `alignscore_large` | 0.8055 [0.7347; 0.8659] | 0.0376 [0.0223; 0.0573] | +0.0754 [0.0063; 0.1478], 0.5831 | -0.0169 [-0.0299; -0.0021], 0.5699 |
| `laya_multi_pt_p4` | 0.7925 [0.7215; 0.8462] | 0.0401 [0.0255; 0.0578] | +0.0623 [-0.0128; 0.1373], 1.0 | -0.0143 [-0.0272; 0.0004], 0.9586 |
| `xnli_mdeberta` | 0.7698 [0.6906; 0.8347] | 0.0401 [0.0238; 0.062] | +0.0396 [-0.0427; 0.1228], 1.0 | -0.0143 [-0.0289; 0.0028], 1.0 |
| agregações do Laya com 7 perguntas (6 variantes) | 0.7686 a 0.8068 | 0.0365 a 0.0528 | Holm de 0.9586 a 1.0 | Holm de 0.9586 a 1.0 |

Combinações por posto (família separada de 4 comparações): `rank_mean` 0.8333 e `rank_max` 0.798 de ROC
AUC, ambos acima do cosseno com Holm 0.0016, e ambos abaixo do primário sozinho. Com premissa concatenada
(secundária, fora de Holm): `minicheck_ft5` 0.869, `factcg_deberta` 0.855, `hhem_open` 0.848,
`bge_reranker` 0.830, `minicheck_deberta` 0.817 (`results.validation.concatenated`). Perguntas novas do Laya
(q11) somam no máximo +0.0065 de ROC AUC às de q7 (`added_questions`). A dispersão entre perguntas do Laya
não serve como sinal (ROC AUC 0.4721 para `laya_multi_pt_spread_q7`).

Leitura: o primário tem a maior ROC AUC e o menor AURC de todos os sinais e é o único que supera o cosseno
nas duas métricas após Holm. A comparação direta do primário com cada avaliador da literatura não foi
declarada; a distância para o mais próximo (`bge_reranker`, 0.034 de ROC AUC) é só uma estimativa pontual.

Não rodaram, com o motivo registrado (`CONFIDENCE_V2.md`): juízes LLM (Qwen3-4B-Instruct-2507, Qwen3-1.7B,
Bespoke-MiniCheck-7B) e Granite Guardian 3.2-3B, por decisão registrada antes de qualquer escore; Lynx e
TRUE, por licença e memória; SummaC, por ser uma agregação de NLI por frase já coberta; MiniCheck-RoBERTa,
por redundância; LettuceDetect, por marcar tokens e não ter português confirmado.

**E-B.** O `confidence_v2` também tem uma parte E-B, sobre 121 UDVs da amostra de validação. Ela usa
rótulos de referência que não são julgamento humano e por isso é omitida deste relatório.

### 5.7 Camada do verificador sobre udv_v1

Fonte: `artifacts/udv/udv_v1_verifier_report.json`.

**Problema.** Ver como a decisão do primário se distribui nas UDVs reais, com premissa de uma sentença.

| nível de `udv_v1` | com evidência | passa em 0.7478 |
|---|---|---|
| `quote_found` | 277 | 78 (28.2%) |
| `semantic_match_high` | 1785 | 411 (23.0%) |
| `semantic_match_weak` | 43 | 0 (0%) |
| todas | 2105 | 489 (23.2%) |

Por tipo de suporte, `semantic_with_short_quote` passa em 47 de 111. Spearman entre probabilidade e cosseno:
0.6909 em `semantic_match_high` e 0.7088 nos dois níveis semânticos. O relatório também tem uma seção
`annotation_agreement` construída com rótulos que não são humanos; ela não é usada aqui.

**Decisão.** O corte do benchmark reprova 72% das citações literais com premissa de uma sentença, o que
indica que a premissa curta deixa o verificador conservador. Isso motivou `udv_v2` (seção 3.5).

### 5.8 E6: casamento aproximado de citações e nomes (`fuzzy_v1`)

Este experimento atende aos dois trabalhos: recuperar citações e nomes que as regras exatas perdem
(Trabalho 1) e medir o risco de aceitar casamentos ao acaso (Trabalho 2).

Fonte: `artifacts/experiments/fuzzy/fuzzy_v1_report.json` (criado 2026-09-23T16:41:41Z, train e validação;
referência `udv_v1_pre`). Bootstrap de 1000 réplicas de audiências.

| medida (train) | valor | chave |
|---|---|---|
| opiniões do train | 1536; 1456 de pessoa resolvida; 651 com citação extraída | `quotes.train.funnel` |
| controle positivo (citação exata confiável) | 183; todas com escore 100 nos dois métodos | `quotes.train.positive_control` |
| concordância do encoder com a sentença da citação exata | 112 de 183, 0.612 [0.53; 0.6884] | `positive_control.encoder_baseline` |
| população sem citação confiável, com citação elegível | 436 | `quotes.train.funnel` |
| com casamento aproximado de caracteres maior ou igual a 80 | 317 opiniões em 132 audiências | `quotes.train.gains.char.80.0` |
| concordância desse trecho com o top-1 do cosseno | 167 de 317, 0.5268 [0.4757; 0.5775] | idem |
| taxa ao acaso, outra pessoa da mesma audiência | 61 de 2165, 0.0282 [0.0149; 0.0432]; 12.3 casamentos esperados ao acaso | `gains.char.80.0.null` |
| taxa ao acaso, outra audiência do mesmo split | 7 de 2180, 0.0032 [0.001; 0.0055] | idem |
| trechos com escore maior em outro falante | 6 no corte, 1 com escore maior que o da própria pessoa | `elsewhere_*` |
| participantes não resolvidos | 39 (80 opiniões, 30 audiências) | `names.train` |
| resolvidos por `token_set_ratio` maior ou igual a 80 | 17 (37 opiniões), todos únicos; 5 com apoio de citação confiável | `names.train.by_rule.token_set_ratio.80.0` |
| controle de impostor (706 participantes resolvidos) | 0.0042 [0.0; 0.0097] | idem, `impostor_control` |

A concordância com o encoder é um indicador fraco de precisão, porque o encoder também vê o texto da
citação. Deve ser lida contra a concordância no controle positivo (0.612), e não contra 1.

**Decisão.** Fica desligado. As planilhas de revisão (`fuzzy_v1_quotes_review.jsonl`, 458 linhas, e
`fuzzy_v1_names_review.jsonl`, 70 linhas, segundo `PIPELINE.md`) não têm julgamento; o cálculo de precisão
(`uv run --no-sync python -m utils.fuzzy_review_precision`) só roda com elas preenchidas.

## 6. Atores e regras de ligação

### 6.1 Falas por ator

**Problema.** Um perfil de ator precisa da fala completa da pessoa em todas as audiências. A matéria cita
poucas pessoas; a transcrição tem a fala inteira, mas o nome aparece com grafias diferentes entre
audiências.

Fontes: `challenge/HEARING_ACTORS.md`, `../bookworm/docs/actors.md`,
`artifacts/hearing_actors/measurements.json`, `artifacts/hearing_actors/actor_speeches_stats.json` e
`artifacts/hearing_actors/ambiguous_names.json`. Comandos: `uv run python -m utils.measure_hearing_actors` e
`uv run python -m utils.build_actor_speeches`; na biblioteca, `uv run bookworm build-udvs --run-name udv_v1
--config configs/udv.toml --actors-config configs/hearing_actors.toml`.

| medida | valor |
|---|---|
| turnos | 17264; 13190 mantidos e conferidos contra a transcrição |
| turnos descartados | 4001 de presidência com menos de 50 palavras, 32 de chaves que não são pessoa, 36 só de rubricas, 5 vazios |
| fusões | 37 grupos (40 aliases, 130 turnos) e 1 reatribuição (1 turno) |
| atores | 1851: 1550 em uma audiência e 301 em duas ou mais |
| pares de nomes ambíguos restantes | 81 (1 com as mesmas palavras, 80 com nome contido) |
| revisão manual dos pares | 153 pares iniciais: 41 confirmados (os 37 grupos), 110 pessoas distintas, 1 resolvido por divisão de chave, 1 sem resolução |
| UDVs com ator | 2104 de 2203; 90 de pessoas não resolvidas, 9 de pessoas cujos turnos foram todos descartados |
| UDVs de ator com duas ou mais audiências | 726; ligações cobrem 827 atores distintos |

Por que o corte de 50 palavras nos turnos de presidência: 42.7% das palavras dos falantes recorrentes estão
em turnos de presidência, boa parte condução de sessão; mas 336 das 2.105 opiniões com evidência de `udv_v1`
estão em turnos de presidência (56 citações literais). O corte de 50 mantém 2264 de 6266 turnos de
presidência, 89.0% das palavras e 332 das 336 opiniões (`HEARING_ACTORS.md`, tabela de cortes).

Por que não filtrar por partido: dos 282 falantes recorrentes (contagem antes das fusões), 73 não têm
partido no cabeçalho, incluindo ministros, secretários e dirigentes da sociedade civil. O critério de
entrada é o número de audiências; o campo `has_party_header` permite separar deputados federais.

### 6.2 Duas regras de ligação UDV para ator

**Problema.** As contagens de avaliação de perfis dependem de qual regra liga uma UDV a um ator. Havia duas
no repositório.

- **Turnos atribuídos** (`bookworm.pipeline.resolve_link`): entre os turnos que a resolução atribuiu à
  pessoa da UDV, fica o ator com mais turnos mantidos.
- **Turno de evidência** (bloco `evaluation` de `filter-actor-speeches`): fica o dono do turno
  `evidence.speaker_turn`.

Resultado sobre `udv_v1` (`../bookworm/docs/actors.md`, "Duas regras de ligação"; fixado por
`tests/integration/test_link_rules.py`):

| regra | UDVs ligadas |
|---|---|
| turnos atribuídos | 2104 |
| turno de evidência | 2099 |
| divergências nas UDVs ligadas pelas duas | 0 |

As 5 ligações a mais da primeira regra são UDVs cujo turno de evidência foi descartado pela política, mas
que têm outros turnos mantidos da mesma pessoa. Em 6 UDVs o nome de exibição do ator difere do nome da
matéria por causa do nome entre parênteses no cabeçalho ou de uma fusão; as duas regras escolhem o mesmo
ator nelas. No split de teste, as duas regras dão 106 UDVs ligadas a atores com perfil, 48 atores e 27
audiências.

### 6.3 Perfis de ator e simulação

Fontes: `challenge/ACTOR_PROFILES.md`, `challenge/ACTOR_SIMULATION.md`, `../bookworm/docs/profiles.md`,
`../bookworm/docs/profile_validation.md` e `artifacts/actor_profiles/train_speeches_stats.json`.

**Problema.** Um perfil gerado com as 206 audiências já leu as falas do teste; avaliá-lo contra as opiniões
do teste mediria cópia. Por isso o perfil de avaliação usa só audiências de treino.

| medida (`train_speeches_stats.json`) | valor |
|---|---|
| atores recorrentes na entrada | 301, em 198 audiências, 6323 turnos |
| depois do filtro de treino | 264 atores, 139 audiências, 4598 turnos; 37 atores sem audiência de treino; 79 com uma só |
| atores com perfil que falam no teste | 116 |
| UDVs de teste de `udv_v1` | 359, das quais 106 ligadas a 48 atores com perfil, em 27 de 30 audiências |
| níveis dessas 106 | 17 `quote_found`, 87 `semantic_match_high`, 2 `semantic_match_weak` |

Implementação: o gerador de perfis (`utils/generate_actor_profiles.py`), a conferência de perfis contra
UDVs (`bookworm validate-profiles`) e a simulação de atores com três abordagens (só perfil; perfil com
recuperação; perfil com recuperação e classifier-free guidance) estão em `ACTOR_PROFILES.md` e
`ACTOR_SIMULATION.md`.

Rodada completa com o backend MLX (`mlx_alternative/README.md`), modelo `mlx-community/Qwen3.8-27B-8bit`,
27 e 28/09/2026, num Apple M4 Max de 128 GB. Fontes: `mlx_alternative/runs/qwen38_27b/actor_simulation/evaluation.json`
e `mlx_alternative/runs/qwen38_27b/relatorio_rodada_completa.md`. Perfis gerados só com as audiências de
treino (264 atores, nenhuma falha); k e γ escolhidos na validação; resultado no teste. Múltipla escolha
com 4 proposições da mesma audiência (acaso 0.25), lida pelas probabilidades das letras no primeiro token,
média de 4 rotações; intervalos por bootstrap pareado de 1000 reamostragens de audiências.

| condição (teste, 101 perguntas, 46 atores, 24 audiências) | acerto |
|---|---|
| 0, nome e cargo | 0.3069 |
| 1, perfil | 0.4257 |
| 2, perfil e trechos (k = 3) | 0.4455 |
| 3, condição 2 com guidance (γ = 1.0) | 0.4455 |

| diferença (teste) | acerto [IC 95%] |
|---|---|
| 1 menos 0 | +0.1188 [0.0495; 0.1965] |
| 2 menos 1 | +0.0198 [-0.0286; 0.0702] |
| 3 menos 2 | 0 [0; 0] |

Na validação (81 perguntas), k = 3 e γ = 1.0 tiveram o maior acerto da grade (`selection`), então a
guidance não mudou nada. Os níveis de evidência declarados pelo modelo não ordenaram o acerto como o
desenho exige (ESPECULATIVA acima de INDIRETA nas condições 1 e 2). Limites: um modelo, quantizado em 8
bits, uma semente; o prompt de múltipla escolha foi revisado depois de um piloto que usou perguntas de
teste, com base na massa das letras e sem uso do acerto (`ACTOR_SIMULATION.md`). A conferência de perfis
(`bookworm validate-profiles`) não tem rodada versionada.

## 7. Validação humana (parcial)

### 7.1 Desenho da amostra

**Problema.** Nenhum benchmark automático mede se o trecho da UDV sustenta a opinião do LDS. Isso exige
julgamento humano, sobre audiências que nenhuma escolha de método usou.

Fonte: `artifacts/validation/human_validation_v1_udv_v1/sample_report.json` e `challenge/annotation_guide.md`.
Comando: `uv run --no-sync python -m utils.generate_validation_sample --run-name udv_v1 --final-test`.

| item | valor |
|---|---|
| rodada amostrada | `udv_v1` (corte 0.45 calibrado sem audiências de teste) |
| split | só test (a validação foi usada para escolher o método) |
| linhas, UDVs, audiências | 127 linhas, 134 UDVs, 30 audiências |
| sorteio | numpy PCG64, semente 42, fluxos separados para estrato, ordem das linhas e repetição |
| critérios | declarados em 23/09/2026, antes de qualquer julgamento (sha256 `c4fc0116…`) |
| reanotação | 20 linhas em outra ordem, pelo menos 24 horas depois, sem consultar a primeira rodada (planejada) |
| cegamento | colunas de estrato, nível, tipo de suporte, escore e prefixo ficam só na chave |

| estrato | pergunta | população no teste | alvo | sorteadas |
|---|---|---|---|---|
| `direct_quote` | `trecho_sustenta` | 41 | 35 | 35 |
| `semantic_match_high` | `trecho_sustenta` | 277 | 65 | 65 |
| `semantic_with_short_quote` | `trecho_sustenta` | 22 | 15 | 15 |
| `semantic_match_weak` | `trecho_sustenta` | 6 | 15 | 6 (todas) |
| `speaker_check` | `pessoa_falou` | 13 UDVs | 20 | 13 UDVs em 6 linhas (todas) |

Rótulos de `trecho_sustenta`: `correta` (o trecho sustenta o conteúdo principal, inclusive posição e números),
`parcial` (sustenta só parte, ou o mesmo assunto sem o detalhe principal, ou o apoio está no contexto) e
`incorreta` (outro assunto, posição oposta, procedimento ou fala de outra pessoa). Precisão estrita conta só
`correta`; tolerante conta `correta` ou `parcial`. O julgamento não avalia se a afirmação é verdadeira no
mundo.

Critérios declarados (PASS exige que o limite inferior de Wilson de 95% alcance o mínimo):

| critério | estrato | métrica | mínimo | viabilidade no tamanho sorteado |
|---|---|---|---|---|
| `quote_found_strict_precision` | `direct_quote` | precisão estrita | 0.9 | 35 acertos em 35 (nenhum erro permitido) |
| `semantic_match_high_tolerant_precision` | `semantic_match_high` | precisão tolerante | 0.75 | 56 em 65 (até 9 erros) |

Vazamentos conhecidos do cegamento (`sample_report.json`, `blinding.known_leaks`): itens `direct_quote` e
`semantic_with_short_quote` são reconhecíveis porque as palavras citadas reaparecem no trecho; só a
distinção entre `semantic_match_high` e `semantic_match_weak` fica cega.

### 7.2 Recálculo parcial em 28/09/2026

O relatório final exige as duas planilhas completas (`annotation_guide.md`). Este relatório traz um
recálculo intermediário, feito a pedido para o artigo, sobre as linhas já julgadas. Ele não é uma decisão
sobre os critérios.

Comando (saída gravada fora do repositório):

```
cd challenge && uv run python -m utils.udv_v2_analysis score-annotation --final-test \
  --annotation artifacts/validation/human_validation_v1_udv_v1/annotation.csv \
  --output <arquivo temporário>
```

Os números abaixo só se reproduzem com a versão da planilha de sha256 indicado a seguir.

Estado da planilha lida: sha256 `5d4de458fbd1ad86c0dedccaaa180b1640fb96c8a50d05071b581dba5010bbc1`; 65 de 127
linhas julgadas; 0 rótulos inválidos; o campo `existe_trecho_melhor` está vazio nas 65. Saída gerada em
2026-09-28T15:27:47Z, status `interim`.

| estrato (`udv_v1`) | julgadas | correta / parcial / incorreta | precisão estrita [Wilson 95%] | precisão tolerante [Wilson 95%] |
|---|---|---|---|---|
| `direct_quote` | 18 de 35 (população 41) | 17 / 0 / 1 | 0.9444 [0.7424; 0.9901] | 0.9444 [0.7424; 0.9901] |
| `semantic_match_high` | 32 de 65 (população 277) | 21 / 4 / 7 | 0.6562 [0.4831; 0.7959] | 0.7812 [0.6125; 0.8898] |
| `semantic_with_short_quote` | 8 de 15 (população 22) | 8 / 0 / 0 | 1.0 [0.6756; 1.0] | 1.0 [0.6756; 1.0] |
| `semantic_match_weak` | 4 de 6 | 1 / 1 / 2 | 0.25 [0.0456; 0.6994] | 0.5 [0.15; 0.85] |
| `speaker_check` | 3 de 6 linhas (6 UDVs; população 13) | todas `falou` | n/a | n/a |

No `speaker_check`, todas as respostas foram `falou`: a taxa de falsa ausência (a pessoa falou, mas a
pipeline não registrou fala utilizável) é 1.0 [0.4385; 1.0] por pessoa e 1.0 [0.6097; 1.0] por UDV, com
3 pessoas e 6 UDVs.

Estado dos critérios, só como leitura intermediária:

| critério | valor atual | situação |
|---|---|---|
| `quote_found_strict_precision` | limite inferior 0.7424 contra 0.9 | não pode mais passar: exige 35 de 35 e já há 1 `incorreta` (`udv-112-4-0`) |
| `semantic_match_high_tolerant_precision` | limite inferior 0.6125 contra 0.75 | indefinido: 7 erros em 32, até 9 permitidos em 65 |

Subconjunto sem mudança de evidência em `udv_v2` (`udv_v2_unchanged`): 23 itens da amostra; `direct_quote`
10 de 11, 0.9091 [0.6226; 0.9838]; o critério semântico não é avaliável nesse subconjunto.

Todas as 18 linhas `direct_quote` julgadas e as 8 `semantic_with_short_quote` têm a pista de citação
visível; as 32 de `semantic_match_high` não têm (`by_quote_cue`). Os números de `direct_quote` e
`semantic_with_short_quote` não foram obtidos às cegas quanto ao modo de busca.

Uma revisão exploratória anterior, não cega e sem protocolo, foi retirada desta versão (fica na tag Git
`research-2026-09-28`); ela não é validação e não entra em nenhum número deste relatório.

## 8. Glossário de métricas

Cada métrica está explicada pelo problema que resolve.

| métrica | problema que resolve | definição |
|---|---|---|
| posto (rank) | saber onde o recuperador colocou a primeira unidade relevante | posição, a partir de 1, da primeira unidade relevante na ordem por escore decrescente; empates por índice de candidato crescente |
| `rank_optimistic`, `rank_pessimistic` | medir quanto empates podem mudar o resultado | 1 + número de irrelevantes com escore estritamente maior (otimista) ou maior ou igual (pessimista) ao da melhor relevante |
| acc@k | taxa de consultas em que o leitor acha a evidência entre as k primeiras | fração de consultas com posto menor ou igual a k |
| MRR | resumir a ordem inteira, não só o top-1 | média de 1 / posto |
| acc@1 ao acaso | comparar unidades de tamanhos diferentes | média de n_relevantes / n_unidades por consulta |
| lift | comparar unidades descontando o acaso | por consulta, 1 se o top-1 é relevante, senão 0, menos n_relevantes / n_unidades |
| ROC AUC | medir se um sinal ordena positivos acima de negativos, sem escolher limiar | probabilidade de um positivo ter sinal maior que um negativo, empates contam metade (Mann-Whitney) |
| precisão média (AP) | o mesmo, com peso na parte alta da ordem | média da precisão a cada positivo; ao acaso vale a taxa base |
| risco | quanto erro passa quando se aceitam os itens de maior confiança | fração de incorretos entre os aceitos |
| AURC | resumir o risco em todas as coberturas | média do risco nas coberturas j/n, j = 1..n, itens ordenados por sinal decrescente; menor é melhor |
| E-AURC | separar a ordenação da taxa base de erro | AURC menos o AURC de um oráculo que põe todos os corretos primeiro |
| precisão a c de cobertura | descrever a qualidade da ordem num ponto | 1 menos o risco nos primeiros ceil(c n) itens; o corte é feito no próprio conjunto avaliado e não é uma política implantável |
| kappa de Cohen | concordância além do acaso entre uma decisão binária e o rótulo | (p_o - p_e) / (1 - p_e), com p_o a concordância observada e p_e a esperada pelas marginais |
| F1, precisão, revocação | qualidade da decisão binária na classe não inferível | com o corte do train; o sistema prediz inferível quando escore maior ou igual ao corte |
| `max_f1_not_inferable` | escolher o corte do verificador sem usar validação | corte que maximiza o F1 da classe não inferível no train |
| índice J de Youden | escolher um corte que equilibre sensibilidade e especificidade | J = sensibilidade + especificidade - 1, maximizado no corte |
| regra `legacy_random` | calibrar o corte de cosseno com pares de treino | ponto médio entre o quartil 25 dos positivos (opinião contra a sentença ou janela da própria citação) e o quartil 75 dos negativos (outra audiência do train) |
| regra `hard_negative` | idem, com negativo mais difícil | negativo = outra sentença ou janela da mesma pessoa |
| precisão estrita e tolerante | medir a validação humana | fração de UDVs julgadas `correta` (estrita) ou `correta`/`parcial` (tolerante) |
| intervalo de Wilson | intervalo de proporção que se comporta bem com n pequeno e proporções perto de 0 ou 1 | intervalo de escore de Wilson de 95% |
| taxa de falsa ausência | medir se a pipeline deixa de achar quem falou | fração de respostas decididas do `speaker_check` que são `falou` |
| bootstrap por audiência | intervalos que respeitam a dependência entre opiniões da mesma audiência | reamostragem com reposição de audiências inteiras; intervalo percentil; o número de réplicas é dado em cada experimento |
| bootstrap pareado e p | comparar dois sistemas nas mesmas réplicas | Δ = candidato menos referência em cada réplica; p bilateral (k + 1) / (B + 1) |
| McNemar exato | comparar acc@1 de dois recuperadores nas mesmas consultas | teste binomial bilateral nas consultas em que só um acerta; trata consultas como independentes, por isso o IC de bootstrap é mostrado ao lado |
| permutação por sinal de audiências | comparar MRR ou lift respeitando audiências | troca aleatória do sinal da diferença de audiências inteiras, 10000 amostras |
| Holm | controlar o erro de família em comparações múltiplas | ajuste sequencial de Holm dentro de cada família declarada |
| DeLong | comparar ROC AUC correlacionadas | usado só no E-B do confidence_v2, omitido aqui |
| Spearman | medir se dois sinais ordenam da mesma forma | correlação de postos |

## 9. Software e reprodutibilidade

### 9.1 Ambiente

| item | valor | fonte |
|---|---|---|
| Python | 3.12.13 (a maioria dos relatórios); 3.12.12 no relatório de splits | `environment` de cada relatório |
| numpy, scipy, scikit-learn | 2.5.1, 1.18.0, 1.9.0 | `retrieval_v1_report.json` |
| rank_bm25 | 0.2.2 | idem |
| torch, transformers, huggingface_hub, sentence-transformers | 2.13.0, 5.14.1, 1.24.0, 5.6.1 | `retrieval_v1_report.json`, `configs/nli_verifier.toml` |
| tokenizers | 0.22.2 | relatórios de tradução |
| laya | 0.3.11 | `retrieval_v2_report.json` |
| ambiente de `udv_v2` | torch 2.14.0, sentence-transformers 6.1.0, Python 3.12.13 | `artifacts/udv/udv_v2_coverage.json` |
| plataforma | macOS 26.5.2 arm64 (`macOS-26.5.2-arm64-arm-64bit`), 10 CPUs | relatórios de tradução |
| hardware | Apple M5 com aceleração MPS, fp32; máquina de 24 GB; a rodada MLX de perfis e simulação, num Apple M4 Max de 128 GB | `configs/retrieval_experiments.toml` (`batch_size_note`), `CONFIDENCE_V2.md`, `mlx_alternative/runs/qwen38_27b/relatorio_rodada_completa.md` |
| gerenciador | `uv` com lock por projeto (`bookworm/` e `challenge/`) | estrutura do repositório |
| Hub | `HF_HUB_OFFLINE=1` nas rodadas; revisões de modelo fixadas | `environment.hub_offline` dos relatórios de tradução |

A diferença de versões entre as rodadas antigas (torch 2.13.0, sentence-transformers 5.6.1) e `udv_v2`
(torch 2.14.0, sentence-transformers 6.1.0) está registrada, mas o efeito sobre os vetores não foi medido
separadamente; `udv_v2` usa o cache de embeddings, e o refit do verificador confere coeficientes e corte
idênticos (`udv_v2_verifier_report.json`, `refit_check`).

### 9.2 Sementes e réplicas

| experimento | semente | réplicas |
|---|---|---|
| splits | 42 | n/a |
| calibração de cortes (`threshold_v1`, `threshold_v2`, `udv_verifier_threshold_v1`) | 42 | 1000 |
| retrieval_v1 e retrieval_v2 | 42 | bootstrap 2000, permutação 10000 |
| E3 v1 e v2 | 42 | 1000 |
| E3x | CV 20260925 (+ repetição); bootstrap 42 | 25 dobras; 1000 |
| E5 (`confidence_v1`) | fixa | 1000 |
| E6 (`fuzzy_v1`) | fixa | 1000 |
| confidence_v2 E-A | 42 | 5000 |
| amostra de validação humana | 42 (PCG64) | n/a |
| `udv_v2` | 42 | 1000 |

### 9.3 Comandos principais

Dentro de `challenge/` (salvo indicação):

| etapa | comando |
|---|---|
| baixar o dataset | `uv run python -m utils.download_dataset` |
| splits | `uv run python -m utils.build_splits`; `uv run python -m utils.verify_splits` |
| benchmarks | `uv run python -m utils.build_quote_benchmark`; `uv run python -m utils.build_nli_benchmark` |
| corte de sentença | `uv run python -m utils.calibrate_threshold` |
| UDVs e verificação | `uv run python -m utils.build_udvs --run-name udv_v1`; `uv run python -m utils.verify_udvs --run-name udv_v1` |
| cortes de `udv_v2` | `utils.calibrate_udv_v2` (subcomandos `cosine` e `verifier`) |
| verificador sobre UDVs | `uv run python -m utils.udv_verifier --config configs/udv_v2_verifier.toml translate`, depois o subcomando `apply` |
| análise de `udv_v2` | `uv run python -m utils.udv_v2_analysis analyze`; `score-annotation --final-test --annotation <csv>` |
| recuperação | `utils.retrieval_experiments` (subcomandos `queue`, `run`, `summarize`) |
| E3 | `utils.nli_verifier_experiments` e `utils.translation` (subcomandos no `--help`) |
| E3x | `utils.nli_verifier_exploration` (subcomandos `confirm` e `final-test`, além da seleção) |
| E5 | `utils.confidence_policies` |
| E6 | `utils.fuzzy_matching_experiments`; `uv run --no-sync python -m utils.fuzzy_review_precision` |
| confidence_v2 | `uv run python -m utils.confidence_v2 smoke`, `decide`, `score`, `laya-smoke`, `laya-score`, `evaluate-ea` |
| atores | `uv run python -m utils.measure_hearing_actors`; `uv run python -m utils.build_actor_speeches` |
| amostra humana | `uv run --no-sync python -m utils.generate_validation_sample --run-name udv_v1 --final-test`; relatório final: `uv run --no-sync python -m utils.precision_report --sample-dir artifacts/validation/human_validation_v1_udv_v1 --final-test` |
| `udv_v2` (CLI da biblioteca) | `uv run --project ../bookworm bookworm build-udvs --config configs/udv_v2.toml --run-name udv_v2`; `verify-udvs` com os mesmos argumentos e `--baseline artifacts/udv/udv_v1.jsonl` |
| lint e tipos | `uv run ruff check .`; `uv run ruff format --check .`; `uv run mypy` |

Os relatórios de recuperação, E3 e E3x não gravam a linha de comando usada; eles gravam o sha256 do código,
das entradas e da configuração (`code`, `inputs`, `config`). A reprodução exata passa por esses hashes.

### 9.4 Testes

Rodados nesta data no worktree:

| conjunto | resultado |
|---|---|
| `bookworm`, com `BOOKWORM_LDS_PATH=challenge/dataset/PublicHearingBR_LDS.jsonl` e `BOOKWORM_EMBEDDING_CACHE=challenge/artifacts/cache/embeddings` | 843 passaram, 4 desmarcados (marcador `model`), 84 s |
| `bookworm`, só marcador `dataset` | 95 passaram |
| `bookworm`, sem as variáveis de ambiente | 755 passaram, 88 pulados; cobertura total 97% (4046 linhas, 120 sem cobertura) |
| `challenge` | 105 passaram; na versão de release, 118 passaram, sem o dataset e sem os arquivos pesados |

Os testes de integração com marcador `dataset` refazem, a partir do LDS, a paridade byte a byte de
`udv_v1.jsonl` e dos arquivos de falas por ator com os do `challenge/`, e as contagens das duas regras de
ligação (`../bookworm/docs/actors.md`).

### 9.5 Decisões registradas (ADRs)

Em `../bookworm/docs/adr/`: 0001 estrutura da biblioteca e paridade com o `challenge/`; 0002 segmentação em
sentenças por turno; 0003 aspas simples como citação; 0004 exportações leem só o cache da rodada; 0005 uma
passada pela transcrição para UDVs e perfis de ator; 0006 `udv_v2` com janelas, citação inteira e
verificador.

## 10. Limitações

1. **Validação humana incompleta.** 65 de 127 linhas; nenhum critério decidido; o de `quote_found` não pode
   mais passar. O campo `existe_trecho_melhor` está vazio em todas as linhas julgadas, então a revocação da
   busca (se havia trecho melhor) não é estimável. A reanotação de 20 linhas não foi feita, então a
   concordância intra-anotador não existe. Há um único anotador.
2. **Leitura intermediária da validação.** O protocolo manda não ver números antes de completar a anotação;
   o recálculo parcial deste relatório quebra essa regra por pedido explícito, e quem continuar a
   anotação deve saber disso.
3. **Cegamento parcial.** Citações literais são reconhecíveis na planilha.
4. **A validação humana é de `udv_v1`.** Ela não mede diretamente `udv_v2`, cuja evidência mudou em 1969 dos
   2203 registros. Só 23 itens da amostra têm evidência igual nas duas versões.
5. **Rótulo do Trabalho 2.** O verificador foi ajustado e avaliado no rótulo NLI sob recuperação (quatro
   trechos, opiniões geradas por LLM a partir da transcrição). A aplicação a UDVs (opiniões da matéria,
   premissa de uma sentença ou janela) é uma mudança de domínio não medida com rótulo humano.
6. **O verificador não substitui um juiz.** kappa 0.5924 no teste, abaixo dos juízes LLM do dataset.
7. **Escolha de candidatos do E3x.** Os candidatos do e3x_v2 foram criados depois de ver resultados de
   validação; o teste avaliou os cinco melhores da validação, escolhidos depois da confirmação.
8. **B1 pequeno.** 40 consultas na validação; nenhuma conclusão sobre recuperadores se apoia só nele.
9. **Relevância do B2.** Uma unidade é relevante quando se sobrepõe a um trecho recuperado; o rótulo não diz
   que cada unidade sobreposta sustenta a opinião.
10. **Corte de 6 palavras e política de citação.** Vieram de leitura exploratória sem protocolo.
11. **Tradução.** Não comercial (NLLB) e sem checagem manual preenchida.
12. **Identidade de atores.** Revisão só com evidência interna do dataset; homônimos exatos entre audiências
    ficam juntos sem aviso; um par segue sem resolução.
13. **Perfis e simulação com uma rodada.** Um modelo quantizado, uma semente e 101 perguntas de teste
    (seção 6.3); os níveis de evidência não ordenam o acerto; a conferência de perfis não tem rodada
    versionada.
14. **Versões de biblioteca.** Duas combinações de torch e sentence-transformers entre rodadas, sem medida
    de efeito.
15. **Modelos não rodados.** Jev (declarado no E3 v2), juízes LLM e outros avaliadores do confidence_v2
    (seção 5.6).

## 11. Referências

As referências abaixo foram conferidas em cartões de modelo do Hugging Face, na API do arXiv, no Crossref,
no DataCite e em páginas de periódicos. "A confirmar" marca o que não pôde ser confirmado. As revisões são
os commits fixados nas configurações do repositório.

### 11.1 Dataset

- Fernandes, L. C.; Dobins, G. Z. R.; Lotufo, R.; Pereira, J. A. 2024. PublicHearingBR: A Brazilian
  Portuguese Dataset of Public Hearing Transcripts for Summarization of Long Documents. arXiv:2410.07495
  (v2 de 22/08/2025). Local de publicação revisado por pares: a confirmar (só o arXiv foi confirmado).

### 11.2 Modelos usados

| modelo (id do Hugging Face) | revisão fixada | licença | referência |
|---|---|---|---|
| `PORTULAN/serafim-335m-portuguese-pt-sentence-encoder` | `a01887015444f7599669c509447c5bdbce958916` | MIT | Gomes, L.; Branco, A.; Silva, J.; Rodrigues, J.; Santos, R. 2024. Open Sentence Embeddings for Portuguese with the Serafim PT* Encoders Family. EPIA 2024, LNCS 14969, Springer, pp. 267-279. DOI 10.1007/978-3-031-73503-5_22. arXiv:2407.19527 |
| `PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir` | `222eaea5c8df6b61d8032ffffaa9bc9febd35a2e` | MIT | idem |
| `PORTULAN/serafim-900m-portuguese-pt-sentence-encoder` | `8b6aac1431bf56dd4b8f6a27a98ea78d3a7c5d31` | MIT | idem |
| `intfloat/multilingual-e5-large` | `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` | MIT | Wang, L.; Yang, N.; Huang, X.; Yang, L.; Majumder, R.; Wei, F. 2024. Multilingual E5 Text Embeddings: A Technical Report. arXiv:2402.05672 |
| `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` | MIT | Chen, J.; Xiao, S.; Zhang, P.; Luo, K.; Lian, D.; Liu, Z. 2024. M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. Findings of ACL 2024, pp. 2318-2335. DOI 10.18653/v1/2024.findings-acl.137 |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | `4328cf26390c98c5e3c738b4460a05b95f4911f5` | Apache-2.0 | Reimers e Gurevych 2019 e 2020 (11.4) |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` | Apache-2.0 | idem |
| `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | Apache-2.0 | sem artigo próprio; o cartão cita Chen et al. 2024 (M3-Embedding) e arXiv:2312.15503, hoje intitulado "Llama2Vec: Unsupervised Adaptation of Large Language Models for Dense Retrieval" (Liu, Z.; Li, C.; Xiao, S.; Shao, Y.; Lian, D.); versão ACL 2024 a confirmar |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `1427fd652930e4ba29e8149678df786c240d8825` | Apache-2.0 | sem artigo próprio; dados: Bonifacio, L. et al. 2021. mMARCO: A Multilingual Version of the MS MARCO Passage Ranking Dataset. arXiv:2108.13897 (local revisado por pares a confirmar); base: MiniLMv2 (11.4) |
| `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | `b5113eb38ab63efdd7f280f8c144ea8b13f978ce` | MIT | Laurer, M.; van Atteveldt, W.; Casas, A.; Welbers, K. 2024. Less Annotating, More Classifying: Addressing the Data Scarcity Issue of Supervised Machine Learning with Deep Transfer Learning and BERT-NLI. Political Analysis 32(1):84-100. DOI 10.1017/pan.2023.20 |
| `ruanchaves/mdeberta-v3-base-assin2-entailment` | `68c1e5976bc51b085869cfe98b69e32aaa178d42` | nenhuma declarada no Hugging Face (a confirmar) | Chaves Rodrigues, R.; Tanti, M.; Agerri, R. 2023. Evaluation of Portuguese Language Models (eplm), v1.0.0. Zenodo. DOI 10.5281/zenodo.7781848; artigo do modelo a confirmar |
| `convaiinnovations/laya` (pasta `multilingual` e raiz) | `aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8` | Apache-2.0 | sem artigo ou relatório técnico; citar como software: Convai Innovations, Laya, Hugging Face e PyPI (`laya`), com versão (0.3.11 usada aqui) e data de acesso. Artigo: a confirmar (nenhum encontrado). Segundo o cartão, a raiz usa ModernBERT-large (referência a confirmar) e a pasta `multilingual` usa mmBERT-base |
| `facebook/nllb-200-distilled-600M` | `f8d333a098d19b4fd9a8b18f94170487ad3f821d` | CC-BY-NC-4.0 | NLLB Team et al. 2022. No Language Left Behind: Scaling Human-Centered Machine Translation. arXiv:2207.04672; versão em periódico: NLLB Team et al. 2024. Scaling neural machine translation to 200 languages. Nature 630(8018):841-846. DOI 10.1038/s41586-024-07335-x |
| `facebook/m2m100_418M` | `55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636` | MIT | Fan, A. et al. 2021. Beyond English-Centric Multilingual Machine Translation. JMLR 22(107):1-48. arXiv:2010.11125 (o JMLR lista 16 autores; o arXiv, 17) |
| `lytang/MiniCheck-Flan-T5-Large` | `96eafd01cee2d16cf81aaa2fb226b14f422a37b3` | MIT | Tang, L.; Laban, P.; Durrett, G. 2024. MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents. EMNLP 2024, pp. 8818-8847. DOI 10.18653/v1/2024.emnlp-main.499 |
| `lytang/MiniCheck-DeBERTa-v3-Large` | `2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb` | MIT | idem |
| `yaxili96/FactCG-DeBERTa-v3-Large` | `0430e3509dbd28d2dff7a117c0eae25359ff3e80` | MIT | Lei, D. et al. 2025. FactCG: Enhancing Fact Checkers with Graph-Based Multi-Hop Data. NAACL 2025, pp. 5002-5020. DOI 10.18653/v1/2025.naacl-long.258 |
| `yzha/AlignScore` (base roberta-large `722cf37b1afa9454edce342e7895e588b6ff1d59`) | `8509e78d25bb914939fc585c626500c9b2944249` | MIT | Zha, Y.; Yang, Y.; Li, R.; Hu, Z. 2023. AlignScore: Evaluating Factual Consistency with a Unified Alignment Function. ACL 2023, pp. 11328-11348. DOI 10.18653/v1/2023.acl-long.634 |
| `vectara/hallucination_evaluation_model` (HHEM-2.1-Open; base flan-t5-base `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`) | `8e4a2e6e96c708cc76c2344f7e4757df2515292c` | Apache-2.0 | Li, M.; Luo, R.; Mendelevitch, O. 2024. HHEM-2.1-Open. Hugging Face. DOI 10.57967/hf/3240 |

### 11.3 Modelos declarados e não rodados

| modelo | revisão | referência |
|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` | `cdbee75f17c01a7cc42f958dc650907174af0554` | Qwen Team 2025. Qwen3 Technical Report. arXiv:2505.09388 (Apache-2.0) |
| `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | idem |
| `ibm-granite/granite-guardian-3.2-3b-a800m` | `3de033d89b499a18d9a573b5192bf3b967ef48c5` | Padhi, I. et al. 2024. Granite Guardian. arXiv:2412.07724 (Apache-2.0) |
| TypeSafe Jev `jev-1.13.0` | n/a | só documentação web proprietária (docs.typesafe.ai/models); artigo: a confirmar (nenhum encontrado) |
| Lynx, TRUE, LettuceDetect, Bespoke-MiniCheck-7B | n/a | citados em `CONFIDENCE_V2.md`; referências a confirmar |

### 11.4 Arquiteturas de base e bibliotecas

- Reimers, N.; Gurevych, I. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
  EMNLP-IJCNLP 2019, pp. 3980-3990. DOI 10.18653/v1/D19-1410. Referência também da biblioteca
  sentence-transformers.
- Reimers, N.; Gurevych, I. 2020. Making Monolingual Sentence Embeddings Multilingual using Knowledge
  Distillation. EMNLP 2020, pp. 4512-4525. DOI 10.18653/v1/2020.emnlp-main.365.
- Song, K.; Tan, X.; Qin, T.; Lu, J.; Liu, T.-Y. 2020. MPNet: Masked and Permuted Pre-training for Language
  Understanding. NeurIPS 2020. arXiv:2004.09297.
- Wang, W.; Wei, F.; Dong, L.; Bao, H.; Yang, N.; Zhou, M. 2020. MiniLM: Deep Self-Attention Distillation for
  Task-Agnostic Compression of Pre-Trained Transformers. NeurIPS 2020. arXiv:2002.10957.
- Wang, W.; Bao, H.; Huang, S.; Dong, L.; Wei, F. 2021. MiniLMv2: Multi-Head Self-Attention Relation
  Distillation for Compressing Pretrained Transformers. Findings of ACL-IJCNLP 2021, pp. 2140-2151. DOI
  10.18653/v1/2021.findings-acl.188.
- He, P.; Gao, J.; Chen, W. 2023. DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with
  Gradient-Disentangled Embedding Sharing. ICLR 2023. arXiv:2111.09543.
- He, P.; Liu, X.; Gao, J.; Chen, W. 2021. DeBERTa: Decoding-enhanced BERT with Disentangled Attention.
  arXiv:2006.03654; local ICLR 2021 a confirmar.
- Liu, Y. et al. 2019. RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.
- Chung, H. W. et al. 2024. Scaling Instruction-Finetuned Language Models. JMLR 25(70):1-53. arXiv:2210.11416.
- Marone, M.; Weller, O.; Fleshman, W.; Yang, E.; Lawrie, D.; Van Durme, B. 2025. mmBERT: A Modern
  Multilingual Encoder with Annealed Language Learning. arXiv:2509.06888.
- Wang, L. et al. 2022. Text Embeddings by Weakly-Supervised Contrastive Pre-training. arXiv:2212.03533 (E5
  original).
- Pedregosa, F. et al. 2011. Scikit-learn: Machine Learning in Python. JMLR 12(85):2825-2830.
- Sanchez, G. et al. 2023. Stay on topic with Classifier-Free Guidance. arXiv:2306.17806 (citado em
  `ACTOR_SIMULATION.md`; a confirmar).

### 11.5 Métodos e conjuntos de dados de NLI

- Robertson, S.; Zaragoza, H. 2009. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and
  Trends in Information Retrieval 3(4):333-389. DOI 10.1561/1500000019.
- Robertson, S. E.; Walker, S.; Jones, S.; Hancock-Beaulieu, M. M.; Gatford, M. 1995. Okapi at TREC-3. Proc.
  TREC-3, NIST SP 500-225, pp. 109-126.
- Salton, G.; Buckley, C. 1988. Term-weighting approaches in automatic text retrieval. Information Processing
  & Management 24(5):513-523. DOI 10.1016/0306-4573(88)90021-0.
- Cormack, G. V.; Clarke, C. L. A.; Buettcher, S. 2009. Reciprocal Rank Fusion Outperforms Condorcet and
  Individual Rank Learning Methods. SIGIR 2009, pp. 758-759. DOI 10.1145/1571941.1572114.
- Bowman, S. R.; Angeli, G.; Potts, C.; Manning, C. D. 2015. A large annotated corpus for learning natural
  language inference. EMNLP 2015, pp. 632-642. DOI 10.18653/v1/D15-1075.
- Williams, A.; Nangia, N.; Bowman, S. 2018. A Broad-Coverage Challenge Corpus for Sentence Understanding
  through Inference. NAACL-HLT 2018, pp. 1112-1122. DOI 10.18653/v1/N18-1101.
- Conneau, A.; Rinott, R.; Lample, G.; Williams, A.; Bowman, S.; Schwenk, H.; Stoyanov, V. 2018. XNLI:
  Evaluating Cross-lingual Sentence Representations. EMNLP 2018, pp. 2475-2485. DOI 10.18653/v1/D18-1269.
- Real, L.; Fonseca, E.; Gonçalo Oliveira, H. 2020. The ASSIN 2 Shared Task: A Quick Overview. PROPOR 2020,
  LNCS, Springer, pp. 406-412. DOI 10.1007/978-3-030-41505-1_39.
- Laban, P.; Schnabel, T.; Bennett, P. N.; Hearst, M. A. 2022. SummaC: Re-Visiting NLI-based Models for
  Inconsistency Detection in Summarization. TACL 10:163-177. DOI 10.1162/tacl_a_00453.

### 11.6 Estatística

- Wilson, E. B. 1927. Probable Inference, the Law of Succession, and Statistical Inference. JASA
  22(158):209-212. DOI 10.1080/01621459.1927.10502953.
- McNemar, Q. 1947. Note on the Sampling Error of the Difference Between Correlated Proportions or
  Percentages. Psychometrika 12(2):153-157. DOI 10.1007/BF02295996.
- Holm, S. 1979. A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of Statistics
  6(2):65-70. JSTOR 4615733.
- Efron, B. 1979. Bootstrap Methods: Another Look at the Jackknife. The Annals of Statistics 7(1):1-26. DOI
  10.1214/aos/1176344552.
- Efron, B.; Tibshirani, R. J. 1993. An Introduction to the Bootstrap. Chapman & Hall. DOI
  10.1007/978-1-4899-4541-9.
- Cohen, J. 1960. A Coefficient of Agreement for Nominal Scales. Educational and Psychological Measurement
  20(1):37-46. DOI 10.1177/001316446002000104.
- Youden, W. J. 1950. Index for rating diagnostic tests. Cancer 3(1):32-35.
  DOI 10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3.
- Hanley, J. A.; McNeil, B. J. 1982. The meaning and use of the area under a receiver operating
  characteristic (ROC) curve. Radiology 143(1):29-36. DOI 10.1148/radiology.143.1.7063747.
- Fawcett, T. 2006. An introduction to ROC analysis. Pattern Recognition Letters 27(8):861-874. DOI
  10.1016/j.patrec.2005.10.010.
- DeLong, E. R.; DeLong, D. M.; Clarke-Pearson, D. L. 1988. Comparing the Areas under Two or More Correlated
  Receiver Operating Characteristic Curves: A Nonparametric Approach. Biometrics 44(3):837-845. DOI
  10.2307/2531595.
- Referência para AURC e E-AURC (curva risco-cobertura): a confirmar (nenhuma foi verificada).

## Apêndice A. Inconsistências entre artefatos

A coluna "situação" diz o que foi feito na versão de release: erro de documentação corrigido no próprio
documento, ou divergência explicada.

| tema | fonte A | fonte B | situação |
|---|---|---|---|
| Validação humana intermediária, `semantic_match_high` | `artifacts/udv/udv_v2_precision_interim_20260928T143948Z.json`: 31 julgadas, 21 de 31, um rótulo inválido (A003) | recálculo desta data: 32 julgadas (21 `correta`, 4 `parcial`, 7 `incorreta`), 0 inválidos | vale o recálculo (a planilha mudou depois do arquivo anterior); `PIPELINE.md` deixou de copiar os números e aponta para a seção 7.2 |
| Kappa do "melhor juiz" na validação | `PIPELINE.md`: 0.69 | `nli_verifier_v2/evaluation_report.json`: `prompt_2_deepseek-chat` 0.7114 | os dois são verdadeiros: 0.6909 é o juiz de referência escolhido no treino (`prompt_1_gpt-4o-mini-2024-07-18`), 0.7114 é o máximo visto na validação; `PIPELINE.md` agora diz isso |
| Registros alterados de `udv_v1` para `udv_v2` | `udv_v2_analysis.json`: 1969 | `udv_v2_verify.json`, `baseline_diff.evidence_changed`: 1970 | explicado: `diff.evidence_changed` (1969) compara texto, `start_char`, `end_char` e `speaker_turn`; `diff.records_changed_any` (1970) soma `udv-6-0-6`, que só mudou de nível (cosseno 0.4838, entre os cortes 0.45 e 0.50); `baseline_diff.evidence_changed` (1970) compara o objeto `evidence` inteiro e soma `udv-117-0-0`, mesma sentença e mesmos offsets com cosseno diferente em 1.2e-7 (0.69387329 contra 0.69387317). As 1969 com evidência diferente estão nas três contagens |
| Nome da sigla UDV | `PIPELINE.md`: "unidades documentais verificáveis" | `UDV.md` e `README.md`: "Unidade Deliberativa Verificável" | corrigido em `PIPELINE.md` |
| Versões de biblioteca | `configs/nli_verifier.toml` e relatórios de recuperação: torch 2.13.0, sentence-transformers 5.6.1 | `udv_v2_coverage.json`: torch 2.14.0, sentence-transformers 6.1.0 | as duas; `udv_v2` é construída pela CLI de `../bookworm/`, que tem ambiente próprio |
| Medianas do top-1 na regra degenerada | `UDV.md`: "medianas 0,655 e 0,578" (errados e corretos) | `threshold_v1.json`: medianas 0.6472 e 0.5543; 0.655 e 0.5776 são as médias | corrigido em `UDV.md` |
| Afirmações antigas de `UDV.md` | resumo de recuperação cobre só rodadas lexicais; nenhuma linha julgada na amostra | `retrieval_v1_report.json` com todos os densos; planilha com 65 linhas julgadas | corrigido: `UDV.md` descreve só o registro e as regras de `udv_v1` e aponta para este relatório nos experimentos e na validação |
| "Nenhum avaliador da literatura supera o primário" | `PIPELINE.md` e a seção "Resposta" de `CONFIDENCE_V2.md` | `CONFIDENCE_V2.md`, E-A: comparação direta não declarada; só estimativa pontual | corrigido nos dois documentos: a ordem é estimativa pontual sem teste |
| Contagem de falantes recorrentes | `HEARING_ACTORS.md`: 282 falantes em 2 ou mais audiências (antes das fusões, 1901 falantes) | `actors.md`: 301 atores em 2 ou mais audiências (depois das fusões e regras de descarte) | as duas, em etapas diferentes |
| Versão do pacote Laya | `retrieval_v2_report.json`: 0.3.11 instalada | PyPI atual: 0.3.21 | 0.3.11 é a usada nos resultados e a fixada em `pyproject.toml` |
| Perfis e simulação | versão anterior deste relatório: sem rodada versionada | `mlx_alternative/runs/qwen38_27b/` | corrigido na seção 6.3; a rodada entrou com a união dos ramos |

## Apêndice B. O que foi planejado e não rodado

| item | situação | onde está declarado |
|---|---|---|
| Jev (`jev_en`, `jev_pt`) no E3 v2 | declarado, nunca rodou (`scorers_missing`) | `configs/nli_verifier.toml` |
| e3x_v1 na validação | selecionado, nunca confirmado | `e3x_v1/selection.json` |
| juízes LLM, Granite Guardian, Lynx, TRUE, SummaC, MiniCheck-RoBERTa, LettuceDetect | não rodados, com motivo | `CONFIDENCE_V2.md` |
| revisão das planilhas do E6 | sem julgamento | `fuzzy_v1_quotes_review.jsonl`, `fuzzy_v1_names_review.jsonl` |
| checagem manual de tradução | não preenchida | `artifacts/validation/translation_spot_check_v2/` |
| segunda metade da validação humana e reanotação de 20 linhas | pendentes | `annotation_guide.md` |
| validação humana própria de `udv_v2` | plano existe (`artifacts/udv/udv_v2_annotation_plan.json`); nenhuma planilha nova | ADR 0006 |
| conferência de perfis contra UDVs (`bookworm validate-profiles`) | implementada, sem rodada versionada; perfis e simulação têm a rodada da seção 6.3 | `../bookworm/docs/profile_validation.md` |
| auditoria editorial (índice de desvio editorial) e espaço latente multifacetado | descritos no documento de visão; sem artefato de resultado | `CONSTITUTION.md` |
| teste do benchmark de recuperação e do E5 | não lido por desenho | `retrieval_v1_report.json`, `confidence_v1_report.json` (`final_test` falso) |
