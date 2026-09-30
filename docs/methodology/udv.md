# Unidades Deliberativas Verificáveis (UDV)

Este documento descreve a UDV: o problema que ela resolve, o que cada registro contém, como os
registros são construídos a partir do PublicHearingBR e como o corte de similaridade de `udv_v1` foi
calibrado. A rodada descrita aqui é `udv_v1`. As mudanças de `udv_v2` (janelas de duas sentenças,
citação inteira, cortes recalibrados e confiança do verificador) estão em
[`pipeline.md`](../pipeline.md#pipeline-recomendado-udv_v2) e na
[ADR 0006](../../bookworm/docs/adr/0006-udv-v2-windows-full-quotes-and-verifier.md). Os resultados dos
experimentos e da validação humana estão no [relatório](../report.md). Os números saem dos artefatos
versionados em `experiments/artifacts/`; os caminhos desta página são relativos a `experiments/`.

## O problema

Cada uma das 206 audiências do arquivo LDS traz três textos: a transcrição oficial da sessão, a
matéria da Agência Câmara sobre ela e `metadados.envolvidos`, a lista das pessoas citadas na matéria,
cada uma com `nome`, `cargo` e uma lista de `opinioes`. São 1.065 participantes e 2.203 opiniões.
Segundo o artigo do dataset (seção 3), essas opiniões foram extraídas da matéria por um LLM e depois
corrigidas à mão, por comparação com a própria matéria. Elas registram o que a matéria atribui a cada
pessoa, com a redação da matéria.

A matéria tem em média 627 palavras e a transcrição, 18.102 (tabela 1 do artigo). Uma opinião
publicada resume ou cita uma fala que pode estar em qualquer ponto de horas de sessão, e nada no LDS
indica onde. O arquivo NLI do dataset não preenche essa lacuna: suas 4.238 opiniões foram geradas por
um LLM a partir da transcrição, no experimento de sumarização do artigo (seção 4), e não são as 2.203
do LDS. Cada uma vem com os quatro trechos de cinco sentenças que um recuperador por embedding achou na
fala da pessoa, e o rótulo do especialista diz se a opinião pode ser inferida desses quatro trechos,
não da transcrição inteira.

Sem um vínculo entre a opinião publicada e a transcrição, não há como mostrar a um leitor a passagem
de onde a opinião saiu, conferir se a fala a sustenta, nem estudar o que uma pessoa defendeu a partir
do que ela efetivamente disse. A UDV cria esse vínculo para cada uma das 2.203 opiniões.

## Objetivos

1. Ligar cada opinião do LDS a um trecho da fala da própria pessoa na transcrição, com posição exata
   (offsets de caracteres e índice do turno de fala), para que qualquer leitor possa conferir.
2. Declarar, em cada registro, que tipo de ligação foi encontrada e quanto ela merece confiança:
   citação literal localizada, similaridade semântica acima ou abaixo de um corte, pessoa sem fala
   localizada.
3. Registrar a proveniência de cada ligação (regra de casamento de texto ou modelo), sem apresentá-la
   como anotação humana.
4. Ser reproduzível: o mesmo LDS, a mesma configuração e o mesmo encoder geram o mesmo arquivo, e um
   verificador separado refaz as ligações e confere cada registro.

Outras partes do projeto já dependem das UDVs ou da mesma regra de ligação: o benchmark de citações
mascaradas (B1) usa a regra de citação literal para definir o alvo; a calibração do corte usa esse
benchmark e os pares de citação do train; [`hearing_actors.md`](hearing_actors.md) usa `udv_v1` para medir quantas opiniões
publicadas vêm de turnos de presidente de sessão (336 de 2.105); e a amostra de validação humana é
sorteada sobre `udv_v1`.

## O registro

Uma UDV é uma linha JSON por opinião do LDS. O identificador `udv-<audiência>-<participante>-<opinião>`
usa o `id` da audiência e as posições em `metadados.envolvidos` e em `opinioes`. Exemplo de
`artifacts/udv/udv_v1.jsonl`:

```json
{
  "id": "udv-1-3-4",
  "hearing_id": 1,
  "actor": {"name": "Arlindo Chinaglia", "role": "Deputado (PT-SP)"},
  "proposition": "\"A resposta do Musk foi atacar a presidente da comissão, que é eleita por todos os países da Comunidade Europeia.\"",
  "evidence": {
    "text": "A resposta do Musk foi atacar a presidente que foi eleita por todos os países da comunidade europeia, inclusive a democrática Hungria.",
    "support_type": "direct_quote",
    "score": null,
    "quote_prefix": "A resposta do Musk foi atacar",
    "start_char": 58193,
    "end_char": 58327,
    "speaker_turn": 27
  },
  "tier": "quote_found",
  "provenance": "weak",
  "method": {
    "encoder": "PORTULAN/serafim-335m-portuguese-pt-sentence-encoder",
    "revision": "a01887015444f7599669c509447c5bdbce958916",
    "embedding_threshold": 0.45
  }
}
```

| campo | conteúdo |
|---|---|
| `id`, `hearing_id` | identificador da UDV e `id` da audiência no LDS |
| `actor` | `name` e `role`, copiados de `nome` e `cargo` |
| `proposition` | o texto da opinião no LDS, sem alteração |
| `evidence` | o trecho escolhido, ou `null` quando não há fala utilizável da pessoa |
| `evidence.text` | a sentença da transcrição, com espaços normalizados |
| `evidence.support_type` | `direct_quote`, `semantic_with_short_quote` ou `semantic_similarity` |
| `evidence.score` | cosseno entre opinião e sentença no encoder; `null` em `direct_quote` |
| `evidence.quote_prefix` | o começo da citação da opinião que foi achado na fala, quando houver |
| `evidence.start_char`, `evidence.end_char` | posição do trecho em `transcricao` (início inclusivo, fim exclusivo) |
| `evidence.speaker_turn` | índice do turno de fala na numeração de `split_into_turns` |
| `tier` | nível de confiança (tabela abaixo) |
| `provenance` | `weak` (regra de casamento de texto), `model` (encoder) ou `null` (sem evidência) |
| `method` | nome e revisão do encoder e corte de similaridade da rodada |

| `tier` | condição | registros em `udv_v1` |
|---|---|---|
| `quote_found` | uma citação da opinião tem prefixo de 6 ou 10 palavras achado literalmente num turno da pessoa | 277 |
| `semantic_match_high` | sem citação confiável; a sentença da pessoa mais próxima da opinião no encoder tem cosseno ≥ 0,45 | 1.785 |
| `semantic_match_weak` | idem, com cosseno < 0,45 | 43 |
| `no_evidence` | a pessoa tem turnos, mas nenhuma sentença candidata | 8 |
| `person_not_resolved` | nenhum turno da transcrição foi atribuído à pessoa | 90 |

As 2.105 evidências se dividem, por tipo de suporte, em 277 `direct_quote`, 111
`semantic_with_short_quote` (110 em alta confiança, 1 em baixa) e 1.717 `semantic_similarity`. Todas
as 2.105 têm offsets localizados na transcrição. Em `semantic_match_high` o cosseno vai de 0,455 a
0,989 (mediana 0,687); em `semantic_match_weak`, de 0,275 a 0,449.

O que cada nível afirma e o que não afirma:

- `quote_found`: o prefixo de 6 ou 10 palavras da citação aparece literalmente num turno da pessoa,
  sem distinção de maiúsculas, e a evidência é a sentença que o contém. O resto da citação não é
  conferido. No exemplo acima, a matéria escreve "presidente da comissão, que é eleita" e a
  transcrição, "presidente que foi eleita": o prefixo casa e a diferença depois dele fica registrada
  no par `proposition`/`evidence.text`. Dos 277 prefixos, 152 têm 6 palavras, 3 têm 7 (citação de 7
  palavras) e 122 têm 10.
- `semantic_match_high` e `semantic_match_weak`: a evidência é a sentença da pessoa com maior cosseno
  em relação à opinião, e o nível diz de que lado do corte esse cosseno ficou. Nenhum dos dois níveis
  afirma que a sentença sustenta a opinião; a seção "Qualidade da evidência" reúne o que foi medido
  sobre isso.
- `semantic_with_short_quote`: além da escolha do encoder, um prefixo curto (menos de 6 palavras) de
  uma citação da opinião cai dentro da sentença escolhida. A corroboração é fraca: dos 111 prefixos
  gravados, 1 tem 5 palavras, 65 têm 4, 37 têm 3, 7 têm 2 e 1 tem uma palavra só.
- `no_evidence` e `person_not_resolved`: não foi achada fala utilizável da pessoa. Isso não indica erro
  da matéria. Os 8 registros `no_evidence` são de 3 participantes cuja única fala registrada é a
  marcação "(Manifestação em LIBRAS.)"; entre as 45 pessoas não resolvidas há grafias diferentes do
  nome, instituições citadas como participantes e pessoas que não falaram.

Nenhum registro tem proveniência `human`. `weak` vem do sentido de supervisão fraca: rótulo produzido
por regra, sem conferência humana.

O desenho inicial da UDV previa também ato de fala, posição, alvo, argumentos, relações, relevância e
incerteza por registro. Nenhum desses campos foi implementado: o registro atual contém só a ligação
entre opinião e evidência.

## Como reproduzir

Dentro de `experiments/`:

```bash
uv run bookworm build-udvs --run-name udv_v1
uv run bookworm verify-udvs --run-name udv_v1
```

`build_udvs` lê `configs/udv.toml` (caminho e SHA-256 do LDS, nome e revisão do encoder, corte de
similaridade com a origem da calibração, caminhos) e grava `artifacts/udv/udv_v1.jsonl` e
`artifacts/udv/udv_v1_coverage.json`. O relatório de cobertura guarda as contagens por nível e por tipo
de suporte, os offsets localizados, a descrição da pipeline (padrão de fronteira de sentença, padrões
de citação, regras de escolha da citação e da ocorrência), o dispositivo, as versões de Python, torch e
sentence-transformers, o tempo de execução e a configuração inteira. `--ids` e `--limit` restringem a
rodada a algumas audiências. Os embeddings de sentenças e opiniões ficam em cache em
`artifacts/cache/embeddings/` (não versionado), chaveados por encoder, revisão, dispositivo e lista de
textos. Com o cache pronto, a rodada `udv_v1` levou 23,5 s em MPS (Apple). A última medição sem cache
registrada para as 206 audiências é a da primeira rodada completa, em 18/09/2026, com o código daquela
data: 1.419 s em MPS.

`verify_udvs` refaz, a partir do LDS e com as mesmas funções, a resolução de pessoa, as sentenças
candidatas e a busca de citação de cada opinião, e confere: ids completos e sem repetição; opinião e
ator iguais aos do LDS; texto da evidência presente dentro de um único turno do ator; sentença
semântica pertencente à lista de candidatas da pessoa; citação, prefixo e sentença de `quote_found`
iguais aos recalculados; nenhuma opinião semântica com citação confiável disponível; tipo de suporte e
prefixo curto coerentes com a busca de citação; nível coerente com o score e o corte; texto recuperado
pelos offsets igual ao gravado e turno dos offsets igual ao turno de origem da sentença; contadores do
relatório. Ele não recalcula a escolha do encoder nem avalia se a evidência sustenta a opinião: confere
a coerência do arquivo com as regras. Em 24/09/2026, `udv_v1` passa sem nenhuma acusação. Com
`--baseline <rodada anterior>.jsonl`, o verificador também lista as mudanças de nível e conta os
registros com evidência diferente em relação a outra rodada.

A reprodutibilidade foi testada regenerando uma rodada antiga: `artifacts/udv/udv_v0_repro.jsonl`,
gerado de novo com o código que produziu `udv_v0` (`backup/*_2026-09-21.py`, na tag Git
`research-2026-09-28`), é idêntico byte a byte a `udv_v0.jsonl`.

Os comandos que geram os artefatos relacionados (splits, benchmarks B1 e B2, calibração, medições das
ADRs e amostra de validação) estão no [guia de reprodução](../reproduce.md#etapas-em-ordem).

## Construção

O código está na biblioteca, nos módulos `bookworm.transcript` e `bookworm.udv` (a rodada em lote é
`bookworm build-udvs`). Esta seção é a descrição das regras; a biblioteca não as repete. Para cada
audiência, as etapas são estas.

### 1. Turnos de fala

A transcrição marca cada troca de falante com um cabeçalho (`O SR. JORGE SOLLA (Bloco/PT - BA) -`,
`A SRA. PRESIDENTE (Erika Kokay. PT - DF) -`). `split_into_turns` encontra os cabeçalhos por expressão
regular e define como turno o texto até o cabeçalho seguinte, com posição inicial e final na
transcrição: 17.264 turnos nas 206 audiências. O turno resolve a atribuição de autoria sem depender da
matéria: qualquer trecho escolhido dentro de um turno tem autor conhecido.

### 2. Resolução da pessoa

Cada participante de `metadados.envolvidos` precisa ser associado aos seus turnos. O nome no LDS é o da
matéria, que costuma diferir do cabeçalho ("Rodrigo Marinho" na matéria, "RODRIGO SARAIVA MARINHO" na
transcrição), e quem preside a sessão aparece como `PRESIDENTE` com o nome entre parênteses.

- Cada turno gera candidatos de nome: o nome do cabeçalho; nos cabeçalhos de presidência, o conteúdo
  do parêntese antes do último ". " quando o que vem depois é partido e UF; e o parêntese inteiro
  quando ele não é partido (caso do nome social).
- Um participante casa com um candidato quando os conjuntos de palavras (sem acento, em maiúsculas)
  são iguais ou um contém o outro, se ambos têm duas palavras ou mais. Um nome de uma palavra só casa
  com um candidato idêntico ou, se não houver nenhum, com o único falante da audiência que tenha
  aquela palavra no nome; se houver mais de um, a pessoa fica sem turno.

A regra de nome contido é aceitável dentro de uma audiência, que tem poucos falantes; entre audiências
ela junta pessoas diferentes, e por isso [`hearing_actors.md`](hearing_actors.md) usa outra regra para identificar a mesma
pessoa em audiências distintas. Resultado: 1.020 dos 1.065 participantes têm ao menos um turno; as 45
pessoas restantes somam 90 opiniões.

### 3. Sentenças candidatas

A fala de cada pessoa é dividida em sentenças dentro de cada turno, separadamente, e cada sentença
guarda o índice do turno de onde veio (ADR 0002). A fronteira é um `.`, `!` ou `?` seguido de espaço,
ou um parêntese fechado cujo conteúdo termina nessa pontuação, como `(Palmas.)` e `(Pausa.)`, exceto a
elisão `(...)`. Ficam as partes com 4 palavras ou mais que não são só marcação de palco. São 115.599
sentenças candidatas.

A divisão por turno corrigiu um defeito da versão anterior, que concatenava todos os turnos da pessoa:
350 sentenças, de 244 pessoas, atravessavam a fronteira entre dois turnos e colavam o fim de uma fala
ao começo de outra, com falas de terceiros no meio. Uma evidência gravada em `udv_v0` (`udv-1-1-2`)
era um texto desse tipo, que não existe em lugar nenhum da transcrição.

### 4. Citação literal

Muitas opiniões do LDS trazem trechos entre aspas, que a matéria apresenta como fala literal. A
citação permite uma ligação que não depende de modelo.

- Citações são trechos de 10 caracteres ou mais entre `"…"`, `“…”` ou `'…'`; as aspas simples exigem
  fronteira de palavra, para que um apóstrofo como em `d'água` não abra citação (ADR 0003).
- De cada citação saem prefixos de 10, 6, 4 e 3 palavras, tentados nessa ordem. O prefixo é procurado
  dentro de cada turno da pessoa, sem distinção de maiúsculas e sem casar no meio de uma palavra.
- Entre todas as citações da opinião vence a de prefixo mais longo; no empate, a que aparece primeiro.
- Prefixo de 6 palavras ou mais gera `quote_found`, e a evidência é a sentença que contém o prefixo.
  Se esse prefixo ocorre mais de uma vez na fala, vence a ocorrência cuja sentença tem maior índice de
  Jaccard de palavras com a opinião. Em `udv-65-1-0`, o prefixo "A Justiça do Trabalho é fundamental"
  aparece nos turnos 6 e 18, e a opinião ("em um país tão desigual") corresponde ao turno 18.

O funil nas 2.203 opiniões: 2.113 são de pessoas resolvidas, 960 têm ao menos uma citação extraída, 549
têm algum prefixo achado na fala e 277 têm prefixo de 6 palavras ou mais
(`artifacts/benchmarks/masked_quotes_v1_report.json`, seção `splits.all.opinions`).

O corte de 6 palavras existe porque prefixos de 3 ou 4 palavras costumam ser expressões comuns, que
casam em pontos da fala sem relação com a citação. A escolha veio de uma leitura exploratória dos
casamentos agrupados por tamanho de prefixo, sem protocolo de anotação. A precisão de `quote_found`
ainda não foi medida por anotação humana; ela é um dos estratos da amostra cega descrita adiante.

### 5. Similaridade semântica

Quando não há citação confiável, a opinião e todas as sentenças candidatas da pessoa são codificadas
pelo Serafim 335m (`PORTULAN/serafim-335m-portuguese-pt-sentence-encoder`, variante para português
brasileiro, revisão fixada em `configs/udv.toml`, vetores de 1.024 dimensões, entrada limitada a 128
tokens), e a evidência é a sentença de maior cosseno. O protótipo começou com TF-IDF, que compara
vocabulário compartilhado e falha quando a matéria parafraseia a fala; o encoder denso foi adotado
para comparar significado.

Se um prefixo curto de uma citação da opinião cai na sentença escolhida pelo encoder (ou numa sentença
que a contém), o registro recebe `support_type = semantic_with_short_quote` e grava o prefixo; o nível
continua decidido pelo score.

### 6. Nível e proveniência

- Nenhum turno da pessoa: `person_not_resolved`, sem evidência.
- Turnos, mas nenhuma sentença candidata: `no_evidence`, sem evidência.
- Citação confiável: `quote_found`, proveniência `weak`.
- Caso contrário: `semantic_match_high` se o score for ≥ 0,45, senão `semantic_match_weak`, ambos com
  proveniência `model`.

### 7. Offsets

A sentença escolhida é localizada na transcrição original por uma expressão regular com as palavras
da sentença separadas por qualquer espaço, restrita ao turno de origem. As 2.105 evidências de
`udv_v1` são localizadas. A busca devolve a primeira ocorrência do texto no turno: em 85 das 115.599
sentenças candidatas ela aponta uma posição anterior à da sentença, em 76 delas porque a mesma frase se
repete no turno (mesmo texto, outro ponto) e em 9 porque o texto aparece dentro ou no começo de outra
sentença (`artifacts/udv/turn_segmentation_summary.json`, seção `offset_search`). Nas 263 sentenças
de citação confiável medidas na ADR 0002 isso não ocorre.

## Calibração do corte de similaridade

O corte separa `semantic_match_high` de `semantic_match_weak`. Ele não pode ser escolhido olhando
audiências de validação ou teste, porque a amostra de validação humana é sorteada no teste e depende
dessa fronteira.

Os cortes anteriores (0,25 com TF-IDF; 0,46, 0,44 e 0,47 com o encoder) foram calibrados nas 20
primeiras audiências, que incluem audiências dos conjuntos de validação e de teste do split temporal.
O corte atual, 0,45, foi recalculado só no train (144 audiências) por
`uv run python -m experiments.udv.calibrate_threshold`, com saída em `artifacts/calibration/threshold_v1.json`.

A regra primária, declarada em 22/09/2026 antes de qualquer resultado, calibrava diretamente a decisão
que o corte toma: no benchmark de citações mascaradas (B1, abaixo), o top-1 do encoder está correto
quando coincide com a sentença da citação, e o corte maximizaria o índice J de Youden de "score ≥ t
prevê top-1 correto". A regra se mostrou degenerada: só 12 das 142 consultas do train têm top-1
correto, os top-1 errados têm score mais alto que os corretos (medianas 0,6472 e 0,5543; médias 0,655
e 0,5776, `artifacts/calibration/threshold_v1.json`), o melhor J é 0
e o corte correspondente aceita todas as consultas. Ela foi registrada como não adotada.

O corte adotado vem da regra usada desde o protótipo, recalculada só no train:

| regra | positivos | negativos | corte | positivos abaixo | negativos acima | IC 95% (bootstrap por audiência) |
|---|---|---|---|---|---|---|
| `legacy_random` (adotada) | opinião contra a sentença da própria citação confiável (183 pares, 106 audiências) | a mesma opinião contra uma sentença sorteada de outra audiência do train | 0,4543 | 1 | 0 | 0,4366 a 0,4687 |
| `hard_negative` (comparação) | idem | a mesma opinião contra outra sentença da própria pessoa | 0,5019 | 7 | 21 | 0,5004 a 0,5416 |

Nas duas regras o corte é o ponto médio entre o quartil 25 dos positivos (0,628) e o quartil 75 dos
negativos (0,281 contra sentenças de outra audiência, 0,376 contra sentenças da própria pessoa). O
corte adotado separa a sentença da própria citação de sentenças de outras audiências; contra outras
sentenças da mesma pessoa, que é a escolha que o encoder faz de fato, a mesma regra daria 0,50. O
corte define a fronteira entre os dois níveis semânticos e não foi validado como medida de confiança,
como registrado em `threshold_decision` de `configs/udv.toml`.

A troca de 0,47 para 0,45 (`udv_v1_pre` para `udv_v1`) move 9 registros de `semantic_match_weak` para
`semantic_match_high` e não muda nenhuma evidência
(`uv run bookworm verify-udvs --run-name udv_v1 --baseline artifacts/udv/udv_v1_pre.jsonl`).

## Versões

O protótipo foi construído nos notebooks `udv_v00` a `udv_v05` sobre as 20 primeiras audiências, na
seguinte ordem: evidência só por citação literal; similaridade TF-IDF; troca para o encoder denso;
correção do resolvedor de nomes para quatro formas de cabeçalho quase ausentes da amostra de 20
(presidente com abreviatura no parêntese, nome social entre parênteses, marcador de tradução
simultânea lido como nome, nomes de uma palavra); e a política de citação sem distinção de maiúsculas
com prefixo mínimo de 6 palavras. `experiments/notebooks/udv.ipynb` foi executado com o código anterior às ADRs 0002 e
0003; reexecutá-lo com o código atual pode dar números diferentes, o que não foi medido.

As rodadas completas versionadas em `artifacts/udv/`:

| rodada | `created_at` (UTC) | mudança | `quote_found` | `semantic_match_high` | `semantic_match_weak` | `no_evidence` | `person_not_resolved` |
|---|---|---|---|---|---|---|---|
| `udv_v0` | 21/09/2026 | política de citação de 6+ palavras, corte 0,47 | 260 | 1.793 | 52 | 8 | 90 |
| `udv_v1_pre` | 23/09/2026 | sentenças por turno e escolha de ocorrência (ADR 0002), aspas simples (ADR 0003) | 277 | 1.776 | 52 | 8 | 90 |
| `udv_v1` | 23/09/2026 | corte 0,45 calibrado só no train | 277 | 1.785 | 43 | 8 | 90 |

De `udv_v0` para `udv_v1_pre`, 17 opiniões passam de `semantic_match_high` para `quote_found` (14 pelas
aspas simples, 3 porque uma citação posterior da opinião casa com prefixo mais longo) e 42 registros
têm alguma diferença na evidência: texto, posição, tipo de suporte, prefixo gravado ou score
(`uv run bookworm verify-udvs --run-name udv_v1_pre --baseline artifacts/udv/udv_v0.jsonl`). As
causas estão nas ADRs 0002 e 0003. `udv_v0` não passa mais em `verify_udvs`, que confere a lógica
atual; as acusações que ele recebe estão listadas na ADR 0002.

Também ficam em `artifacts/udv/` arquivos gerados com código anterior: `dev20.jsonl` (rodada nas 20
primeiras audiências, 21/09/2026) e os `case_insensitive_*` do experimento que levou à política de
citação sem distinção de maiúsculas (20/09/2026). Eles servem de histórico e não refletem a pipeline
atual.

## Distribuição por split

Com o split temporal `temporal_v1` (datas do carimbo de publicação da matéria), as UDVs se distribuem
assim. Os estratos são os da amostra de validação humana
(`artifacts/validation/human_validation_v1_udv_v1/sample_report.json`, seção `population`):
`semantic_match_high` e `semantic_match_weak` contam só `semantic_similarity`, os registros
`semantic_with_short_quote` formam um estrato próprio e `speaker_check` reúne `no_evidence` e
`person_not_resolved`.

| split | audiências | UDVs | `direct_quote` | `semantic_match_high` | `semantic_with_short_quote` | `semantic_match_weak` | `speaker_check` |
|---|---|---|---|---|---|---|---|
| train | 144 | 1.536 | 183 | 1.170 | 70 | 31 | 82 |
| validation | 32 | 308 | 53 | 228 | 19 | 5 | 3 |
| test | 30 | 359 | 41 | 277 | 22 | 6 | 13 |

Os níveis sem evidência se concentram no train (82 de 98).

## Qualidade da evidência

Os números desta seção, exceto os da amostra de validação humana, foram medidos em audiências de train
e validação; o teste fica reservado a essa amostra.

### Benchmarks usados

- **B1, citações mascaradas** (`artifacts/benchmarks/masked_quotes_v1.jsonl`): opiniões com citação
  confiável, com todos os trechos entre aspas removidos. A tarefa é achar, entre todas as sentenças
  candidatas da pessoa, a sentença que contém a citação. A opinião mascarada faz o papel das opiniões
  sem citação, que são as que recebem evidência semântica. Das 277 opiniões com citação confiável, 210
  ficam no benchmark (142 train, 40 validação, 28 teste); as outras ficam com menos de 4 palavras depois
  da máscara. No train há em média 128,6 sentenças candidatas por consulta, e um sorteio acertaria o
  top-1 em 1,5% das vezes. O alvo é um rótulo prata: a sentença da citação, achada por casamento de
  texto e não conferida por humano. A parte da opinião fora das aspas pode descrever outra passagem da
  fala, e nesse caso uma sentença diferente do alvo também pode sustentar a opinião; por isso B1 mede a
  capacidade de achar a passagem citada e subestima a precisão da evidência quando outra sentença
  também sustenta a opinião.
- **B2, NLI** (`artifacts/benchmarks/nli_v1.jsonl`): as 4.238 opiniões do arquivo NLI, com os quatro
  trechos localizados na transcrição do LDS (todos os 11.920 trechos do train localizados) e o rótulo
  do especialista. Na recuperação, uma sentença é relevante quando se sobrepõe a qualquer um dos quatro
  trechos de uma opinião rotulada como inferível. É um critério permissivo: os quatro trechos de cinco
  sentenças cobrem boa parte da fala de muitas pessoas (um sorteio acerta o top-1 em 28% das consultas
  do train e 30% das da validação), e eles foram escolhidos por similaridade de embedding com a própria
  opinião.

### O que os números dizem sobre a camada semântica

1. **O encoder raramente acha a sentença da citação quando ela está mascarada.** No B1, o top-1 do
   Serafim coincide com o alvo em 12 de 142 consultas do train (8,5%) e 4 de 40 da validação (10%)
   (`artifacts/experiments/confidence/confidence_v1/confidence_v1_report.json`). Os seis
   recuperadores léxicos (TF-IDF de palavras e de caracteres, BM25) ficam na mesma faixa por sentença
   (acc@1 entre 5,5% e 8,2% em train mais validação); quando a unidade é o turno inteiro, põem o turno
   da citação em primeiro em 69% a 86% das consultas, contra 46% de um sorteio de turno
   (`artifacts/experiments/retrieval/retrieval_v1/retrieval_v1_report.json`). A busca costuma chegar
   ao turno da citação e raramente à sentença exata.
2. **Mesmo com a citação no texto, o encoder erra a sentença com frequência.** Nas 183 opiniões do
   train com citação confiável e sem máscara, o top-1 do encoder é a sentença da citação em 112 (61,2%,
   IC 95% 53,0% a 68,8%) (`artifacts/experiments/fuzzy/fuzzy_v1_report.json`, controle positivo).
3. **O score não indica acerto.** No B1, os top-1 errados têm score mais alto que os corretos, o que
   tornou degenerada a regra primária de calibração. No experimento de políticas de confiança, nenhum
   sinal (score do top-1, margem para o segundo colocado, z-score entre as candidatas, probabilidade de
   entailment de um modelo NLI, regressão logística sobre esses sinais) prevê o acerto do top-1 melhor
   que o próprio score, na validação, com intervalo pareado que exclua zero, nem no B1 nem no B2.
4. **No critério permissivo do B2, o top-1 cai num trecho do especialista na maioria das vezes.** A
   sentença de maior cosseno do Serafim se sobrepõe a um dos quatro trechos em 544 de 620 consultas da
   validação (87,7%), contra 30% de um sorteio. Pelos motivos descritos acima, esse número não mede se
   a sentença sustenta a opinião.

Em conjunto: `semantic_match_high` garante que a melhor sentença candidata tem cosseno ≥ 0,45 com a
opinião. Os resultados por turno (recuperadores léxicos no B1) e por trecho (Serafim no B2) indicam
que a região certa da fala costuma ser encontrada, mas a sentença exata escolhida pode não ser a que
sustenta a opinião. Quanto disso é erro e quanto é outra sentença válida só a anotação humana mede.

Uma revisão exploratória anterior, de 65 pares das 20 primeiras audiências, foi feita sem protocolo
cego e sem anotador independente; ela não é validação e foi retirada desta versão (fica na tag
`research-2026-09-28`).

### Validação humana cega

A amostra `artifacts/validation/human_validation_v1_udv_v1/` foi sorteada de `udv_v1`, só nas
audiências de teste, porque a validação é usada pelos experimentos para escolher método e uma estimativa
tirada dela seria enviesada a favor do método escolhido. São 127 linhas em cinco estratos
(`direct_quote`, `semantic_match_high`, `semantic_with_short_quote`, `semantic_match_weak` e
`speaker_check`), com semente fixa, rótulos `correta`, `parcial` ou `incorreta` e critérios declarados
em 23/09/2026, antes de qualquer julgamento. O desenho completo da amostra, o cegamento e os vazamentos
conhecidos estão no [relatório, seção 7.1](../report.md#71-desenho-da-amostra); o critério de cada rótulo,
no [guia do anotador](../validation/annotation_guide.md); os resultados, nas seções 7.2 e 7.4 do
relatório.

## Experimentos que testam alternativas

Os experimentos que usam B1 e B2 para testar componentes que poderiam substituir ou complementar os da
UDV estão no relatório, com os números e as fontes: recuperadores e unidades (E1, E2 e retrieval_v2,
seções 4.3 a 4.5), verificador de suporte e tradução (E3, E3x, seções 5.1 a 5.4), sinais de confiança
(E5 e confidence_v2, seções 5.5 e 5.6) e casamento aproximado de citações e nomes (E6, seção 5.8). O
que cada um decidiu para a construção está em [`docs/pipeline.md`](../pipeline.md).

## Relação com os outros documentos

[`hearing_actors.md`](hearing_actors.md) separa a fala completa de cada pessoa entre audiências usando o mesmo
`split_into_turns`, e seus arquivos usam os mesmos nomes de campo (`actor`, `hearing_id`,
`start_char`), de modo que uma UDV pode ser cruzada com o turno correspondente na fala do ator. As
duas bases resolvem identidade de formas diferentes: a UDV associa o nome da matéria aos cabeçalhos de
uma audiência pela regra de nome contido; [`hearing_actors.md`](hearing_actors.md) associa cabeçalhos entre audiências pelo
nome exato mais uma lista revisada de mesclas. [`actor_profiles.md`](actor_profiles.md) gera perfis a partir da fala
completa, sem usar as UDVs.

## Limitações

- Na validação humana de `udv_v1` (relatório, seção 7.2; `artifacts/udv/udv_v2_precision_final.json`),
  os dois critérios declarados falham: `direct_quote` tem precisão estrita 0,7143 [0,5495; 0,8367]
  (25 de 35) e `semantic_match_high` tem precisão tolerante 0,7385 [0,6205; 0,8298] (48 de 65). Nos
  estratos `semantic_match_weak` (6 UDVs) e `speaker_check` (6 pessoas), o teste tem tão poucos casos
  que os intervalos são largos.
- A camada semântica, com 1.828 dos 2.105 registros com evidência (1.785 deles em alta confiança), é a
  menos sustentada: o encoder acha a sentença citada em 8,5% a 10% das consultas mascaradas e em 61%
  das não mascaradas, e o score não distingue acerto de erro. O corte de 0,45 depende do que se usa como negativo (0,50 com sentenças da
  própria pessoa).
- Cada opinião recebe uma única sentença. Opiniões que juntam vários pontos (o artigo do dataset
  registra que afirmação e citação ilustrativa foram fundidas numa opinião só) podem não caber numa
  sentença, e tendem a ser julgadas `parcial` quando a sentença cobre só um dos pontos.
- O encoder corta a entrada em 128 tokens; quantas opiniões e sentenças passam desse limite não foi
  medido.
- `quote_found` garante o prefixo de 6 ou 10 palavras, não a citação inteira. A regra de preferir a
  citação de prefixo mais longo troca a evidência mesmo quando a primeira citação já era confiável e
  tinha sobreposição maior com a opinião (`udv-24-1-0`, único caso). A corroboração por prefixo curto
  usa a primeira ocorrência do prefixo e aceita prefixos pouco distintivos.
- A resolução de pessoa depende do nome da matéria estar contido no cabeçalho (ou vice-versa). Uma
  letra diferente entre a grafia da matéria e a do cabeçalho basta para deixar a pessoa sem turno; dois
  falantes da mesma audiência cujos nomes se contenham poderiam ser confundidos, e isso não foi medido
  na rodada atual.
- A atribuição de autoria herda os erros da segmentação em turnos: fala sem cabeçalho próprio, como a
  marcada `(Não identificado)-`, fica dentro do turno da pessoa anterior.
- Os 8 registros `no_evidence` são de participantes que falaram em Libras; a tradução está nos turnos
  do intérprete, que a UDV não associa a eles.
- A busca de offsets devolve a primeira ocorrência do texto no turno; em 85 das 115.599 sentenças
  candidatas isso aponta outra posição (9 delas com texto que não coincide com uma sentença inteira).
- `verify_udvs` confere a coerência do arquivo com as regras e não recalcula a escolha do encoder.
- `no_evidence` e `person_not_resolved` registram que a fala não foi achada e não servem de indício de
  que a matéria atribuiu à pessoa algo que ela não disse. Uma evidência semântica fraca também não
  indica erro da matéria.
