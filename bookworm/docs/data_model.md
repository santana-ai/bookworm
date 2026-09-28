# Modelo de dados

Este documento descreve, campo por campo, os arquivos que a biblioteca lê e grava: o arquivo LDS do
PublicHearingBR, o registro UDV, o arquivo de cobertura de uma execução de UDV, o JSON de
demonstração de uma audiência, o diretório de demonstração de uma execução, os arquivos de falas por
ator e de ligações UDV → ator, o manifesto de split e o relatório de split. Os scripts de origem
trocam dicionários sem schema publicado, então o significado de um campo, os valores possíveis e os
casos em que ele fica nulo só podiam ser descobertos lendo o código. Os números citados vêm dos
arquivos reais (`PublicHearingBR_LDS.jsonl` com o sha256
`c4e392ab95ce22f6228eace16f15e9efe1c846724846b873dec420aec3d872c0`, `udv_v1.jsonl`,
`udv_v1_pre.jsonl` e `temporal_v1*.json`) e podem ser recalculados com o trecho da seção
[Como conferir](#como-conferir) e com os testes do marcador `dataset`.

Convenções usadas abaixo:

- Todos os arquivos são UTF-8. JSONL tem um objeto JSON por linha; os JSON são gravados com
  `json.dump(..., ensure_ascii=False, indent=2)`.
- A ordem das chaves é parte do formato: a biblioteca grava na ordem listada aqui. Os testes de
  paridade comparam byte a byte os JSONL de UDV (reconstruídos a partir do cache de embeddings, ou
  lidos e regravados sem ele), o manifesto de split (depois de igualar `created_at`) e o relatório de
  split (depois de igualar `created_at` e `environment`). Nos arquivos de cobertura, comparam as
  seções recontadas (`hearings`, `people`, `opinions`, `evidence_offsets`, `evidence_support_types`) e
  `pipeline`, porque `created_at`, `timing`, `encoder_runtime` e `environment` dependem da execução e
  da máquina.
- Datas são `AAAA-MM-DD` (ISO 8601); instantes de criação são UTC com segundos
  (`2026-09-21T17:25:23+00:00`).
- Em JSON, chaves de objeto são sempre texto. Contadores indexados por inteiro (`by_year`,
  `lag_days_when_agreeing`, `article_dates`) aparecem com a chave como string no arquivo.

## Arquivo LDS (`PublicHearingBR_LDS.jsonl`)

Um registro por audiência pública, 206 no total, com as chaves nesta ordem: `id`, `materia`,
`metadados`, `transcricao`. A biblioteca lê cada linha como `HearingRecord` (pydantic, `extra="forbid"`,
`strict=True`, `frozen=True`), então uma chave nova, uma chave ausente ou um tipo diferente fazem a
leitura falhar em vez de passar adiante. As chaves ficam em português, como no dataset.

| Campo | Tipo | Conteúdo | Observado no arquivo |
|---|---|---|---|
| `id` | inteiro | Identificador da audiência. | 1 a 206, sem repetição, em ordem crescente. |
| `materia` | texto | Matéria jornalística sobre a audiência, com título e carimbo de publicação `DD/MM/AAAA - HH:MM`. | 288 a 1.215 palavras (mediana 607, contando por espaço). Carimbo de publicação nas 206; `Atualizado em DD/MM/AAAA - HH:MM` em 13. |
| `metadados.assunto` | texto | Tema da audiência. | Nunca vazio nem só com espaços. |
| `metadados.envolvidos` | lista | Participantes citados na matéria, na ordem do dataset. | 2 a 12 por audiência, 1.065 no total. |
| `envolvidos[].nome` | texto | Nome do participante como aparece na matéria. | 879 nomes distintos depois de `strip()` e maiúsculas. |
| `envolvidos[].cargo` | texto | Cargo ou vínculo do participante. | Nunca vazio nem só com espaços. |
| `envolvidos[].opinioes` | lista de texto | Opiniões atribuídas ao participante. | 1 a 15 por pessoa, 2.203 no total; nenhuma lista vazia. |
| `transcricao` | texto | Transcrição integral da audiência. | 4.437 a 147.728 palavras (mediana 16.424,5). |

Nenhum texto de `assunto`, `nome`, `cargo` ou `opinioes` é vazio ou só com espaços. O schema não
exige isso: é uma propriedade observada neste arquivo, e a leitura aceita textos vazios. O arquivo não
tem campo de data nem de comissão: a data usada no split é extraída do carimbo de publicação de
`materia` (ver [Manifesto de split](#manifesto-de-split-temporal_v1json)). Segundo a descrição do
dataset, as opiniões foram extraídas da matéria; por isso uma opinião pode não ter correspondente na
fala da pessoa.

Na transcrição, cada turno de fala começa com um cabeçalho `O SR. NOME` ou `A SRA. NOME`, às vezes
seguido de `(partido, cargo ou nome social)`, e de ` - `. `split_into_turns` corta a transcrição nesses
cabeçalhos; o índice de um turno nessa lista é o `speaker_turn` do registro UDV. As sentenças
candidatas de um participante são divididas dentro de cada turno atribuído a ele, nunca na fala
concatenada: nas 206 audiências são 115.599 sentenças, de 1.020 participantes com pelo menos um
turno.

## Registro UDV (`<run>.jsonl`)

Uma linha por opinião do LDS, na ordem audiência, participante, opinião. `udv_v1.jsonl` e
`udv_v1_pre.jsonl` cobrem as 206 audiências (2.203 registros cada) e diferem só no corte. Modelo:
`UdvRecord`, com a ordem de campos igual à ordem de chaves da linha.

| Campo | Tipo | Conteúdo |
|---|---|---|
| `id` | texto | `udv-<hearing_id>-<i>-<j>`, com `i` o índice do participante em `envolvidos` e `j` o índice da opinião em `opinioes`, ambos a partir de 0. |
| `hearing_id` | inteiro | `id` da audiência no LDS. |
| `actor.name` | texto | Cópia de `envolvidos[i].nome`. |
| `actor.role` | texto | Cópia de `envolvidos[i].cargo`. |
| `proposition` | texto | Cópia literal de `opinioes[j]`. |
| `evidence` | objeto ou `null` | Trecho da fala do participante que sustenta a opinião. `null` exatamente quando `tier` é `person_not_resolved` ou `no_evidence`. |
| `tier` | texto | Nível de confiança da ligação (valores abaixo). |
| `provenance` | texto ou `null` | Como a evidência foi escolhida: `weak` (casamento de citação), `model` (similaridade do encoder) ou `null` (sem evidência). |
| `method.encoder` | texto | Nome do encoder da execução. |
| `method.revision` | texto | Revisão do encoder (commit do Hugging Face, ou versão do scikit-learn no TF-IDF). |
| `method.embedding_threshold` | número | Corte entre `semantic_match_high` e `semantic_match_weak`. |

`method` é igual em todos os registros de uma execução; em `udv_v1`, o encoder é
`PORTULAN/serafim-335m-portuguese-pt-sentence-encoder`, revisão
`a01887015444f7599669c509447c5bdbce958916`, com corte 0,45; em `udv_v1_pre`, o mesmo encoder com corte
0,47.

Campos de `evidence`:

| Campo | Tipo | Conteúdo |
|---|---|---|
| `text` | texto | Sentença de um único turno do participante, com espaços normalizados. Em `direct_quote`, a sentença do turno da ocorrência escolhida que contém o prefixo (as partes que o prefixo toca, quando ele atravessa uma fronteira de sentença); nos demais, uma sentença candidata. |
| `support_type` | texto | `direct_quote`, `semantic_with_short_quote` ou `semantic_similarity`. |
| `score` | número ou `null` | Similaridade de cosseno entre opinião e sentença. `null` em `direct_quote`. Em `udv_v1`, vai de 0,275 a 0,989. |
| `quote_prefix` | texto ou `null` | Prefixo da citação vencedora da opinião (a de mais palavras de prefixo encontradas nos turnos do participante). Preenchido em `direct_quote` e em `semantic_with_short_quote`, `null` em `semantic_similarity`. |
| `start_char` | inteiro ou `null` | Início do trecho em `transcricao` (índice de caractere do texto Python, intervalo semiaberto). |
| `end_char` | inteiro ou `null` | Fim do trecho em `transcricao`. |
| `speaker_turn` | inteiro ou `null` | Índice, na lista de `split_into_turns`, do turno de onde a sentença veio; o trecho é procurado só nele. |

Os três offsets são nulos juntos ou preenchidos juntos. Quando preenchidos,
`transcricao[start_char:end_char]` com espaços normalizados é igual a `text` e o trecho fica dentro do
turno `speaker_turn`. Em `udv_v1` e em `udv_v1_pre`, as 2.105 evidências têm offsets; `verify-udvs`
acusa `evidence_offsets_missing` em qualquer evidência sem eles. Em `udv_v0`, gerado antes da
segmentação por turno, a evidência de `udv-1-1-2` não tinha offsets, porque juntava o fim de um turno
com o começo do turno seguinte da mesma pessoa.

Valores de `tier`, com as contagens de `udv_v1` e de `udv_v1_pre`:

| `tier` | Condição | `evidence` | `provenance` | `udv_v1` | `udv_v1_pre` |
|---|---|---|---|---|---|
| `quote_found` | O prefixo da citação vencedora tem 6 ou mais palavras e aparece num turno do participante. | `direct_quote` | `weak` | 277 | 277 |
| `semantic_match_high` | Sem citação confiável; sentença candidata mais similar com `score >= embedding_threshold`. | semântica | `model` | 1.785 | 1.776 |
| `semantic_match_weak` | Idem, com `score < embedding_threshold`. | semântica | `model` | 43 | 52 |
| `no_evidence` | Participante encontrado na transcrição, sem citação confiável e sem sentença candidata. | `null` | `null` | 8 | 8 |
| `person_not_resolved` | Nenhum turno de fala atribuído ao participante. | `null` | `null` | 90 | 90 |

Os 9 registros que mudam de `semantic_match_weak` para `semantic_match_high` entre `udv_v1_pre` e
`udv_v1` são os de score entre 0,45 e 0,47; as evidências dos dois arquivos são idênticas.

Em `udv_v1`, `quote_prefix` tem 6 palavras em 152 registros `direct_quote`, 10 em 122 e 7 em 3
(citações com exatamente 7 palavras, em que o degrau de 10 palavras cobre a citação inteira). Em
`semantic_with_short_quote` o prefixo tem de 1 a 5 palavras (4 em 65, 3 em 37, 2 em 7, 1 em 1 e 5 em
1). `support_type` se distribui em 277 `direct_quote`, 111 `semantic_with_short_quote` e 1.717
`semantic_similarity`; os números são os mesmos em `udv_v1_pre`.

## Arquivo de cobertura (`<run>_coverage.json`)

Resumo de uma execução de `build-udvs`, gravado ao lado do JSONL. `verify-udvs` reconta cada contador a
partir dos registros e acusa `coverage` quando algum diverge.

| Campo | Conteúdo | `udv_v1` |
|---|---|---|
| `run_name` | Nome da execução, igual ao nome do arquivo. | `udv_v1` |
| `created_at` | Instante UTC da execução. | |
| `hearings.count`, `hearings.ids` | Audiências processadas, na ordem do LDS. | 206, ids 1 a 206 |
| `people.total`, `people.resolved` | Participantes e participantes com pelo menos um turno de fala. | 1.065 e 1.020 |
| `opinions.total` | Registros UDV. | 2.203 |
| `opinions.by_tier` | Contagem por `tier`, sempre com as cinco chaves, na ordem da tabela de `tier`. | 277, 1.785, 43, 8, 90 |
| `evidence_offsets.total`, `evidence_offsets.located` | Evidências e evidências com offsets. | 2.105 e 2.105 |
| `evidence_support_types` | Contagem por `support_type`, sempre com as três chaves. | 277, 111, 1.717 |
| `pipeline` | Regras da rodada, em texto: `sentence_segmentation`, `sentence_boundary_pattern` (a regex), `quote_patterns` (lista de regex), `quote_search`, `quote_selection`, `quote_occurrence` e `trusted_prefix_words`. É gerada a partir da política de citação usada na construção. | segmentação por turno, 2 padrões de citação, 6 palavras |
| `encoder_runtime` | `device`, `max_seq_length` e `embedding_dimension` do encoder. | `mps`, 128, 1024 |
| `timing` | `elapsed_seconds` (soma), `mean_seconds_per_hearing` e `max_seconds_per_hearing`, medidos por audiência na construção. | `elapsed_seconds` 23,5 |
| `environment` | `python`, `torch`, `sentence_transformers` e `platform`. As versões vêm dos metadados dos pacotes instalados e ficam `null` sem o extra `embeddings`. | Python 3.12.13, torch 2.13.0, sentence-transformers 5.6.1 |
| `config` | Cópia do TOML de configuração inteiro, inclusive chaves que a biblioteca não usa. | |

Em `udv_v1_pre_coverage.json` os contadores são os de `udv_v1`, exceto `opinions.by_tier` (277,
1.776, 52, 8, 90), e `config.evidence` guarda o registro da calibração anterior (corte 0,47). Os
arquivos de cobertura de `udv_v0` e `dev20`, anteriores à ADR 0002, não têm a seção `pipeline`.

## JSON de demonstração (`export-hearing`)

Um objeto JSON por arquivo, gravado por `bookworm export-hearing` com
`json.dump(..., ensure_ascii=False, indent=2)`, com tudo o que a demonstração web precisa para mostrar
uma audiência de uma execução sem recalcular nada: a transcrição, os turnos com as sentenças
candidatas e os offsets de cada uma, os participantes, as UDV da execução e, para cada opinião, as
sentenças do participante mais parecidas com ela. Os candidatos permitem mostrar por que a evidência
foi escolhida e quais trechos ficaram perto dela. Chaves de primeiro nível, nesta ordem: `hearing`,
`transcript`, `turns`, `people`, `udvs`, `run` e, só com `--verifier-report`, `signals` (ver
[Sinais do verificador](#sinais-do-verificador---verifier-report)).

| Campo | Tipo | Conteúdo |
|---|---|---|
| `hearing.id` | inteiro | `id` da audiência no LDS. |
| `hearing.split` | texto ou `null` | Conjunto (`train`, `validation` ou `test`) da audiência no manifesto passado em `--split-manifest`; `null` sem a opção. |
| `hearing.article_date` | texto ou `null` | Data de publicação da matéria (`AAAA-MM-DD`), extraída como no split; `null` sem carimbo. |
| `hearing.assunto`, `hearing.materia` | texto | Cópias de `metadados.assunto` e de `materia`. |
| `hearing.transcript_chars`, `hearing.transcript_words` | inteiro | Tamanho de `transcricao` em caracteres e em palavras separadas por espaço. |
| `transcript` | texto | Cópia de `transcricao`; todos os offsets do arquivo apontam para ela. |
| `turns[]` | lista | Todos os turnos de `split_into_turns`, inclusive os de quem não está em `envolvidos`. |
| `turns[].index` | inteiro | Índice do turno (o `speaker_turn` das UDV). |
| `turns[].speaker`, `turns[].party` | texto | Nome no cabeçalho do turno e conteúdo do parêntese (`""` quando não há). |
| `turns[].start`, `turns[].end` | inteiro | Offsets da fala do turno, sem o cabeçalho. |
| `turns[].sentences[]` | lista | Sentenças candidatas do turno, na ordem do texto, com `text`, `start` e `end`; os offsets são procurados só nesse turno e ficam `null` se o texto não for encontrado nele. |
| `people[]` | lista | Participantes de `envolvidos`, na ordem do LDS: `index`, `name`, `role`, `turns` (índices dos turnos atribuídos) e `resolved` (pelo menos um turno). |
| `udvs[]` | lista | Os registros UDV da audiência, lidos de `<run>.jsonl`, com os campos de `UdvRecord` na mesma ordem e três campos a mais (quatro com `--verifier-report`, que acrescenta `signals` no fim). |
| `udvs[].candidates[]` | lista | Até `--top-k` (padrão 8) unidades candidatas do participante, em ordem decrescente de similaridade de cosseno com a opinião, calculada com os embeddings que `build-udvs` gravou no cache para a execução; empates ficam na ordem das unidades. A unidade é a de `semantic_unit` na configuração: a sentença, ou a janela de sentenças seguidas do mesmo turno (`window2` em `udv_v2`), com o texto do trecho da transcrição que ela cobre. Cada uma tem `text`, `score` (arredondado a 4 casas), `turn`, `start` e `end`. Vazia quando o participante não tem sentença candidata. |
| `udvs[].n_candidates` | inteiro | Número de unidades candidatas do participante. |
| `udvs[].quotes` | lista de texto | Citações extraídas da opinião com os padrões da rodada, na ordem da opinião. |
| `run.name` | texto | Nome da execução. |
| `run.encoder`, `run.revision`, `run.threshold` | texto, texto, número | `method.encoder`, `method.revision` e `method.embedding_threshold` dos registros. |
| `run.semantic_unit`, `run.quote_extent` | texto | Só quando a configuração muda o padrão: a unidade candidata (`window2` em `udv_v2`) e a extensão da evidência de citação (`full_quote` em `udv_v2`). Sem elas, a execução usa sentenças e a sentença do começo da citação, e os bytes do arquivo são os de antes desses campos existirem. |

Os offsets contam caracteres Unicode, como os índices de `str` do Python. Um navegador indexa texto em
unidades UTF-16, e as duas contagens passam a diferir depois do primeiro caractere fora do plano básico
(um emoji, por exemplo); por isso a demonstração web recusa um arquivo em que `transcript_chars` não é
igual ao tamanho da transcrição medido por ela.

O comando lê a execução com a configuração passada em `--config` e confere que a seção `pipeline` do
arquivo de cobertura é a que essa configuração descreve (`pipeline_description` com `semantic_unit`,
`quote_extent` e a política de citações) e que o `embedding_threshold` gravado é o da configuração; se
não for, para com código 2 e diz em que campos a execução difere. Assim `udv_v1` se exporta com
`configs/udv.toml` e `udv_v2` com `configs/udv_v2.toml`, e nenhuma das duas com a configuração da
outra.

Nas UDV semânticas, o primeiro candidato é a própria evidência: mesmo texto, turno e offsets, e
`score` igual ao da evidência arredondado a 4 casas. Em `quote_found` os candidatos continuam sendo os
da similaridade, e a evidência pode não estar entre eles. O comando para com código 2 quando os
registros da audiência na execução não são exatamente as opiniões do LDS, na mesma ordem, ou quando o
encoder da configuração tem nome ou revisão diferente de `method`, e também quando falta no cache o
arquivo de embeddings das sentenças ou das opiniões da audiência (`EmbeddingCacheMissError`, com o
caminho esperado); a exportação nunca calcula embeddings
([ADR 0004](adr/0004-exports-read-only-the-run-cache.md)). Para a audiência 70 de `udv_v1`
(teste `tests/integration/test_export_hearing.py`): 21 turnos, 6 participantes, 12 UDV, 10 evidências,
todas as posições conferidas contra a transcrição.

### Sinais do verificador (`--verifier-report`)

A pasta de cada afirmação na demonstração web mostra, ao lado da similaridade de cosseno, a decisão do
verificador primário, as oito perguntas que ele combina e a cópia em inglês que ele leu. Esses números
já existem nos artefatos de `experiments/artifacts/udv/`; a opção `--verifier-report` (em
`export-hearing` e `export-site`) copia os valores para o JSON da audiência, sem carregar modelo nem
recalcular nada. O ponto de entrada é o relatório do verificador (`udv_v1_verifier_report.json`), que
registra, cada um com o seu sha256: os registros da execução (`inputs.udv`), a saída do verificador
(`outputs`), os arquivos de notas das três leituras (`inputs.udv_score_files`: `laya_multi_pt`,
`laya_en_en`, `xnli_mdeberta`) e os relatórios dessas notas (`score_runs`). O relatório de
`laya_en_en` diz onde está o cache de tradução (`translation.store`) e com que assinatura e segmentação
ele foi gravado. Os caminhos do relatório são relativos à pasta de onde o comando roda (`experiments/`).

Sem a opção, os arquivos gravados são os mesmos, byte a byte, de uma exportação anterior à opção (teste
`tests/unit/test_signals.py`, que também confere que tirar os campos `signals` de uma exportação com a
opção devolve esses bytes). `index.json` não muda com a opção. O comando para com código 2 quando: o
relatório ou um dos arquivos que ele registra não existe; um sha256 não bate, inclusive o dos registros
da execução lidos pelo comando; os ids das UDVs da saída do verificador não são exatamente os da execução, ou os de um arquivo de notas
não são exatamente os das UDV com nota (a mensagem diz quantos faltam e quantos sobram, com um exemplo
de cada); uma linha da saída tem nota sem probabilidade ou sem decisão, ou a audiência, o `tier` ou o
`support_type` dela diferem dos do registro; uma UDV com nota não tem evidência no registro; as perguntas das duas leituras
Laya são diferentes; o corte do ajuste (`primary.fit.threshold`) difere do corte usado no teste final;
ou um texto que precisa de tradução não está no cache.

`signals` do primeiro nível, com as chaves nesta ordem:

| Campo | Tipo | Conteúdo |
|---|---|---|
| `verifier.name` | texto | O candidato primário do relatório (`primary.candidate`). |
| `verifier.threshold` | número | O corte da probabilidade, ajustado no conjunto indicado a seguir. |
| `verifier.threshold_fitted_on` | texto | Conjunto em que o corte foi ajustado (`train` para `udv_v1`). |
| `verifier.report_sha256` | texto | sha256 do relatório lido. |
| `verifier.udv_threshold` | objeto | Só quando o relatório tem um corte para premissas de UDV (`udv_v2_verifier_report.json`): `value` (o corte exato, 0,2428652… em `udv_v2`), `rounded` (0,2429), `rule` (`legacy_random`), `interval` (intervalo de bootstrap por audiência, `[0,221, 0,2782]`) e `source` (`artifacts/calibration/udv_verifier_threshold_v1.json`). A página usa esse corte como principal e mostra `threshold` (0,7478, o corte do treino no benchmark NLI) como segundo traço. |
| `scorers.<nome>` | objeto | Para `laya_multi_pt`, `laya_en_en` e `xnli_mdeberta`: `model`, `revision` e `language` (`pt` ou `en`, a língua do texto que o modelo leu). |
| `translation` | objeto | `name`, `revision` e `license` do modelo de tradução. |
| `questions[]` | lista | As perguntas Laya, na ordem das notas: `id`, `type` (`choice`, `noul` para sim ou não, `score` para escala), `instructions` e `options` como o modelo as recebeu, `support_option` (a opção que favorece a afirmação nas perguntas de escolha; `null` nas outras) e `reverses` (o id da pergunta cujas opções esta apresenta em ordem inversa, ou `null`). |

`udvs[].signals`, com as chaves nesta ordem:

| Campo | Tipo | Conteúdo |
|---|---|---|
| `scored` | booleano | Verdadeiro quando o verificador deu nota à UDV, o que acontece exatamente nas UDV com `evidence`. Quando é falso, os outros quatro campos são `null`. |
| `verifier.probability` | número | Probabilidade de o trecho sustentar a opinião, segundo o verificador. |
| `verifier.supported` | booleano | Cópia de `supported_at_train_threshold` da saída do verificador, que é `probability >= verifier.threshold`; a demonstração web recusa o arquivo se as duas coisas diferirem. |
| `verifier.supported_at_udv_threshold` | booleano | Só com `verifier.udv_threshold`: cópia do campo de mesmo nome da saída do verificador, conferida contra `probability >= udv_threshold.value`; o comando para com código 2, e a página recusa o arquivo, se as duas coisas diferirem. |
| `laya.laya_multi_pt`, `laya.laya_en_en` | objeto | Um número de 0 a 1 por id de pergunta, a favor da afirmação, lido de `items[0].signals` do arquivo de notas: nas perguntas de escolha, a probabilidade da `support_option`; nas de sim ou não, a de sim; nas de escala, o nível esperado dividido por 4. `laya_multi_pt` leu a frase e a opinião em português; `laya_en_en`, as cópias em inglês. |
| `xnli` | objeto | `entailment`, `neutral` e `contradiction` do outro modelo de NLI, que leu o texto em português, e `truncated` (se a entrada foi cortada). |
| `translation.premise`, `translation.hypothesis` | texto | As cópias em inglês que `laya_en_en` leu: a frase da evidência, dividida em segmentos como na tradução e com os segmentos traduzidos unidos por espaço, e a opinião traduzida inteira. |

A regra de `laya` foi conferida nas 33.680 respostas de `udv_v1` (2.105 UDV com nota, 8 perguntas, 2
leituras) contra as distribuições gravadas em cada resposta, sem nenhuma diferença. Nas 2.105 UDV,
`supported` é igual a `probability >= threshold`, e das 2.105 frases de evidência, 4 são divididas em
mais de um segmento para a tradução; todos os segmentos estão no cache. A chave de cada texto no cache é
o sha256 da assinatura do cache seguida do caractere `\x1e` e do texto normalizado; `Segmentation`
repete, na biblioteca, a divisão em segmentos usada na tradução, com as opções gravadas no cache.

## Diretório de demonstração (`export-site`)

A demonstração web lista todas as matérias de uma execução e abre qualquer uma delas lendo arquivos
estáticos, sem servidor de aplicação. `bookworm export-site` grava esse conjunto num diretório (por
padrão `bookworm/web/app/data/`, fora do Git):

```text
<saída>/
  index.json
  hearings/
    1.json
    ...
    206.json
```

Cada `hearings/<id>.json` é o arquivo que `export-hearing --hearing <id>` gravaria com as mesmas
opções (`--top-k`, `--split-manifest`), byte a byte, então a demonstração lê um e outro do mesmo jeito;
o formato é o da seção anterior. `index.json` resume cada audiência para a tela que lista as matérias,
sem que ela precise abrir os 206 arquivos. Nenhum arquivo do diretório tem data de criação: duas
exportações com as mesmas entradas geram os mesmos bytes. O índice é gravado depois de todas as
audiências, e um `index.json` que já exista no diretório é apagado antes da primeira (com
`--overwrite`), então uma exportação que para no meio deixa o diretório sem `index.json`. Só as
audiências listadas no índice pertencem à exportação: `--overwrite` não apaga arquivos de audiências
que ficaram fora da execução, e a demonstração web recusa abrir uma audiência que não está no índice
ou cujo bloco `run` difere do dele.

`index.json` é um objeto com duas chaves, nesta ordem: `run` e `hearings`.

| Campo | Tipo | Conteúdo |
|---|---|---|
| `run` | objeto | O bloco `run` dos arquivos de audiência (`name`, `encoder`, `revision`, `threshold` e, quando existem, `semantic_unit` e `quote_extent`). Precisa ser igual em todas as audiências; se não for, o comando para com código 2. |
| `hearings[]` | lista | Uma entrada por audiência da execução, em ordem crescente de `id`, com as chaves na ordem desta tabela. |
| `hearings[].id`, `hearings[].split`, `hearings[].article_date`, `hearings[].assunto` | inteiro, texto ou `null`, texto ou `null`, texto | Cópias dos campos de mesmo nome em `hearing` do arquivo da audiência. |
| `hearings[].title` | texto | Título curto para exibição, pela regra descrita abaixo. |
| `hearings[].n_udvs` | inteiro | Número de UDV da audiência (tamanho de `udvs`). |
| `hearings[].n_people`, `hearings[].n_people_resolved` | inteiro | Número de participantes em `people` e quantos deles têm pelo menos um turno atribuído (`resolved`). |
| `hearings[].tiers` | objeto | Número de UDV por `tier`, com os cinco valores sempre presentes, na ordem de `Tier`, e `0` quando não há nenhuma. |
| `hearings[].support_types` | objeto | Número de UDV com evidência por `evidence.support_type`, com os três valores sempre presentes, na ordem de `SupportType`. UDV sem evidência (`no_evidence`, `person_not_resolved`) não entram, então a soma é o número de UDV com `evidence`. |
| `hearings[].transcript_words` | inteiro | Cópia de `hearing.transcript_words`. |
| `hearings[].actors` | lista de texto | `name` de cada participante de `people`, na mesma ordem; a posição na lista é `people[].index`. |

O título é a primeira linha não vazia de `materia` (a manchete da matéria), com os espaços
normalizados; se `materia` não tiver nenhuma linha com texto, é `assunto`, também normalizado. Um
título com mais de 120 caracteres é cortado no último espaço que deixa o resultado, já com as
reticências (`…`), em até 120 caracteres; vírgula, ponto e vírgula, dois pontos e hífen soltos antes
das reticências são removidos, e uma palavra única maior que o limite é cortada no caractere 119. A
manchete identifica a matéria que a demonstração mostra, e o limite impede que uma matéria sem quebra
de linha vire um título do tamanho do texto. No LDS, as manchetes têm de 58 a 124 caracteres, todas
diferentes, e 1 das 206 é cortada.

Para `udv_v1` com `temporal_v1` (teste `tests/integration/test_export_site.py`): 206 arquivos de
audiência, 2.203 UDV, com as contagens por `tier` e por `support_type` iguais às do arquivo de
cobertura (`quote_found` 277, `semantic_match_high` 1.785, `semantic_match_weak` 43, `no_evidence` 8,
`person_not_resolved` 90; `direct_quote` 277, `semantic_with_short_quote` 111, `semantic_similarity`
1.717). Os arquivos de audiência somam 76.526.309 bytes: o menor tem 115.500 (audiência 67), a mediana
é 329.899 e o maior tem 3.377.789 (audiência 6, a transcrição mais longa do LDS, com 147.728
palavras). O índice tem 184.112 bytes. Com `--verifier-report artifacts/udv/udv_v1_verifier_report.json`,
os arquivos de audiência somam 80.892.695 bytes (o menor tem 132.368, a mediana é 350.800,5 e o maior
tem 3.423.632, as mesmas audiências), e o índice continua com os mesmos 184.112 bytes.

## Falas por ator

Arquivos gravados por `build-udvs --actors-config` (ver [Falas por ator](actors.md)). Os quatro
primeiros têm o formato de `experiments/src/experiments/actors/build_speeches.py` e são byte a byte iguais aos que o
script grava com a mesma configuração; o quinto é novo.

### Registro de falas (`single_hearing_path`, `multi_hearing_path`)

JSONL com um ator por linha, em ordem alfabética da chave normalizada do ator. O primeiro arquivo tem
os atores de uma única audiência (1.550 no LDS completo) e o segundo os de duas ou mais (301). Modelo:
`ActorSpeechRecord`, com `ActorHearing` e `ActorTurn`, na ordem de campos abaixo. A chave normalizada
não é gravada; quem precisa dela usa o registro em memória (`ActorSpeeches.records`, indexado por
chave).

| Campo | Tipo | Conteúdo |
|---|---|---|
| `actor` | texto | Nome de exibição, escolhido entre as grafias dos cabeçalhos; único entre todos os registros. |
| `has_party_header` | booleano | `true` quando algum turno do ator tem partido e UF no cabeçalho. |
| `party_uf` | lista de texto | Textos de partido e UF dos cabeçalhos, sem repetição, na ordem em que aparecem (`Bloco/PT - SP`, `PSB - PE`). |
| `hearings[]` | lista | Audiências em que o ator tem turno mantido, em ordem crescente de `hearing_id`. |
| `hearings[].hearing_id` | inteiro | `id` da audiência no LDS. |
| `hearings[].full_speech` | texto | Textos dos turnos da audiência unidos por uma linha em branco (`"\n\n"`). |
| `hearings[].turns[]` | lista | Turnos mantidos, na ordem da transcrição. |
| `turns[].turn_index` | inteiro | Índice do turno em `split_into_turns` (o mesmo `speaker_turn` das UDVs). |
| `turns[].role` | texto | `chair` (cabeçalho em `chair_names`) ou `speaker`. |
| `turns[].start_char`, `turns[].end_char` | inteiro | Offsets da fala do turno em `transcricao`, sem o cabeçalho. |
| `turns[].text` | texto | `transcricao[start_char:end_char]`, sem normalizar espaços. |

### Nomes ambíguos (`ambiguous_names_path`)

Objeto com `criterion` (o critério, em texto) e `pairs`. Cada par junta duas chaves distintas, em ordem
alfabética, cujos conjuntos de palavras são iguais ou um contido no outro: `relation`
(`equal_tokens` ou `name_subset`), `a` e `b` (cada um com `key`, `actor`, `has_party_header`,
`party_uf`, `hearing_ids` e `turns`) e `shared_hearing_ids`. No LDS completo são 81 pares, 1
`equal_tokens` e 80 `name_subset`.

### Contagens (`stats_path`)

| Campo | Conteúdo | LDS completo |
|---|---|---|
| `dataset` | `path` (como escrito na configuração de atores), `sha256` e `hearings` (audiências processadas). | 206 |
| `policy` | `chair_min_words`, `non_person_keys` e a descrição da fusão (`merge`). | 50 |
| `turns.total`, `turns.kept`, `turns.verified_against_transcript` | Turnos de `split_into_turns`, turnos mantidos e turnos conferidos contra a transcrição (sempre iguais aos mantidos). | 17.264, 13.190, 13.190 |
| `turns.dropped_non_person` | `turns_dropped`, `by_key` e a lista `turns` (`key`, `raw_name`, `hearing_id`) dos turnos de chaves que não são pessoa. | 32 |
| `turns.dropped_stage_direction`, `turns.dropped_empty`, `turns.dropped_short_chair` | Turnos descartados por serem só rubricas, vazios ou de presidência curta. | 36, 5, 4.001 |
| `merges` | `groups`, `alias_keys`, `alias_turns_kept`, `reassignments`, `reassigned_turns_kept`. | 37, 40, 130, 1, 1 |
| `hearings_per_actor` | Número de atores por número de audiências, com a chave como texto. | `"1"`: 1.550, `"30"`: 1 |
| `files.single_hearing`, `files.multi_hearing` | `path`, `actors`, `with_party_header`, `without_party_header`, `speaker_words` e `chair_words` de cada arquivo. | 1.550 e 301 atores |
| `ambiguous_name_pairs` | `path`, `criterion`, `pairs`, `equal_tokens`, `name_subset`. | 81, 1, 80 |

### Ligações UDV → ator (`<run>_actor_links.jsonl`)

Uma linha por registro UDV, na mesma ordem de `<run>.jsonl`. Modelo: `UdvActorLink`.

| Campo | Tipo | Conteúdo |
|---|---|---|
| `udv_id` | texto | `id` do registro UDV. |
| `hearing_id` | inteiro | `id` da audiência. |
| `actor_key` | texto ou `null` | Chave normalizada do ator (depois das fusões) com mais turnos entre os atribuídos à pessoa; empate vai para a primeira em ordem alfabética. |
| `actor` | texto ou `null` | Nome de exibição desse ator, igual ao campo `actor` do registro de falas e do perfil; é a chave de junção. |
| `matched_turns` | inteiro | Turnos que a resolução de envolvidos da UDV atribuiu à pessoa. `0` exatamente nas UDVs `person_not_resolved`. |
| `linked_turns` | inteiro | Quantos desses turnos sobreviveram à política de falas e têm a chave `actor_key`. |

`actor_key` e `actor` são nulos juntos, exatamente quando `linked_turns` é `0`. Em `udv_v1`: 2.203
ligações, 2.104 com ator, 90 com `matched_turns` 0, 9 com turnos atribuídos mas nenhum mantido, 355
com `0 < linked_turns < matched_turns`.

## Manifesto de split (`temporal_v1.json`)

Define a partição das 206 audiências em treino, validação e teste. O LDS não tem data, então a ordem
temporal vem do primeiro carimbo `DD/MM/AAAA - HH:MM` de `materia` que não faz parte de um
`Atualizado em`, isto é, a data de publicação da matéria, que não é necessariamente a data da audiência.

| Campo | Conteúdo | `temporal_v1` |
|---|---|---|
| `split_version` | Nome do split, também usado como nome do arquivo. | `temporal_v1` |
| `grouping_method` | Descrição textual do método. | |
| `seed` | Semente da configuração. O split temporal não a consome; ela fica registrada para os splits que dependem de sorteio. | 42 |
| `created_at` | Instante UTC da construção. | |
| `dataset.path`, `dataset.sha256` | Caminho do LDS como escrito na configuração e o hash esperado. | `dataset/PublicHearingBR_LDS.jsonl` |
| `date_field` | Descrição do campo de data usado. | |
| `boundaries.train_end` | Última data do treino. | 2023-11-13 |
| `boundaries.validation_start` | Primeira data da validação. | 2023-11-21 |
| `boundaries.validation_end` | Última data da validação. | 2023-12-20 |
| `boundaries.test_start` | Primeira data do teste. | 2024-03-05 |
| `boundaries.train_gap_days` | Dias entre `train_end` e `validation_start`. | 8 |
| `boundaries.test_gap_days` | Dias entre `validation_end` e `test_start`. | 76 |
| `boundaries.train_fraction_reached` | Fração das audiências até `train_end`, arredondada a 4 casas. | 0,699 |
| `boundaries.validation_end_fraction_reached` | Fração acumulada até `validation_end`. | 0,8544 |
| `train`, `validation`, `test` | Listas de `id`, ordenadas por data e, na mesma data, por `id`. | 144, 32 e 30 audiências |
| `article_dates` | Data extraída de cada audiência, com a chave `id` como texto, em ordem crescente de `id`. | 206 entradas |

Cada corte só pode cair entre duas datas consecutivas separadas por pelo menos
`min_boundary_gap_days` (2 em `temporal_v1`), para que um erro de um dia na data extraída não mude uma
audiência de conjunto. Entre os cortes elegíveis, escolhe-se o de fração acumulada mais próxima de
`train_fraction` (0,70) e, depois dele, o mais próximo de `train_fraction + validation_fraction` (0,85);
empates vão para o maior intervalo e, depois, para a data mais antiga.

## Relatório de split (`temporal_v1_report.json`)

Números que sustentam o manifesto, gravados ao lado dele. `verify-splits` reconta os contadores de cada
conjunto e acusa divergência.

| Campo | Conteúdo |
|---|---|
| `split_version`, `created_at` | Como no manifesto. |
| `date_extraction` | Validação da data extraída contra o próprio texto da matéria (abaixo). |
| `calendar` | `first_date`, `last_date`, `distinct_dates`, `hearings_per_date_max` e `by_year` (audiências por ano). |
| `boundaries` | Cópia de `boundaries` do manifesto. |
| `cut_candidates_considered` | Número de cortes elegíveis pelo intervalo mínimo (65 em `temporal_v1`). |
| `splits.<nome>` | Por conjunto: `hearings`, `share_of_hearings`, `first_date`, `last_date`, `distinct_dates`, `people` (entradas de `envolvidos`), `opinions`, `udvs` e `udvs_by_tier` (só os `tier` presentes, em ordem alfabética). |
| `leakage.actors` | `distinct_actors` (nomes distintos depois de `strip()` e maiúsculas), `in_more_than_one_split` e `in_train_and_test`. |
| `leakage.near_duplicates` | Similaridade TF-IDF entre matérias de conjuntos diferentes: `method`, `threshold`, `cross_split_pairs`, `above_threshold`, `max`, `p99`, `median` e `top_pairs` (`similarity`, `hearings`, `splits`), com os `near_duplicate_pairs` pares mais similares. |
| `udv_source` | Caminho do JSONL de UDV usado nas contagens, ou `null` quando ele não existe (as contagens de `udvs` ficam 0). |
| `environment` | `python` e `platform`. |
| `config` | Cópia do TOML de configuração. |

`date_extraction` existe porque a data de publicação pode diferir da data da audiência. Muitas matérias
nomeiam o dia do evento na forma `nesta quarta-feira (17)`; a menção é resolvida para a data mais
recente, até 30 dias antes da publicação, com aquele dia do mês, e confere-se se o dia da semana bate.
Campos: `hearings`, `with_publication_timestamp`, `with_update_timestamp`,
`with_current_event_mention` (audiências com menção iniciada por `nesta` ou `neste`),
`confirmed_by_at_least_one_mention`, `confirmed_by_no_mention`, `current_event_mentions`,
`weekday_agrees`, `weekday_disagrees`, `lag_days_when_agreeing` (dias entre evento e publicação, quando o
dia da semana bate), `disagreements` (`hearing_id`, `article_date`, `mention`, `resolved_date`) e
`hearings_without_any_agreeing_mention`. Em `temporal_v1`: 206 matérias com carimbo, 13 com
atualização, 156 com menção ao dia do evento, 155 confirmadas, defasagem de 0 dia em 142 menções e de 1
dia em 15. Como a defasagem máxima observada (1 dia) é menor que os intervalos das fronteiras (8 e 76
dias), `verify-splits` confere que nenhuma fronteira fica dentro dessa incerteza.

Contagens por conjunto em `temporal_v1`:

| Conjunto | Audiências | Período | Pessoas | Opiniões | UDVs |
|---|---|---|---|---|---|
| `train` | 144 | 2021-11-18 a 2023-11-13 | 745 | 1.536 | 1.536 |
| `validation` | 32 | 2023-11-21 a 2023-12-20 | 161 | 308 | 308 |
| `test` | 30 | 2024-03-05 a 2024-05-09 | 159 | 359 | 359 |

## Como conferir

Os contadores dos artefatos (pessoas, `tier`, `support_type`, offsets, tamanhos dos conjuntos e
fronteiras) e as comparações byte a byte descritas nas convenções do início deste documento são
fixados pelos testes de `tests/integration/` com o marcador `dataset`; a reconstrução das UDVs a partir
do cache exige também `BOOKWORM_EMBEDDING_CACHE`. As estatísticas descritivas do LDS e do JSONL de UDV
citadas acima saem deste trecho, rodado a partir de um projeto que tenha
`dataset/PublicHearingBR_LDS.jsonl` e `artifacts/udv/udv_v1.jsonl`. As da tabela do LDS e as contagens
de palavras de `quote_prefix` são fixadas também por `tests/integration/test_documented_statistics.py`:

```python
import statistics
from collections import Counter
from pathlib import Path

from bookworm import check_article_date, load_hearings, load_udv_jsonl

hearings = load_hearings(
    Path("dataset/PublicHearingBR_LDS.jsonl"),
    expected_sha256="c4e392ab95ce22f6228eace16f15e9efe1c846724846b873dec420aec3d872c0",
)
people = [person for hearing in hearings for person in hearing.metadados.envolvidos]
ids = [hearing.id for hearing in hearings]
print("audiências:", len(hearings), "ids:", ids[0], "a", ids[-1])
print("ids crescentes e sem repetição:", ids == sorted(set(ids)))
print(
    "textos vazios ou só com espaços:",
    {
        "assunto": sum(not h.metadados.assunto.strip() for h in hearings),
        "nome": sum(not p.nome.strip() for p in people),
        "cargo": sum(not p.cargo.strip() for p in people),
        "opinioes": sum(not o.strip() for p in people for o in p.opinioes),
    },
)
print("envolvidos por audiência:", Counter(len(h.metadados.envolvidos) for h in hearings))
print("envolvidos:", len(people), "nomes distintos:", len({p.nome.strip().upper() for p in people}))
print("opiniões:", sum(len(p.opinioes) for p in people), Counter(len(p.opinioes) for p in people))
for field in ("materia", "transcricao"):
    words = [len(getattr(h, field).split()) for h in hearings]
    print(field, "palavras:", min(words), statistics.median(words), max(words))
checks = [check_article_date(h.materia) for h in hearings]
print("com data de publicação:", sum(c.article_date is not None for c in checks))
print("com 'Atualizado em':", sum(c.updated_at is not None for c in checks))

records = load_udv_jsonl(Path("artifacts/udv/udv_v1.jsonl"))
evidences = [r.evidence for r in records if r.evidence is not None]
by_id = {h.id: h for h in hearings}
print("registros:", len(records), "com evidência:", len(evidences))
print("tier x evidência nula:", Counter((r.tier, r.evidence is None) for r in records))
print("tier x provenance:", Counter((r.tier, r.provenance) for r in records))
print(
    "prefixo (palavras):",
    Counter((e.support_type, len(e.quote_prefix.split())) for e in evidences if e.quote_prefix),
)
print("score nulo:", Counter((e.support_type, e.score is None) for e in evidences))
scores = [e.score for e in evidences if e.score is not None]
print("score:", round(min(scores), 3), "a", round(max(scores), 3))
located = [
    (r, r.evidence) for r in records if r.evidence is not None and r.evidence.start_char is not None
]
print("offsets localizados:", len(located))
print(
    "texto diferente do trecho:",
    sum(
        " ".join(by_id[r.hearing_id].transcricao[e.start_char : e.end_char].split()) != e.text
        for r, e in located
    ),
)
```
