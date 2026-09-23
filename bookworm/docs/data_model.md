# Modelo de dados

Este documento descreve, campo por campo, os arquivos que a biblioteca lê e grava: o arquivo LDS do
PublicHearingBR, o registro UDV, o arquivo de cobertura de uma execução de UDV, o JSON de
demonstração de uma audiência, o manifesto de split e o relatório de split. Os scripts de origem
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
`transcript`, `turns`, `people`, `udvs`, `run`.

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
| `udvs[]` | lista | Os registros UDV da audiência, lidos de `<run>.jsonl`, com os campos de `UdvRecord` na mesma ordem e três campos a mais. |
| `udvs[].candidates[]` | lista | Até `--top-k` (padrão 8) sentenças candidatas do participante, em ordem decrescente de similaridade de cosseno com a opinião, com o mesmo encoder e o mesmo cache de `build-udvs`; empates ficam na ordem das sentenças. Cada uma tem `text`, `score` (arredondado a 4 casas), `turn`, `start` e `end`. Vazia quando o participante não tem sentença candidata. |
| `udvs[].n_candidates` | inteiro | Número de sentenças candidatas do participante. |
| `udvs[].quotes` | lista de texto | Citações extraídas da opinião com os padrões da rodada, na ordem da opinião. |
| `run.name` | texto | Nome da execução. |
| `run.encoder`, `run.revision`, `run.threshold` | texto, texto, número | `method.encoder`, `method.revision` e `method.embedding_threshold` dos registros. |

Nas UDV semânticas, o primeiro candidato é a própria evidência: mesmo texto, turno e offsets, e
`score` igual ao da evidência arredondado a 4 casas. Em `quote_found` os candidatos continuam sendo os
da similaridade, e a evidência pode não estar entre eles. O comando para com código 2 quando os
registros da audiência na execução não são exatamente as opiniões do LDS, na mesma ordem, ou quando o
encoder da configuração tem nome ou revisão diferente de `method`. Para a audiência 70 de `udv_v1`
(teste `tests/integration/test_export_hearing.py`): 21 turnos, 6 participantes, 12 UDV, 10 evidências,
todas as posições conferidas contra a transcrição.

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
