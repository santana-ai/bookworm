# Arquitetura da biblioteca

Como o pacote `bookworm` está organizado, como o split temporal é construído, o que as verificações
conferem e como a paridade com os scripts da versão de pesquisa é testada. As regras de construção de
uma UDV (turnos, resolução da pessoa, sentenças candidatas, citação, similaridade, nível e offsets)
estão descritas uma vez só, em [`docs/methodology/udv.md`](../../docs/methodology/udv.md#construção);
as chaves que ligam `udv_v2` estão em [configuration.md](configuration.md#configuração-de-udv).

- [Módulos](#módulos)
- [Como o split temporal é construído](#como-o-split-temporal-é-construído)
- [Verificações](#verificações)
- [Paridade com os scripts de referência](#paridade-com-os-scripts-de-referência)

## Módulos

O pacote segue o caminho dos dados: o LDS é lido e conferido (`data`), cada transcrição é recortada em
turnos e sentenças (`transcript`), as UDVs são construídas e verificadas (`udv`) com um encoder
(`features`), e a mesma passada alimenta as falas por ator (`actors`) e os perfis (`profiles`).

```text
src/bookworm/
  __init__.py            API pública (reexporta os pontos de entrada do README)
  cli.py                 aplicação Typer; create_app(encoder_factory, client_factory)
  config.py              UdvConfig e SplitConfig, lidos de TOML
  errors.py              BookwormError e subclasses; a CLI traduz todas para o código 2
  models.py              modelos pydantic base (congelados e estritos)
  pipeline.py            uma passada por transcrição: UDVs, falas por ator e ligações
  data/
    io.py                JSON/JSONL em UTF-8, sha256, leitura do LDS com checagem de hash
    schemas.py           HearingRecord, HearingMetadata, Participant (schema do LDS)
    dates.py             data de publicação da matéria e validação por menções ao dia da semana
    splits.py            cortes, atribuição, manifesto e relatório do split temporal
    verify_splits.py     verificação independente de um manifesto e de um relatório
  transcript/
    text.py              normalização de espaços, acentos e nomes
    turns.py             Turn e split_into_turns (cabeçalhos "O SR." / "A SRA.")
    speakers.py          resolução de participante para turnos de fala
    sentences.py         fronteira de sentença, sentenças por turno e sentença que contém um trecho
    offsets.py           Span e localização de uma sentença no turno de origem
  udv/
    schemas.py           UdvRecord, Evidence, Actor, Method, Tier, SupportType, Provenance
    quotes.py            extração de citações, busca de prefixo por turno e escolha da ocorrência
    windows.py           janelas de sentenças consecutivas de um turno (semantic_unit)
    evidence.py          evidência por citação ou por similaridade, e classificação de tier
    build.py             construção das UDVs de uma audiência e de uma execução
    coverage.py          arquivo de cobertura de uma execução, com a seção pipeline
    verify.py            verificação independente de uma execução e diferença contra outra
    export.py            JSON de uma audiência para a demonstração web (export-hearing)
    site.py              diretório da demonstração: um JSON por audiência e index.json (export-site)
    signals.py           sinais do verificador, das perguntas e da tradução, lidos do relatório
  features/
    encoders.py          protocolo SentenceEncoder, cache em disco (CachedEncoder, com modos
                         read_only e cache_only) e RunCacheEncoder, que só dá nome ao cache
    tfidf.py             TfidfEncoder, ajustado no corpus, em CPU
    loading.py           construção do encoder sentence-transformers, com import tardio
    sentence_transformer.py  encoder sentence-transformers (extra embeddings)
  actors/
    config.py            política de turnos, fusões de nomes e caminhos de saída
    schemas.py           registros das falas por ator e das ligações UDV → ator
    speeches.py          coleta dos turnos de pessoa em falas por ator
  profiles/
    config.py            configuração do filtro por split e da geração
    split_filter.py      falas por ator restritas às audiências dos splits configurados
    prompts.py, prompts/ templates Jinja2 e versão do prompt
    llm.py               protocolo ChatClient e checagens de cada resposta
    transformers_client.py  ChatClient com transformers (extra profiles)
    generate.py          geração retomável, com --dry-run
    schemas.py           registros de perfil
    validate.py          conferência do perfil contra as UDVs do ator
    review.py            amostra cega para julgamento humano e intervalos de Wilson
```

Dependências entre módulos: `transcript` não conhece `udv`; `udv` usa `transcript` e `features`, e
`udv.export` usa também a data da matéria (`data.dates`) e os nomes dos conjuntos (`data.splits`);
`data.splits` usa `data.dates` e só lê `tier` e `hearing_id` das UDVs; `actors` usa `transcript`, e
`profiles` lê os arquivos de `actors` e de `udv`. `sentence_transformer.py` e
`transformers_client.py` são os únicos módulos que importam `torch`, `sentence_transformers` ou
`transformers`, e só são importados quando a configuração pede esse encoder ou esse cliente; por isso
`import bookworm` funciona sem nenhum extra. A CLI recebe as fábricas de encoder e de cliente de chat
por `create_app`, o que permite testar os comandos sem carregar modelo. `export-hearing` e
`export-site` não usam a fábrica de encoder: com um encoder `sentence-transformers`, montam um
`RunCacheEncoder` com `name` e `revision` da configuração e o dispositivo gravado em
`encoder_runtime.device` do arquivo de cobertura, que só dá nome aos arquivos do cache
([ADR 0004](adr/0004-exports-read-only-the-run-cache.md)); com TF-IDF, reajustam o encoder no
corpus das audiências da execução, como `build-udvs`.


## Como o split temporal é construído

O LDS não tem campo de data. A data de cada audiência é a do primeiro carimbo `DD/MM/AAAA - HH:MM` da
matéria que não pertence a um `Atualizado em`, ou seja, a data de publicação da matéria. Como ela pode
diferir da data do debate, o relatório confere a extração contra o próprio texto: menções como
`nesta quarta-feira (17)` são resolvidas para uma data e comparadas com o dia da semana.

As datas distintas são ordenadas, e um corte só pode cair entre duas datas consecutivas separadas por
pelo menos `min_boundary_gap_days`. Entre esses candidatos, o corte do treino é o de fração acumulada de
audiências mais próxima de `train_fraction`, e o da validação, escolhido depois dele, o mais próximo de
`train_fraction + validation_fraction`; empates vão para o maior intervalo e depois para a data mais
antiga. Uma audiência vai para o treino se a data dela é até o primeiro corte, para a validação se é até
o segundo, e para o teste nos demais casos. Todas as audiências de uma mesma data ficam no mesmo
conjunto.

Com `experiments/configs/splits.toml`, o resultado é `temporal_v1`: 144 audiências e 1.536 UDVs no treino
(2021-11-18 a 2023-11-13), 32 e 308 na validação (2023-11-21 a 2023-12-20), 30 e 359 no teste
(2024-03-05 a 2024-05-09).

## Verificações

### Checagens de `verify-udvs`

Por registro, na ordem em que são avaliadas: `duplicate_id`, `missing_id`, `unexpected_id`,
`proposition_mismatch`, `actor_mismatch`, `method_threshold_mismatch`,
`person_not_resolved_but_matched`, `person_not_resolved_shape`, `resolved_tier_but_unmatched`,
`no_evidence_shape`, `evidence_missing`, `evidence_not_in_single_actor_turn`, `quote_shape`,
`quote_score_not_null`, `quote_not_trusted`, `quote_prefix_mismatch`, `quote_text_mismatch`,
`semantic_shape`, `semantic_but_trusted_quote_findable`, `evidence_not_person_sentence`,
`short_quote_support_mismatch`, `short_quote_prefix_mismatch`, `score_out_of_range`,
`tier_inconsistent_with_score`, `offset_shape`, `evidence_offsets_missing`, `offset_text_mismatch`,
`speaker_turn_not_actor`, `span_outside_turn`, `quote_turn_mismatch` e `semantic_turn_mismatch`. Por
execução: `coverage` (contadores do arquivo de cobertura recontados) e `schema_invalid` (linhas do
JSONL que não seguem o schema `UdvRecord`, com número da linha).

As checagens de citação refazem a busca atual: o casamento por turno, a citação vencedora e a
ocorrência escolhida. Por isso uma execução gerada pelo código anterior às ADR 0002 e 0003 é acusada
exatamente nos registros que essas decisões mudam:

- `evidence_not_in_single_actor_turn`: o texto da evidência, com espaços normalizados, não está
  dentro de nenhum turno do ator. É o sinal de uma sentença que junta o fim de um turno ao começo de
  outro.
- `evidence_offsets_missing`: evidência sem offsets. Com a segmentação por turno, toda sentença
  candidata e toda sentença de citação confiável é localizada no turno de origem, então um offset nulo
  só aparece por regressão.
- `quote_turn_mismatch` e `semantic_turn_mismatch`: o turno dos offsets não é o turno de onde a
  sentença veio (o da ocorrência escolhida, para citações; um dos turnos em que a sentença candidata
  aparece, para evidência semântica). Sem elas, offsets que apontam o mesmo texto noutro turno do ator
  passavam na verificação.

Em `udv_v1` e `udv_v1_pre` nenhuma checagem acusa nada. Em `udv_v0`, a verificação termina com
código 1 e acusa `udv-1-1-2` (`evidence_not_in_single_actor_turn`, `evidence_offsets_missing` e
`evidence_not_person_sentence`), mais 3 registros cujo texto deixou de ser sentença candidata, as 17
opiniões que passam a ter citação confiável, 15 registros em que a classificação do prefixo curto
muda (2 deles entre as 17), `udv-24-1-0` (outro prefixo), 3 registros com outra sentença de citação e
`udv-65-1-0` (outro turno).

### Checagens de `verify-splits`

Na ordem em que aparecem no relatório:

- forma do manifesto: `missing_key:<chave>`, `non_integer_ids:<conjunto>`, `non_object:<chave>`; com
  qualquer um deles, as demais checagens do manifesto não rodam;
- partição: `duplicate_hearing`, `unassigned_hearing`, `unknown_hearing`;
- datas: `article_dates_size`, `article_date_mismatch` (data gravada diferente da extraída agora);
- cronologia: `empty_split`, `not_chronological`, `boundary_gap_below_minimum`,
  `date_straddles_splits` (uma mesma data em dois conjuntos);
- fronteiras: `boundary_mismatch` (fronteira gravada diferente da recalculada) e `assignment_mismatch`
  (audiência num conjunto diferente do que as fronteiras recalculadas indicam);
- incerteza da data: `boundary_gap_within_date_uncertainty`, quando o intervalo de uma fronteira não é
  maior que a maior defasagem observada entre evento e publicação;
- relatório: `report_missing_key`, `report.<conjunto>.<contador>` (hearings, people, opinions, udvs
  recontados) e `report.udvs_total_mismatch`;
- execução: `seed_mismatch`, `dataset_sha256_mismatch`, `date_extraction_mismatch`.

## Paridade com os scripts de referência

A biblioteca é o porte, com paridade testada, dos scripts de UDV, de splits e de atores da versão de
pesquisa, `challenge/utils/` na tag `research-2026-09-28` (`udv_pipeline.py`, `build_udvs.py`,
`verify_udvs.py`, `dataset_io.py`, `hearing_dates.py`, `build_splits.py`, `verify_splits.py`,
`build_actor_speeches.py`, `filter_actor_speeches.py` e `generate_actor_profiles.py`; os caminhos
atuais estão em [`docs/path_map.md`](../../docs/path_map.md)), na versão que segmenta as sentenças dentro
de cada turno de fala ([ADR 0002](adr/0002-per-turn-sentence-segmentation.md)) e aceita trechos
entre aspas simples como citação ([ADR 0003](adr/0003-single-quoted-spans-as-quotes.md)). Os
demais módulos de `experiments/src/experiments/` (download, medições, benchmarks, calibração e
experimentos) não foram portados. A paridade com os artefatos em `experiments/artifacts/` é testada assim (ver
[ADR 0001](adr/0001-library-scaffold-and-parity.md)):

- `udv_v1.jsonl` (corte 0,45) e `udv_v1_pre.jsonl` (mesmo código, corte 0,47): reconstruídos byte a
  byte só a partir do cache de embeddings existente (`BOOKWORM_EMBEDDING_CACHE`); sem o cache, os
  testes conferem registro a registro tudo o que não depende do encoder, a igualdade byte a byte depois
  de ler e regravar o arquivo e que `verify-udvs` não acusa nenhum problema;
- `udv_v1_coverage.json` e `udv_v1_pre_coverage.json`: iguais nas seções recontadas (`hearings`,
  `people`, `opinions`, `evidence_offsets`, `evidence_support_types`) e em `pipeline`, porque
  `created_at`, `timing`, `encoder_runtime` e `environment` dependem da execução e da máquina;
- `temporal_v1.json`: igual byte a byte, exceto `created_at`;
- `temporal_v1_report.json`: igual byte a byte, exceto `created_at` e `environment`.
- falas por ator: os arquivos gravados por `build-udvs --actors-config` são iguais byte a byte aos do
  script de referência, e as contagens batem com `experiments/artifacts/hearing_actors/` e
  `experiments/artifacts/actor_profiles/train_speeches_stats.json`.

`udv_v0.jsonl` e `dev20.jsonl` foram gerados pelo código anterior às ADR 0002 e 0003 e ficam como
registro histórico: um teste fixa a lista exata de registros que a verificação atual acusa em `udv_v0`,
que são os que as duas decisões mudam. Os arquivos `case_insensitive_*`, `quote_patterns_*` e
`turn_segmentation_*` de `experiments/artifacts/udv/` vêm de `experiments.udv.measure_*` e
não são produzidos pela biblioteca. O formato de todos os arquivos lidos e gravados está em
[data_model.md](data_model.md).

