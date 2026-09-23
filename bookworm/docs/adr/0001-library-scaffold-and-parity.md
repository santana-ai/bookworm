# ADR 0001: estrutura da biblioteca e porte com paridade

- Status: aceito
- Data: 2026-09-22
- Atualizado em 2026-09-23: porte da segmentação por turno ([ADR 0002](0002-per-turn-sentence-segmentation.md))
  e das aspas simples ([ADR 0003](0003-single-quoted-spans-as-quotes.md)); oráculos de paridade
  passam a ser `udv_v1` e `udv_v1_pre`; comando `export-hearing` (seção
  [Porte das ADR 0002 e 0003](#porte-das-adr-0002-e-0003))

## Contexto

O pipeline de UDV (unidade de evidência: opinião estruturada ligada a um trecho da transcrição) e o de
splits temporais existem hoje como scripts em `challenge/utils/` (`udv_pipeline.py`, `build_udvs.py`,
`verify_udvs.py`, `dataset_io.py`, `hearing_dates.py`, `build_splits.py`, `verify_splits.py`),
configurados por `challenge/configs/udv.toml` e `challenge/configs/splits.toml`. Esses scripts
produziram artefatos versionados (`challenge/artifacts/udv/udv_v0.jsonl`, `dev20.jsonl`, os
respectivos `*_coverage.json` e `challenge/artifacts/splits/temporal_v1*.json`) que já são citados em
análises.

Três problemas motivam a biblioteca:

1. Os scripts trocam dicionários sem tipo; um erro de chave ou de forma só aparece em tempo de
   execução, às vezes depois de minutos de encoder.
2. `load_gated_jsonl` encerra o processo com `SystemExit` quando o hash do dataset não bate, o que
   impede quem chama de tratar o erro.
3. Não há testes. Qualquer refatoração pode mudar silenciosamente os números dos artefatos já
   publicados.

## Decisão

- Projeto `uv` com layout `src/` (`uv init --lib`), backend `uv_build`, `requires-python = ">=3.12"`
  sem limite superior (para continuar compatível com `challenge/pyproject.toml`) e `.python-version`
  fixado em 3.12.
- Dependências do núcleo: `numpy`, `scikit-learn`, `pydantic>=2`, `typer`. Grupo `dev`: `ruff`, `mypy`,
  `pytest`, `pytest-cov`. Extra opcional `embeddings`: `sentence-transformers` e `torch`.
- O único módulo que poderá importar `sentence_transformers` ou `torch` é
  `bookworm/features/sentence_transformer.py`, e ele só será importado de forma tardia pela fábrica de
  encoders da CLI. Essa é a única exceção à regra de imports no topo do módulo: sem ela, instalar ou
  importar `bookworm` exigiria `torch`, e o baseline TF-IDF, que roda em CPU, deixaria de funcionar num
  ambiente enxuto.
- A configuração continua no projeto consumidor (arquivos TOML lidos com `tomllib`); a biblioteca
  recebe valores já lidos e não embute caminhos nem hashes.
- Erros de integridade do dataset viram `DatasetIntegrityError`; a CLI traduz essa exceção para código
  de saída 2.
- O schema do LDS é modelado com pydantic v2 (`extra="forbid"`, `frozen=True`, `strict=True`),
  mantendo as chaves originais em português (`transcricao`, `metadados`, `envolvidos`, `opinioes`).
- Porte com paridade antes de refatorar: regexes, constantes, ordem de iteração, uso de
  `sklearn.metrics.pairwise.cosine_similarity`, serialização com
  `json.dumps(record, ensure_ascii=False)` e o algoritmo de chave de cache de embeddings são copiados
  como estão. Um teste de deriva (`tests/integration/test_drift_challenge.py`, marcador `dataset`)
  roda as duas implementações nas 206 audiências e compara etapa por etapa (a lista atual está na
  seção [Porte das ADR 0002 e 0003](#porte-das-adr-0002-e-0003)). Um segundo teste confere as mesmas
  etapas contra os artefatos versionados. Só depois de zero divergências uma etapa pode ser
  refatorada, e a refatoração precisa manter os dois testes verdes.
- A cobertura de testes é medida por linha, com o core `sysmon` do coverage.py (`sys.monitoring`). No
  Python 3.12 esse core não mede ramos, e o tracer clássico, necessário para cobertura de ramos,
  multiplicava o tempo da execução `-m dataset`: 587 s contra 37 s nas mesmas 25 verificações, porque
  as cerca de 117 mil compilações de regex por implementação no teste de deriva passam por código
  Python da biblioteca padrão. A cobertura de ramos pode voltar quando o projeto adotar Python 3.14.
- Comportamentos conhecidos como incorretos também são portados como estão e ficam fixados por testes
  de caracterização até que a referência os corrija. Foi o caso de `udv-1-1-2`: a fala de um
  participante era a concatenação dos seus turnos, e como `(Palmas.)` termina em parêntese, a sentença
  final de um turno era unida à primeira sentença do turno seguinte da mesma pessoa, sem offset
  localizável. Um teste `xfail(strict=True)` registrava o comportamento desejado (nenhuma evidência
  atravessa turnos); com o porte da ADR 0002 ele passou a ser um teste comum, e o teste de
  caracterização do comportamento antigo foi removido.

### Construção, verificação e encoders de UDV

- Os registros UDV são modelos pydantic (`UdvRecord`, `Evidence`, `Actor`, `Method`) com a ordem de
  campos igual à ordem de chaves do JSONL e `tier`, `support_type` e `provenance` como `Literal`.
  `to_json_line` serializa com `json.dumps(model_dump(), ensure_ascii=False)` e `from_json_line` lê com
  `json.loads`, para que os floats passem pelo mesmo parser e formatador dos scripts. As 2.203 linhas
  de `udv_v0.jsonl` e as 268 de `dev20.jsonl` voltavam idênticas byte a byte depois de lidas e
  reescritas; hoje o teste faz o mesmo com as 2.203 linhas de `udv_v1.jsonl` e de `udv_v1_pre.jsonl`.
- A verificação trabalha sobre registros tipados. A checagem `unknown_support_type` da referência
  deixa de existir como checagem por registro, porque o schema rejeita um `support_type` desconhecido;
  linhas que não seguem o schema entram no relatório como `schema_invalid`, com o número da linha, e a
  verificação continua nas demais. Offsets parciais com `start_char` preenchido e `end_char` nulo, caso
  em que a referência levantava `TypeError`, viram `offset_shape`. As demais checagens mantêm a lógica e
  a ordem de inserção dos problemas da referência.
- Nova checagem `unlocated_evidence_crosses_turns` (substituída em 2026-09-23). Na referência de
  21/09, uma evidência com os três offsets nulos passava sem problema, e não havia como separar uma
  falha de localização de uma sentença que atravessa dois turnos. A checagem marcava a evidência sem
  offsets cujo texto não está contido em nenhum turno do participante; em `udv_v0` e em `dev20`, só
  `udv-1-1-2`. A ADR 0002 acrescentou à referência duas checagens que cobrem esse caso e o
  generalizam: `evidence_not_in_single_actor_turn` (o texto de toda evidência, com ou sem offsets,
  precisa caber num único turno do ator) e `evidence_offsets_missing` (toda evidência sem offsets é
  acusada). A biblioteca passou a usar as duas e removeu `unlocated_evidence_crosses_turns`, que
  marcaria só a interseção delas.
- `SentenceEncoder` é um protocolo (`name`, `revision`, `cache_identity`, `encode`, `runtime_info`).
  `CachedEncoder` usa a mesma chave de arquivo da referência,
  `f"{label}_{sha256(identity + cada texto seguido de \x1e)[:16]}.npy"`, com
  `identity = f"{name}@{revision}@{device}"` para o encoder `sentence-transformers`. Para os 206 pares
  `sentences_<id>`/`opinions_<id>` de `udv_v0`, os nomes calculados pela biblioteca coincidiam com os
  412 arquivos do cache existente, e uma reconstrução só a partir desse cache reproduzia `udv_v0.jsonl`
  e `dev20.jsonl` byte a byte, inclusive a divisão entre `semantic_match_high` e `semantic_match_weak`.
  Depois do porte da ADR 0002, a mesma reconstrução só a partir do cache reproduz `udv_v1.jsonl` e
  `udv_v1_pre.jsonl`.
  O separador `\x1e` não é escapado, então `["a", "b"]` e `["a\x1eb"]` geram a mesma chave; isso é
  mantido por paridade. O modo `read_only` permite apontar para um cache de outra pessoa sem escrever
  nele. O número de linhas dos embeddings é checado contra o número de textos, no cache e na saída do
  encoder.
- O modelo `sentence-transformers` é carregado só no primeiro `encode` ou `runtime_info`. Uma
  reconstrução em que todos os embeddings vêm do cache não carrega o modelo para codificar, mas a CLI
  ainda chama `runtime_info` para o arquivo de cobertura, como a referência fazia.
- O bloco `environment` do arquivo de cobertura mantém as chaves `python`, `torch`,
  `sentence_transformers` e `platform`, mas lê as versões dos metadados das distribuições instaladas
  (`importlib.metadata`) em vez de importar `torch`, para que execuções sem o extra `embeddings` também
  gerem cobertura. Sem o pacote instalado, a versão fica `null`.
- `TfidfEncoder` é um encoder do núcleo, ajustado nas sentenças e opiniões das audiências selecionadas,
  com saída densa em `float32` e `max_features` opcional; a identidade de cache inclui o sha256 do
  corpus e dos parâmetros. Ele existe para rodar o pipeline completo em CPU e como baseline. O limiar
  0,47 foi calibrado para o Serafim, então a divisão `high`/`weak` de uma execução TF-IDF não tem
  significado calibrado.
- A configuração é lida por `UdvConfig` (pydantic, `frozen`, `strict`), que guarda o TOML bruto em
  `source`. `[encoder]` sem `kind` descreve um modelo `sentence-transformers`, como no TOML atual de
  `challenge/`; `kind = "tfidf"` seleciona o TF-IDF. Chaves extras são ignoradas, como nos scripts, e
  caminhos relativos continuam relativos ao diretório de execução. Erros de configuração viram
  `ConfigError`, que a CLI traduz para código de saída 2, igual a `DatasetIntegrityError`; problemas de
  verificação dão código 1.
- A CLI (`build-udvs`, `verify-udvs`) é criada por `create_app(encoder_factory)`, o que permite aos
  testes injetar um encoder de teste sem carregar modelo.

### Splits temporais

- `data/dates.py` porta `hearing_dates.py` com as mesmas regexes e a mesma janela de 30 dias. Os
  retornos que eram dicionários viraram dataclasses congeladas (`WeekdayMention`, `MentionCheck`,
  `ArticleDateCheck`, `DateExtractionSummary`), com datas como `date`/`datetime`; `to_dict` usa
  `dataclasses.asdict` com uma fábrica que converte datas para ISO e tuplas para listas, o que preserva
  a ordem das chaves da referência. `summarize_date_extraction` saiu de `build_splits.py` para esse
  módulo, porque só depende das datas.
- `data/splits.py` porta `build_splits.py`: `CutCandidate` e `SplitBoundaries` são dataclasses, o
  manifesto e o relatório continuam dicionários JSON montados na ordem da referência, com o mesmo
  `TfidfVectorizer(min_df=2, sublinear_tf=True)` e o mesmo `cosine_similarity` na checagem de quase
  duplicatas. As UDVs entram como `UdvRecord`. Onde a referência encerrava o processo com `SystemExit`
  (matéria sem carimbo, nenhum corte elegível), a biblioteca levanta `SplitError`, que a CLI traduz
  para código 2. Manifesto e relatório de uma mesma construção recebem o mesmo `created_at`; na
  referência eram duas chamadas a `datetime.now`.
- `SplitConfig` segue o padrão de `UdvConfig` (pydantic, `frozen`, `strict`, TOML bruto em `source`) e
  acrescenta validações que a referência não tinha: frações entre 0 e 1 com soma menor que 1, intervalo
  mínimo de pelo menos 1 dia, `near_duplicate_pairs` não negativo e `split_version` restrito a um nome
  de arquivo simples, já que ele compõe o caminho de saída. A configuração real passa em todas.
- Resultado da paridade: com `challenge/configs/splits.toml`, o LDS real e `udv_v0.jsonl`,
  `build_temporal_split` produz `temporal_v1.json` idêntico byte a byte depois de igualar `created_at`, e
  `temporal_v1_report.json` idêntico depois de igualar `created_at` e `environment`. Um teste de deriva
  compara ainda, audiência por audiência, `check_article_date` com `hearing_dates.py`, e as datas, os
  cortes candidatos, a escolha de corte, a atribuição e o resumo de extração com `build_splits.py`.
- `data/verify_splits.py` porta `verify_splits.py` com `SplitConfig` tipado (a referência anotava o
  parâmetro como dicionário e o usava como objeto) e mantém as mensagens e a ordem dos problemas. Mudanças
  em relação à referência, todas em entradas que a faziam levantar exceção ou passar sem aviso:
  - a checagem de forma exige também `dataset` e `boundaries`, que a referência lia sem checar, marca
    `non_object:<chave>` quando `dataset`, `boundaries` ou `article_dates` não são objetos e interrompe
    as checagens seguintes quando encontra algum problema de forma;
  - ids desconhecidos são reportados por `unknown_hearing` e ignorados na cronologia e nas recontagens,
    em vez de levantar `KeyError`;
  - o relatório tem checagem de forma (`report_missing_key:<caminho>`) antes das recontagens;
  - nova checagem `assignment_mismatch`: cada audiência precisa estar no conjunto indicado pelas
    fronteiras recalculadas. Sem ela, mover a audiência da última data do treino para a validação e
    regravar o relatório passava na verificação, porque a cronologia continuava válida e as fronteiras
    gravadas continuavam iguais às recalculadas. Nos artefatos publicados ela não marca nada.
- Os cinco defeitos injetados usados para validar a verificação (audiência trocada de conjunto,
  audiência removida, data alterada, fronteira alterada, contador do relatório alterado) viraram testes:
  com a lista exata de problemas esperados na fixture sintética `splits_mini.jsonl`, e sobre o manifesto
  real no teste `dataset`.
- A CLI ganhou `build-splits` e `verify-splits`, com `--config` (padrão `configs/splits.toml`) e os
  mesmos códigos de saída dos comandos de UDV.

## Consequências

- Depois do porte das ADR 0002 e 0003 (descrito abaixo), a biblioteca reproduz, dentro do que os
  testes `dataset` comparam: `udv_v1.jsonl` e `udv_v1_pre.jsonl` byte a byte quando reconstruídos a
  partir do cache de embeddings (`BOOKWORM_EMBEDDING_CACHE`), `temporal_v1.json` byte a byte exceto
  `created_at`, `temporal_v1_report.json` exceto `created_at` e `environment`, e
  `udv_v1_coverage.json` e `udv_v1_pre_coverage.json` só nas seções recontadas e em `pipeline`
  (`created_at`, `timing`, `encoder_runtime` e `environment` dependem da execução). Uma mudança em
  qualquer número comparado aparece como falha de teste. `udv_v0.jsonl` e `dev20.jsonl` eram
  reconstruídos byte a byte a partir do cache até esse porte; o código atual não os reproduz mais, e
  eles ficam como registro histórico. Nenhum teste reconstrói os dois: `udv_v0` só é lido (lista fixa
  de problemas de verificação, comparação com `udv_v1_pre`, entrada dos testes de split e rodada
  recusada pela exportação), e `dev20` não é usado pelos testes. Os arquivos `case_insensitive_*` de
  `challenge/artifacts/udv/` e os scripts `download_dataset.py`, `generate_udv_manual_review.py` e
  `measure_case_insensitive_quotes.py` ficam fora do porte.
- A correção da segmentação de sentenças por turno mudou artefatos e foi decidida na ADR 0002 (e a
  inclusão das aspas simples na ADR 0003); o porte delas está descrito abaixo.
- Os testes de deriva importam `challenge/utils/udv_pipeline.py`, `hearing_dates.py` e
  `build_splits.py` pelo caminho do arquivo, a partir de `BOOKWORM_CHALLENGE_DIR` (ou de `challenge/`
  do repositório), então dependem da estrutura desse projeto; sem ele, são pulados.
- A biblioteca fica mais lenta de evoluir no curto prazo, porque cada etapa passa primeiro por paridade.
- O lock da biblioteca resolve `sentence-transformers` 6.1.0 e `torch` 2.14.0, enquanto os artefatos
  foram produzidos com 5.6.1 e 2.13.0 (registrado em `environment`). Recalcular embeddings com o
  modelo pode dar scores ligeiramente diferentes; o teste `model` aceita diferença de até `1e-5` no
  score e exige igualdade no resto. A versão anterior ao porte, sobre `dev20`, lia o cache quando
  `BOOKWORM_EMBEDDING_CACHE` estava definida; ela foi executada uma vez com a variável apontando para
  o cache existente e passou, mas, como todos os embeddings vieram do cache, não calculou nenhum
  embedding com as versões do lock. A versão atual, sobre as 20 primeiras audiências de `udv_v1`, não
  usa `BOOKWORM_EMBEDDING_CACHE`: o `CachedEncoder` do teste não tem diretório de cache, então todo
  embedding é calculado pelo modelo carregado e nenhum embedding é lido nem gravado em disco. Ela
  ainda não foi executada, e a pergunta sobre a tolerância continua em aberto até essa execução.

## Porte das ADR 0002 e 0003

Contexto: as ADR 0002 (segmentação de sentenças por turno e escolha da ocorrência da citação) e 0003
(trechos entre aspas simples contam como citação) mudaram `challenge/utils/udv_pipeline.py`,
`build_udvs.py` e `verify_udvs.py`, e a rodada completa refeita com esse código gerou
`challenge/artifacts/udv/udv_v1.jsonl` (corte 0,45) e `udv_v1_pre.jsonl` (mesmo código, corte 0,47).
Sem o porte, a biblioteca continuaria reproduzindo `udv_v0` e acusaria problemas nos artefatos novos.

Decisões:

- As duas ADR são copiadas sem alteração para `docs/adr/`, e o comportamento delas é portado como
  está: `SENTENCE_BOUNDARY_PATTERN` com a quebra depois de parêntese fechado e a exceção da elisão
  `(...)`; `split_turn_sentences` (sentenças de cada turno, com o índice do turno de origem);
  busca de citação dentro de cada turno (`find_prefix_occurrences`, `enclosing_turn_sentence`);
  todas as citações da opinião, com vitória do maior número de palavras de prefixo e empate para a
  que aparece primeiro; escolha da ocorrência de um prefixo confiável pela maior sobreposição de
  tokens (Jaccard), com empate para a primeira; primeira ocorrência para prefixos curtos;
  `SINGLE_QUOTE_PATTERN` ao lado do padrão de aspas duplas; offsets procurados só no turno de origem
  (`locate_turn_sentence_span`); seção `pipeline` no arquivo de cobertura.
- Os padrões de citação entram em `QuotePolicy.patterns` (padrão `QUOTE_PATTERNS`), para que uma
  medição possa isolar o efeito das aspas simples com `DOUBLE_QUOTE_PATTERNS`, como a referência faz
  com o argumento `patterns`. A seção `pipeline` é gerada a partir da política usada na rodada.
- `PersonSpeech` ganha `sentence_turns`, paralelo a `sentences`. As funções sobre a fala concatenada
  (`resolve_person_speech`, `split_sentences`, `find_quote_match`, `find_opinion_quote_match`,
  `enclosing_sentence`, `locate_sentence_span`) continuam exportadas, como na referência, mas a
  construção e a verificação não as usam para a evidência.
- A verificação porta as checagens atuais de `verify_udvs.py` com a mesma ordem:
  `evidence_not_in_single_actor_turn` antes das checagens de citação, `evidence_offsets_missing` para
  toda evidência com `start_char` nulo (junto com `offset_shape` quando `end_char` ou `speaker_turn`
  não são nulos), `quote_turn_mismatch` e `semantic_turn_mismatch` no fim; `quote_text_mismatch` e as
  checagens de prefixo curto passam a comparar com a sentença da ocorrência escolhida no turno. O caso
  `start_char` preenchido com `end_char` nulo continua virando `offset_shape`; na referência, ele
  levanta `TypeError` quando `speaker_turn` é um turno do ator.
- O teste de deriva compara, nas 206 audiências: turnos, turnos resolvidos, fala concatenada,
  sentenças por turno com o turno de origem, offsets de cada sentença no turno de origem, sentenças
  da fala concatenada, citações extraídas com os dois conjuntos de padrões, o casamento por turno
  completo (prefixo, palavras, citação, número e índice da ocorrência, turno, posição e sentença),
  confiança do casamento, offsets da sentença da citação, casamento e sentença na fala concatenada, e
  as constantes dos padrões. O módulo de referência é carregado pelo caminho do arquivo, com um nome
  próprio, para que duas cópias de `utils` (a do repositório e a de `BOOKWORM_CHALLENGE_DIR`) não se
  confundam em `sys.modules`.
- Oráculos de paridade: `udv_v1` e `udv_v1_pre`, lidos de `BOOKWORM_ARTIFACTS_DIR`. `udv_v0` fica como
  teste histórico: a verificação atual acusa nele exatamente os registros que as ADR 0002 e 0003
  mudam, e o teste fixa essa lista.
- Novo comando `export-hearing`, que grava uma audiência de uma rodada no JSON lido pela demonstração
  web: transcrição, turnos com sentenças e offsets, participantes, as UDV da rodada e, para cada
  opinião, as `top-k` sentenças mais similares do participante, calculadas com o mesmo encoder e o
  mesmo cache da construção. O formato está em [docs/data_model.md](../data_model.md). O comando
  recusa uma rodada construída com outro encoder (nome ou revisão diferente dos de `method`), porque
  os candidatos deixariam de corresponder à evidência gravada.

Resultado medido:

- Deriva: 0 divergências em todas as etapas, com 206 audiências, 1.065 participantes, 115.599
  sentenças candidatas (e os offsets de cada uma) e 2.203 opiniões.
- `udv_v1.jsonl` e `udv_v1_pre.jsonl` são reconstruídos byte a byte só a partir do cache de embeddings
  (encoder que falha em qualquer texto fora do cache, cache em modo somente leitura e sem nenhum
  arquivo novo ou alterado), e as seções recontadas e `pipeline` do arquivo de cobertura são iguais às
  gravadas. `verify-udvs` não acusa nenhum problema nos dois, e os contadores são os dos arquivos de
  cobertura: 277 `quote_found`, 1.785 `semantic_match_high`, 43 `semantic_match_weak`, 8
  `no_evidence` e 90 `person_not_resolved` em `udv_v1`; 277, 1.776, 52, 8 e 90 em `udv_v1_pre`.
- Em `udv_v0`, a verificação atual acusa a mesma lista registrada na ADR 0002: `udv-1-1-2` em
  `evidence_not_in_single_actor_turn` e `evidence_offsets_missing`; 4 registros em
  `evidence_not_person_sentence`; as 17 opiniões que passam a `quote_found` em
  `semantic_but_trusted_quote_findable`; 15 em cada checagem de prefixo curto; `udv-24-1-0` em
  `quote_prefix_mismatch`; 3 em `quote_text_mismatch`; `udv-65-1-0` em `quote_turn_mismatch`.
  `udv_v1_pre` difere de `udv_v0` em 42 evidências, com 17 mudanças de `semantic_match_high` para
  `quote_found`; `udv_v1` difere de `udv_v1_pre` só em 9 registros que passam de
  `semantic_match_weak` para `semantic_match_high` pelo corte.

Consequências:

- Os números de `udv_v0` citados acima nas seções anteriores descrevem o estado de 2026-09-22; os
  documentos de uso ([README](../../README.md) e [modelo de dados](../data_model.md)) passam a citar
  `udv_v1`.
- `udv_v0.jsonl` e `dev20.jsonl` não passam mais em `verify-udvs` (código 1), como esperado para
  artefatos do código anterior.
- O relatório do split `temporal_v1` continua contando as UDV de `udv_v0`, porque foi gerado com ele;
  regenerá-lo com `udv_v1` é uma decisão do projeto consumidor.

## Adiado

Ficam fora desta estrutura inicial, por não terem ainda implementação validada em `challenge/` que
possa ser portada com paridade: split por comissão, splits por similaridade, grafo de atores, espaço
latente, busca, comparação entre audiências, clusterização, API HTTP, módulo de logging, Makefile e
configuração em YAML. Também não entram módulos de marcação (`NotImplementedError`) para essas partes:
um módulo só é criado quando existe código executável e testado para ele.
