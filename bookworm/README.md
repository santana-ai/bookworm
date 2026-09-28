# bookworm

Uma audiência pública da Câmara dos Deputados gera uma transcrição de dezenas de milhares de palavras,
e a matéria jornalística que a resume atribui opiniões aos participantes sem dizer onde, na fala, cada
opinião se apoia. `bookworm` liga cada opinião estruturada do dataset
[PublicHearingBR](https://huggingface.co/datasets/unicamp-dl/PublicHearingBR) a um trecho verificável da
transcrição, com offsets e um nível de confiança explícito, e constrói splits temporais sem vazamento
para avaliar métodos sobre essas ligações. Cada ligação é uma UDV (unidade de evidência): a opinião, o
participante, a sentença da transcrição que a sustenta, a posição dessa sentença e o critério que a
escolheu.

A biblioteca é o porte, com paridade testada, dos scripts de UDV e de splits de `challenge/utils/`
(`udv_pipeline.py`, `build_udvs.py`, `verify_udvs.py`, `dataset_io.py`, `hearing_dates.py`,
`build_splits.py` e `verify_splits.py`), na versão que segmenta as sentenças dentro de cada turno de
fala ([ADR 0002](docs/adr/0002-per-turn-sentence-segmentation.md)) e aceita trechos entre aspas
simples como citação ([ADR 0003](docs/adr/0003-single-quoted-spans-as-quotes.md)). Os demais scripts
de `challenge/utils/` (download, medições, benchmarks, calibração e experimentos) não foram portados.
A paridade com os artefatos em `challenge/artifacts/` é testada assim (ver
[ADR 0001](docs/adr/0001-library-scaffold-and-parity.md)):

- `udv_v1.jsonl` (corte 0,45) e `udv_v1_pre.jsonl` (mesmo código, corte 0,47): reconstruídos byte a
  byte só a partir do cache de embeddings existente (`BOOKWORM_EMBEDDING_CACHE`); sem o cache, os
  testes conferem registro a registro tudo o que não depende do encoder, a igualdade byte a byte depois
  de ler e regravar o arquivo e que `verify-udvs` não acusa nenhum problema;
- `udv_v1_coverage.json` e `udv_v1_pre_coverage.json`: iguais nas seções recontadas (`hearings`,
  `people`, `opinions`, `evidence_offsets`, `evidence_support_types`) e em `pipeline`, porque
  `created_at`, `timing`, `encoder_runtime` e `environment` dependem da execução e da máquina;
- `temporal_v1.json`: igual byte a byte, exceto `created_at`;
- `temporal_v1_report.json`: igual byte a byte, exceto `created_at` e `environment`.

`udv_v0.jsonl` e `dev20.jsonl` foram gerados pelo código anterior às ADR 0002 e 0003 e ficam como
registro histórico: um teste fixa a lista exata de registros que a verificação atual acusa em `udv_v0`,
que são os que as duas decisões mudam. Os arquivos `case_insensitive_*`, `quote_patterns_*` e
`turn_segmentation_*` de `challenge/artifacts/udv/` vêm de scripts de medição de `challenge/utils/` e
não são produzidos pela biblioteca. O formato de todos os arquivos lidos e gravados está em
[docs/data_model.md](docs/data_model.md).

## Instalação

Requer Python 3.12 e [uv](https://docs.astral.sh/uv/). Para desenvolver a própria biblioteca:

```bash
cd bookworm
uv sync
```

Para usar a partir de outro projeto `uv` (por exemplo `challenge/`), como dependência local editável:

```bash
uv add --editable ../bookworm
```

O núcleo depende só de `numpy`, `scikit-learn`, `pydantic` e `typer`, e roda em CPU: leitura do LDS,
construção e verificação de UDVs com o encoder TF-IDF, splits temporais e todas as verificações. O
extra `embeddings` acrescenta `sentence-transformers` e `torch`, necessários para construir UDVs com o
encoder Serafim, que é o que produziu os artefatos publicados:

```bash
uv sync --extra embeddings
uv add --editable ../bookworm --extra embeddings
```

O dispositivo `auto` escolhe `mps` em Apple Silicon, depois `cuda`, depois `cpu`.

## Dados

O dataset não é versionado. O arquivo usado é `PublicHearingBR_LDS.jsonl` (206 audiências), baixado do
Hugging Face. No repositório, o script `challenge/utils/download_dataset.py` faz isso:

```bash
cd challenge
uv run python -m utils.download_dataset
```

Fora do repositório, o mesmo download com `huggingface_hub` (que não é dependência da biblioteca):

```python
from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="unicamp-dl/PublicHearingBR",
    repo_type="dataset",
    local_dir=Path("dataset"),
    allow_patterns=["*.jsonl", "*.md", "*.py"],
)
```

O sha256 esperado do LDS é `c4e392ab95ce22f6228eace16f15e9efe1c846724846b873dec420aec3d872c0`. Toda
leitura pela CLI confere esse hash e para com `DatasetIntegrityError` (código de saída 2) se o arquivo
for outro, para que nenhum artefato seja gerado a partir de uma versão diferente do dataset sem aviso.

## Uso pela CLI num projeto consumidor

A configuração fica no projeto que usa a biblioteca, em TOML, e os caminhos relativos são resolvidos a
partir do diretório em que o comando roda. O layout esperado pelos TOML de `challenge/configs/` é:

```text
projeto/
  configs/udv.toml
  configs/splits.toml
  dataset/PublicHearingBR_LDS.jsonl
  artifacts/udv/            gravado por build-udvs
  artifacts/splits/         gravado por build-splits
  artifacts/cache/embeddings/
```

Comandos, rodados a partir de `projeto/`:

```bash
uv run bookworm --help
uv run bookworm build-udvs --config configs/udv.toml --run-name udv_v1
uv run bookworm build-udvs --config configs/udv.toml --run-name primeiras20 --limit 20
uv run bookworm build-udvs --config configs/udv.toml --run-name amostra --ids 3 --ids 5
uv run bookworm verify-udvs --config configs/udv.toml --run-name udv_v1
uv run bookworm verify-udvs --config configs/udv.toml --run-name udv_v1 --baseline artifacts/udv/udv_v0.jsonl
uv run bookworm export-hearing --config configs/udv.toml --run-name udv_v1 --hearing 70 \
    --output demo/hearing70.json --split-manifest artifacts/splits/temporal_v1.json
uv run bookworm export-site --config configs/udv.toml --run-name udv_v1 \
    --split-manifest artifacts/splits/temporal_v1.json
uv run bookworm build-splits --config configs/splits.toml
uv run bookworm verify-splits --config configs/splits.toml
```

Sem `--config`, os comandos procuram `configs/udv.toml` e `configs/splits.toml`. Sem `--run-name`,
`build-udvs` e `verify-udvs` usam `udv_v0`, como os scripts de referência; em `export-hearing` e
`export-site` o nome é obrigatório.

- `build-udvs` grava `<output_dir>/<run-name>.jsonl` e `<output_dir>/<run-name>_coverage.json`, e guarda
  os embeddings em `cache_dir`, com o mesmo nome de arquivo dos scripts de referência (o cache existente
  continua válido). `--ids` pode ser repetido; quando presente, `--limit` é ignorado.
- `verify-udvs` recalcula tudo a partir do LDS, compara com os registros e com o arquivo de cobertura e
  imprime um relatório JSON; `--baseline` acrescenta a diferença contra uma execução anterior.
- `export-hearing` grava, em `--output`, uma audiência de uma execução no JSON lido pela demonstração
  web: transcrição, turnos com as sentenças e os offsets de cada uma, participantes, as UDV da
  execução e, para cada opinião, as `--top-k` (padrão 8) sentenças do participante mais similares a
  ela. Os embeddings vêm só do cache que `build-udvs` gravou para a execução
  ([ADR 0004](docs/adr/0004-exports-read-only-the-run-cache.md)): o comando não carrega modelo, não
  importa `torch` e não grava nada no cache, e um embedding ausente do cache faz o comando parar com
  código 2, com o caminho do arquivo esperado. Com `--split-manifest`, o conjunto da audiência no
  manifesto entra em `hearing.split`. O comando também para com código 2 se a execução foi construída
  com outro encoder ou outra revisão, se a audiência não pertence à execução ou se os registros da
  audiência não correspondem às opiniões do LDS. O formato está em
  [docs/data_model.md](docs/data_model.md#json-de-demonstração-export-hearing).
- `export-site` grava os dados da demonstração para todas as audiências de uma execução numa só
  passada: `hearings/<id>.json`, que é exatamente o arquivo de `export-hearing` daquela audiência
  (mesmo código, mesmo `--top-k` padrão, mesmo manifesto), e `index.json`, com um resumo de cada
  audiência para a tela que lista as matérias. O LDS é lido uma vez, e os embeddings vêm só do cache,
  como em `export-hearing`. Sem `--output`, o destino é `web/app/data` do projeto `bookworm` de onde o
  pacote foi instalado em modo editável; fora dessa árvore de código, `--output` é obrigatório.
  `bookworm/web/app/data/` está no `.gitignore` porque os arquivos contêm as transcrições inteiras. O
  comando recusa um destino que já tenha `index.json` ou `hearings/`, a não ser com `--overwrite`, que
  apaga o `index.json` antigo antes de gravar a primeira audiência e regrava os arquivos das audiências
  da execução; arquivos de outras audiências não são apagados e ficam fora do índice. `index.json` é
  gravado por último, então uma exportação interrompida, nova ou com `--overwrite`, deixa o diretório
  sem índice. Nenhum arquivo leva data de criação, e duas exportações com as mesmas entradas geram os
  mesmos bytes. Para `udv_v1` são 206 arquivos de audiência com 76.526.309 bytes no total (mediana de
  329.899; o maior, 3.377.789, é o da audiência 6, cuja transcrição tem 147.728 palavras) e um índice
  de 184.112 bytes. O formato está em
  [docs/data_model.md](docs/data_model.md#diretório-de-demonstração-export-site), e a página que lê
  esses arquivos, com o comando para servi-la, está descrita em [web/README.md](web/README.md).
- `build-splits` grava `<output_dir>/<split_version>.json` (manifesto) e
  `<output_dir>/<split_version>_report.json` (relatório) e imprime um resumo por conjunto. Se `udv_path`
  existir, as UDVs entram nas contagens do relatório; se não existir, `udv_source` fica `null`.
- `verify-splits` refaz a extração de datas, os cortes e a atribuição a partir do LDS e confere o
  manifesto e o relatório gravados.

Códigos de saída, iguais em todos os comandos:

- `0`: nenhum problema.
- `1`: a verificação encontrou problemas, listados em `problems`.
- `2`: erro de entrada: hash do LDS divergente (`DatasetIntegrityError`), TOML ausente ou inválido
  (`ConfigError`), arquivo de execução, manifesto ou relatório ausente ou ilegível, seleção de audiências
  vazia, audiência ou encoder que não correspondem à execução exportada, embedding ausente do cache
  numa exportação (`EmbeddingCacheMissError`), destino de `export-site` já exportado sem
  `--overwrite`, matéria sem carimbo de publicação ou falta de corte elegível (`SplitError`).

### Atores e perfis: a mesma passada das UDVs

As UDVs e os perfis de atores saem dos mesmos turnos de fala. `build-udvs --actors-config` recorta a
transcrição de cada audiência uma vez e, na mesma execução, grava as UDVs, as falas por ator e a
ligação de cada UDV com o ator a que ela pertence
([ADR 0005](docs/adr/0005-one-transcript-pass-for-udvs-and-actor-profiles.md)). A política de atores,
o filtro por split e a geração de perfis reproduzem os scripts de `challenge/utils/`
(`build_actor_speeches.py`, `filter_actor_speeches.py` e `generate_actor_profiles.py`), com os
arquivos de saída testados byte a byte contra eles. A sequência completa, dentro de `challenge/`:

```bash
uv run bookworm build-udvs --config configs/udv.toml --run-name udv_v1 \
    --actors-config configs/hearing_actors.toml
uv run bookworm filter-actor-speeches --config configs/actor_profiles.toml
uv run bookworm generate-profiles --config configs/actor_profiles.toml \
    --input artifacts/cache/hearing_actors/actors_multi_hearing_train.jsonl --dry-run
uv sync --extra profiles
uv run bookworm generate-profiles --config configs/actor_profiles.toml \
    --input artifacts/cache/hearing_actors/actors_multi_hearing_train.jsonl \
    --output artifacts/actor_profiles/actor_profiles_train.jsonl --model <modelo>
uv run bookworm validate-profiles --config configs/profile_validation.toml
uv run bookworm sample-profile-review --config configs/profile_validation.toml
uv run bookworm score-profile-review --config configs/profile_validation.toml \
    --annotations artifacts/profile_validation/profile_review.csv
```

- O primeiro comando é o de sempre, com uma opção a mais: sem `--actors-config`, os arquivos de UDV
  são os mesmos de antes, e com ela a execução grava também os quatro arquivos de `[speeches]` e
  `<output_dir>/<run-name>_actor_links.jsonl`. Detalhes em [docs/actors.md](docs/actors.md).
- `--dry-run` renderiza todos os prompts sem carregar modelo; a geração de verdade precisa do extra
  `profiles` e de um modelo local ou do Hugging Face, atrás da interface `ChatClient`. Detalhes em
  [docs/profiles.md](docs/profiles.md).
- A conferência do perfil faz a pergunta que a UDV faz à matéria: dada uma opinião do ator com
  evidência na transcrição, há uma sentença do perfil que a sustente? Ela separa as audiências que
  entraram no prompt das que ficaram de fora (`test`), mede se o perfil do próprio ator fica em
  primeiro entre todos, e gera uma planilha cega para julgamento humano. Detalhes em
  [docs/profile_validation.md](docs/profile_validation.md).

### Configuração de UDV

Formato de `challenge/configs/udv.toml`:

```toml
[dataset]
lds_path = "dataset/PublicHearingBR_LDS.jsonl"
sha256 = "c4e392ab95ce22f6228eace16f15e9efe1c846724846b873dec420aec3d872c0"

[encoder]
name = "PORTULAN/serafim-335m-portuguese-pt-sentence-encoder"
revision = "a01887015444f7599669c509447c5bdbce958916"
batch_size = 64
device = "auto"

[evidence]
embedding_threshold = 0.45

[run]
seed = 42
output_dir = "artifacts/udv"
cache_dir = "artifacts/cache/embeddings"
```

Sem `kind`, `[encoder]` descreve um modelo `sentence-transformers`; com `kind = "tfidf"` (e
`max_features` opcional), o encoder TF-IDF é ajustado nas sentenças e opiniões das audiências
selecionadas. Chaves extras são ignoradas, e o TOML inteiro é copiado para o campo `config` do arquivo
de cobertura. O arquivo de `challenge/` tem também as chaves de registro da calibração do corte
(`calibration_source`, `calibration_method`, `threshold_decision`, `previous_embedding_threshold` e a
tabela `[calibration]`), que a biblioteca só copia.

Duas chaves opcionais de `[evidence]` ligam a construção de `udv_v2`
([ADR 0006](docs/adr/0006-udv-v2-windows-full-quotes-and-verifier.md)); sem elas, a biblioteca faz
exatamente o que fazia em `udv_v1`.

- `semantic_unit`: `"sentence"` (padrão), `"window2"` ou `"window3"`. Com janela, a unidade candidata
  da busca semântica são 2 ou 3 sentenças candidatas consecutivas do mesmo turno (passo 1); o encoder
  recebe as sentenças juntadas por espaço, e a evidência guarda o trecho da transcrição da primeira à
  última sentença. Os embeddings ficam no cache com o rótulo `window2s_<id>` ou `window3s_<id>`.
- `quote_extent`: `"prefix_sentence"` (padrão) ou `"full_quote"`. Com `full_quote`, a evidência de
  `quote_found` vai da sentença do prefixo até o fim da citação, achado por um sufixo de 6, 4 ou 3
  palavras a no máximo 2 vezes o tamanho da citação; sem sufixo, cobre tantas partes de sentença quantas
  a citação tem, sem sair do turno.

`verify-udvs` lê as duas chaves da configuração gravada na cobertura e confere cada evidência com a
mesma regra.

### Configuração de split

Formato de `challenge/configs/splits.toml`:

```toml
[dataset]
lds_path = "dataset/PublicHearingBR_LDS.jsonl"
sha256 = "c4e392ab95ce22f6228eace16f15e9efe1c846724846b873dec420aec3d872c0"

[temporal]
split_version = "temporal_v1"
train_fraction = 0.70
validation_fraction = 0.15
min_boundary_gap_days = 2

[leakage]
near_duplicate_threshold = 0.5
near_duplicate_pairs = 10

[run]
seed = 42
output_dir = "artifacts/splits"
udv_path = "artifacts/udv/udv_v0.jsonl"
```

`train_fraction` e `validation_fraction` precisam estar entre 0 e 1 e somar menos que 1;
`min_boundary_gap_days` precisa ser pelo menos 1; `split_version` vira nome de arquivo, então só aceita
letras, dígitos, `.`, `_` e `-`. O TOML inteiro é copiado para o campo `config` do relatório.

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

## Uso como biblioteca

```python
from pathlib import Path

from bookworm import (
    CachedEncoder,
    EvidenceSettings,
    TfidfEncoder,
    build_temporal_split,
    build_udvs,
    export_hearing,
    load_hearings,
    load_split_config,
    pipeline_description,
    verify_split_run,
)
from bookworm.udv.build import udv_corpus

hearings = load_hearings(
    Path("dataset/PublicHearingBR_LDS.jsonl"),
    expected_sha256="c4e392ab95ce22f6228eace16f15e9efe1c846724846b873dec420aec3d872c0",
)

encoder = TfidfEncoder.fit(udv_corpus(hearings[:20]), max_features=4096)
cached = CachedEncoder(encoder)
run = build_udvs(hearings[:20], cached, EvidenceSettings(embedding_threshold=0.45))
print(run.records[0].to_json_line())

first = [record for record in run.records if record.hearing_id == hearings[0].id]
demo = export_hearing(
    hearings[0], first, cached, run_name="tfidf", pipeline=pipeline_description(), top_k=3
)
print(demo["udvs"][0]["candidates"])

config = load_split_config(Path("configs/splits.toml"))
artifacts = build_temporal_split(hearings, config, run.records)
print({name: len(artifacts.manifest[name]) for name in ("train", "validation", "test")})
verification = verify_split_run(artifacts.manifest, artifacts.report, hearings, run.records, config)
print(verification.problems)
```

Os resultados de `build_temporal_split` não dependem do encoder; as UDVs só entram nas contagens do
relatório.

## Como uma UDV é construída

Para cada participante listado nos metadados da audiência, a biblioteca procura os turnos de fala dele
na transcrição. As sentenças candidatas do participante são as de cada um desses turnos, divididas
separadamente e concatenadas na ordem dos turnos; cada sentença guarda o índice do turno de onde veio,
e nenhuma atravessa a fronteira entre dois turnos. Uma sentença termina em `.`, `!` ou `?` seguidos de
espaço, ou num parêntese fechado cujo conteúdo termina em `.`, `!` ou `?` seguido de espaço (a
rubrica `(Palmas.)` fecha a sentença anterior), exceto a elisão `(...)`; partes com menos de 4
palavras e rubricas isoladas não viram candidatas.

Para cada opinião desse participante:

1. Participante sem nenhum turno encontrado: `person_not_resolved`, sem evidência.
2. Citações da opinião são os trechos entre `“…”`, `"…"` ou `'…'` com 10 ou mais caracteres (aspas
   simples só entre fronteiras de palavra, para que `d'água` não abra uma citação). Para cada citação,
   procura-se, dentro de cada turno do participante, o prefixo mais longo entre 10, 6, 4 e 3 palavras;
   vence a citação com mais palavras de prefixo encontradas, e o empate fica com a que aparece primeiro
   na opinião. Se esse prefixo tem 6 ou mais palavras, a evidência é a sentença, no turno da
   ocorrência, que contém o prefixo (`support_type = "direct_quote"`, `tier = "quote_found"`, sem
   score). Quando o prefixo ocorre mais de uma vez, vence a ocorrência cuja sentença tem a maior
   sobreposição de palavras (índice de Jaccard sobre tokens em minúsculas e sem acentos) com a opinião;
   o empate fica com a primeira.
3. Caso contrário, se o participante tem sentenças candidatas, a evidência é a de maior similaridade
   de cosseno com a opinião no espaço do encoder. O `tier` é `semantic_match_high` quando o score é
   maior ou igual a `embedding_threshold` e `semantic_match_weak` abaixo dele. Se o prefixo curto
   (menos de 6 palavras) encontrado, na primeira ocorrência, cai numa sentença que contém a escolhida
   ou está contida nela, o `support_type` é `semantic_with_short_quote` e o prefixo é registrado; senão
   é `semantic_similarity`.
4. Participante com turnos, mas sem sentença candidata (por exemplo, só uma indicação de cena como
   `(Manifestação em LIBRAS.)`): `no_evidence`.

Os offsets (`start_char`, `end_char`, `speaker_turn`) apontam para a transcrição original e são
procurados só no turno de onde a sentença veio. Em `udv_v1` as 2.105 evidências têm offsets.

O corte 0,45 de `challenge/configs/udv.toml` foi recalculado só com audiências do treino, pela regra
registrada em `[evidence].threshold_decision` desse arquivo (0,47 antes, com audiências que também
estão na validação e no teste). Ele separa `high` de `weak` e ainda não foi validado como nível de confiança. Com
TF-IDF, a separação não tem calibração e não deve ser interpretada; o TF-IDF serve como baseline em CPU
e para exercitar o pipeline sem modelo neural.

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

Com `challenge/configs/splits.toml`, o resultado é `temporal_v1`: 144 audiências e 1.536 UDVs no treino
(2021-11-18 a 2023-11-13), 32 e 308 na validação (2023-11-21 a 2023-12-20), 30 e 359 no teste
(2024-03-05 a 2024-05-09).

## Arquitetura do pacote

```text
src/bookworm/
  __init__.py            API pública (reexporta os nomes abaixo)
  cli.py                 aplicação Typer: build-udvs, verify-udvs, export-hearing, export-site,
                         build-splits, verify-splits
  config.py              UdvConfig e SplitConfig (pydantic, frozen), lidos de TOML
  errors.py              BookwormError, DatasetIntegrityError, ConfigError, SplitError,
                         EmbeddingCacheMissError
  data/
    io.py                JSONL/JSON em UTF-8, sha256, leitura do LDS com checagem de hash
    schemas.py           HearingRecord, Metadados, Envolvido (schema do LDS)
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
    evidence.py          evidência por citação ou por similaridade, e classificação de tier
    build.py             construção das UDVs de uma audiência e de uma execução
    coverage.py          arquivo de cobertura de uma execução, com a seção pipeline
    verify.py            verificação independente de uma execução e diferença contra outra
    export.py            JSON de uma audiência para a demonstração web (export-hearing)
    site.py              diretório da demonstração: um JSON por audiência e index.json (export-site)
  features/
    encoders.py          protocolo SentenceEncoder, cache em disco (CachedEncoder, com modo
                         cache_only) e RunCacheEncoder, que só dá nome ao cache de uma execução
    tfidf.py             TfidfEncoder, ajustado no corpus, em CPU
    sentence_transformer.py  encoder sentence-transformers (só com o extra embeddings)
```

Dependências entre módulos: `transcript` não conhece `udv`; `udv` usa `transcript` e `features`, e
`udv.export` usa também a data da matéria (`data.dates`) e os nomes dos conjuntos (`data.splits`);
`data.splits` usa `data.dates` e só lê `tier` e `hearing_id` das UDVs. `sentence_transformer.py` é o
único módulo que importa `torch` ou `sentence_transformers`, e a CLI só o importa quando a configuração
pede esse encoder; por isso `import bookworm` funciona sem o extra `embeddings`. A CLI recebe a fábrica
de encoders por `create_app(encoder_factory)`, o que permite testar os comandos sem carregar modelo.
`export-hearing` e `export-site` não usam essa fábrica: com um encoder `sentence-transformers`, montam
um `RunCacheEncoder` com `name` e `revision` da configuração e o dispositivo gravado em
`encoder_runtime.device` do arquivo de cobertura, que só dá nome aos arquivos do cache; com TF-IDF,
reajustam o encoder no corpus das audiências da execução, como `build-udvs`. `udv.site` usa
`udv.export` e não conhece a CLI.

## Testes e CI local

Rodado a partir de `bookworm/`:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
BOOKWORM_LDS_PATH=/caminho/para/challenge/dataset/PublicHearingBR_LDS.jsonl \
BOOKWORM_EMBEDDING_CACHE=/caminho/para/challenge/artifacts/cache/embeddings \
BOOKWORM_ARTIFACTS_DIR=/caminho/para/challenge/artifacts \
BOOKWORM_CHALLENGE_DIR=/caminho/para/challenge \
uv run pytest -m dataset
```

`mypy` roda em modo estrito sobre `src` e `tests`, e `pytest` mede cobertura por linha.

Marcadores e variáveis de ambiente:

- sem marcador: testes unitários e de integração com fixtures sintéticas escritas à mão
  (`tests/fixtures/`), sem rede, sem modelo e sem o dataset real. `lds_mini.jsonl` cobre os casos de
  resolução de participante e de evidência, inclusive um participante com dois turnos separados por
  `(Palmas.)`; `lds_turns.jsonl` cobre as regras de citação por turno (prefixo repetido em dois turnos,
  sentenças idênticas em dois turnos, segunda citação com prefixo mais longo que a primeira, citação
  entre aspas simples, prefixo curto repetido, rubrica colada a uma pergunta, elisão `(...)` e sentença
  contida noutra sentença de outro turno); `splits_mini.jsonl` tem 10 audiências com datas, menções ao
  dia da semana (inclusive uma que não confere) e um `Atualizado em`. `StubEncoder` (em
  `tests/conftest.py`) associa textos a vetores escolhidos à mão, o que permite fixar casos de fronteira
  como score exatamente igual ao limiar.
- `dataset`: usa o LDS real indicado por `BOOKWORM_LDS_PATH`; o teste é pulado quando a variável não
  está definida ou o arquivo não existe, e falha quando o sha256 não bate. `BOOKWORM_ARTIFACTS_DIR`
  aponta o diretório `artifacts/` com as execuções publicadas (padrão
  `<BOOKWORM_CHALLENGE_DIR>/artifacts`, ou `challenge/artifacts` do repositório quando
  `BOOKWORM_CHALLENGE_DIR` não está definida) e `BOOKWORM_CHALLENGE_DIR`, o projeto com os scripts de
  referência (padrão `challenge/` do repositório); um artefato ausente faz o teste correspondente ser
  pulado. O marcador inclui: os testes de deriva contra `udv_pipeline.py`, `hearing_dates.py` e
  `build_splits.py`, carregados pelo caminho do arquivo; a conferência, registro a registro, de tudo o
  que não depende do encoder em `udv_v1.jsonl` e `udv_v1_pre.jsonl` e a igualdade byte a byte desses
  dois arquivos depois de lidos e regravados; a verificação sem problemas dos dois e a lista exata de
  problemas de `udv_v0`; a paridade byte a byte com `temporal_v1.json` (exceto `created_at`) e
  `temporal_v1_report.json` (exceto `created_at` e `environment`); as propriedades do LDS citadas em
  [docs/data_model.md](docs/data_model.md); os cinco defeitos injetados no manifesto real (audiência
  trocada de conjunto, audiência removida, data alterada, fronteira alterada, contador do relatório
  alterado); uma execução TF-IDF nas 20 primeiras audiências que precisa passar em todas as checagens e
  ter as mesmas citações de `udv_v1`; a exportação da audiência 70 de `udv_v1`; e a exportação das 206
  audiências de `udv_v1` com `export_site` (contagens do índice, manchetes e tamanhos dos arquivos).
- `BOOKWORM_EMBEDDING_CACHE`: diretório de cache de embeddings já existente (por exemplo
  `challenge/artifacts/cache/embeddings`). Com ele, `tests/integration/test_parity_udv_cached_rebuild.py`
  reconstrói `udv_v1` e `udv_v1_pre`, `tests/integration/test_export_hearing.py` exporta a audiência 70
  e `tests/integration/test_export_site.py` exporta as 206 audiências, só a partir do cache: o encoder
  desses testes falha em qualquer texto que não esteja no cache, o cache é aberto em modo somente
  leitura e o teste confere que nenhum arquivo do diretório foi criado ou alterado.
- `model`: carrega o encoder Serafim; é opt-in e fica fora da execução padrão (`addopts` usa
  `-m 'not model'`). Rodar com `uv run pytest -m model`. Por padrão reconstrói as 20 primeiras
  audiências de `udv_v1`; `BOOKWORM_PARITY_FULL=1` reconstrói as 206. Ids, tiers, `support_type`,
  texto, offsets e `quote_prefix` precisam ser iguais, e os scores podem diferir em até `1e-5`. O
  teste não usa `BOOKWORM_EMBEDDING_CACHE`: todo embedding é calculado pelo modelo carregado, no
  dispositivo de `config.encoder.device` do arquivo de cobertura (`auto` em `udv_v1`), e nada é lido
  nem gravado no cache.

## Limitações conhecidas

- **Ocorrência de um prefixo repetido.** Um prefixo confiável que aparece mais de uma vez na fala do
  participante é resolvido pela sobreposição de palavras com a opinião; um prefixo curto usa sempre a
  primeira ocorrência, e mudar esse critério é uma questão separada (ADR 0002). Em `udv_v1`, 15
  registros com `quote_prefix` têm o prefixo repetido nos turnos do participante: 3 `direct_quote`
  (`udv-44-3-0` e `udv-192-0-1`, com as duas ocorrências no mesmo turno e a primeira escolhida, e
  `udv-65-1-0`, em que a segunda ocorrência vence) e 12 `semantic_with_short_quote`, fixados em
  `tests/integration/test_known_limitations.py`.
- **Citação com mais palavras de prefixo.** A regra prefere o prefixo mais longo mesmo quando a
  primeira citação já é confiável e tem sobreposição maior com a opinião; nas 206 audiências isso muda
  um registro, `udv-24-1-0` (ADR 0002).
- **Garantia de uma citação localizada.** `quote_found` garante que um prefixo de 6 a 10 palavras da
  citação aparece num turno da própria pessoa: as 10 primeiras palavras, ou a citação inteira quando
  ela tem de 6 a 9 palavras; se esse primeiro degrau não é encontrado, as 6 primeiras. As palavras da
  citação depois do prefixo encontrado não são conferidas. Em `udv_v1`, o prefixo de `direct_quote`
  tem 10 palavras em 122 registros, 6 em 152 e 7 em 3 (citações de exatamente 7 palavras encontradas
  inteiras). A validação humana das citações aceitas ainda não foi feita.
- **Prefixos curtos pouco distintivos.** A corroboração de `semantic_with_short_quote` aceita prefixos
  de 1 a 5 palavras; em `udv_v1` são 1 de 1 palavra, 7 de 2, 37 de 3, 65 de 4 e 1 de 5.
- **Posição de um texto repetido no turno.** O offset é a primeira posição do texto no turno de
  origem. Nas 115.599 sentenças candidatas, essa posição é anterior à da própria sentença em 85 casos:
  76 frases repetidas no turno e 9 em que o texto aparece antes, dentro de outra sentença (ADR 0002).
  Nas 2.105 evidências de `udv_v1`, o texto aparece uma única vez no turno de origem e o trecho gravado
  começa e termina em fronteiras de sentença desse turno, então nenhum desses casos ocorre (teste em
  `tests/integration/test_known_limitations.py`).
- **Turnos que o detector não separa.** A segmentação por turno depende dos cabeçalhos que
  `split_into_turns` reconhece; uma fala marcada como `(Não identificado)-` continua dentro do turno da
  pessoa anterior.
- **Encoder.** O Serafim tem `max_seq_length` de 128 tokens, então sentenças mais longas são truncadas
  antes de virar vetor. O lock da biblioteca resolve `sentence-transformers` 6.1.0 e `torch` 2.14.0,
  enquanto os artefatos foram produzidos com 5.6.1 e 2.13.0; recalcular embeddings pode dar scores
  ligeiramente diferentes, e o teste `model` aceita até `1e-5`.
- **Limiar.** O corte 0,45 separa `high` de `weak` e não foi validado como nível de confiança; com
  TF-IDF, a divisão não tem calibração.
- **Chave do cache de embeddings.** O separador `\x1e` entre textos não é escapado, então `["a", "b"]` e
  `["a\x1eb"]` geram a mesma chave. Mantido por compatibilidade com o cache existente.
- **Data do split.** A data usada é a de publicação da matéria. Em `temporal_v1`, 155 das 156 matérias
  que nomeiam o dia do debate confirmam a data com defasagem de 0 ou 1 dia; a audiência 111 cita um dia
  que não confere com o dia da semana. As fronteiras escolhidas têm 8 e 76 dias de intervalo.
- **O que o split temporal não isola.** 55 dos 879 nomes distintos aparecem em mais de um conjunto, 24
  deles em treino e teste; quem precisar medir generalização para participantes inéditos precisa de um
  split por ator. Os `tier` também não se distribuem igualmente: 80 das 90 opiniões de pessoa não
  resolvida estão no treino, e 6 das 8 sem evidência estão no teste.
- **Relatório do split.** `temporal_v1_report.json` conta as UDV de `udv_v0`, com que foi gerado; as
  contagens por `tier` desse relatório não refletem `udv_v1`.

## Ética e uso dos resultados

- Uma UDV liga uma opinião publicada a um trecho da transcrição por uma regra automática; ela não é
  anotação humana nem substitui a leitura da transcrição original. `provenance` registra como cada
  ligação foi feita (`weak` para casamento de citação, `model` para similaridade).
- Uma opinião sem evidência (`no_evidence`, `person_not_resolved` ou similaridade baixa) é uma opinião
  para a qual o método não encontrou trecho correspondente; a biblioteca não classifica essas opiniões
  como alucinação da matéria.
- A biblioteca descreve o que é observável nos textos; ela não infere intenção política de
  participantes, jornalistas ou veículos.
- Os rótulos NLI do dataset (arquivo que a biblioteca não usa) dizem se uma opinião é inferível a
  partir de até quatro trechos recuperados da fala; eles não cobrem a transcrição inteira.

## Citação

Os dados vêm do PublicHearingBR. Ao usar esta biblioteca ou seus artefatos, cite o artigo do dataset:

```bibtex
@misc{fernandes2024publichearingbrbrazilianportuguesedataset,
      title={PublicHearingBR: A Brazilian Portuguese Dataset of Public Hearing Transcripts for Summarization of Long Documents},
      author={Leandro Carísio Fernandes and Guilherme Zeferino Rodrigues Dobins and Roberto Lotufo and Jayr Alencar Pereira},
      year={2024},
      eprint={2410.07495},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.07495},
}
```
