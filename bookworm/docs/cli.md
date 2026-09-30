# Referência da CLI

Todos os comandos da CLI `bookworm`, com o que cada um grava e a saída de `--help`. O uso comum está
no [README da biblioteca](../README.md); o formato dos arquivos gravados, em
[data_model.md](data_model.md).

| comando | para quê |
|---|---|
| [`build-udvs`](#build-udvs) | liga cada opinião do LDS a um trecho da transcrição e grava a rodada |
| [`verify-udvs`](#verify-udvs) | recalcula a rodada a partir do LDS e acusa qualquer registro divergente |
| [`export-hearing`](#export-hearing) | grava uma audiência, com as unidades candidatas, no JSON da demo |
| [`export-site`](#export-site) | grava todas as audiências e o índice da demo, lendo só o cache |
| [`build-splits`](#build-splits) | separa as audiências em treino, validação e teste pela data |
| [`verify-splits`](#verify-splits) | refaz datas, cortes e atribuição e confere o manifesto |
| [`filter-actor-speeches`](#filter-actor-speeches) | restringe as falas por ator às audiências de alguns splits |
| [`generate-profiles`](#generate-profiles) | escreve um perfil por ator com um modelo de linguagem |
| [`validate-profiles`](#validate-profiles) | confere cada perfil contra as UDVs do próprio ator |
| [`sample-profile-review`](#sample-profile-review) | sorteia pares para revisão humana, com o julgamento vazio |
| [`score-profile-review`](#score-profile-review) | calcula as proporções da revisão preenchida, com Wilson |

Os códigos de saída estão no [fim da página](#códigos-de-saída).

## Visão geral

Sem `--config`, os comandos de UDV e de exportação procuram `configs/udv.toml` e os de split,
`configs/splits.toml`; os comandos de perfil exigem `--config`. `uv run bookworm --version` imprime a
versão.

```text
 Usage: bookworm [OPTIONS] COMMAND [ARGS]...

 Build and verify evidence units (UDVs), temporal splits and actor profiles of the PublicHearingBR
 hearings.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --version          Show the version and exit.                                                    │
│ --help             Show this message and exit.                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────╮
│ build-udvs             Build UDV records (opinion to transcript evidence) from the LDS file.     │
│ verify-udvs            Recompute and cross-check a UDV run against the LDS file.                 │
│ export-hearing         Write one hearing of a UDV run, with ranked candidate units, as demo      │
│                        JSON.                                                                     │
│ export-site            Write every hearing of a UDV run as demo JSON, plus an index.json, read   │
│                        only from the embedding cache.                                            │
│ build-splits           Build the temporal split manifest and report from the LDS file.           │
│ verify-splits          Recompute and cross-check a split manifest against the LDS file.          │
│ filter-actor-speeches  Keep only the hearings of the configured splits in the actor speeches     │
│                        file.                                                                     │
│ generate-profiles      Write one LLM-written profile per actor from an actor speeches file.      │
│ validate-profiles      Score each verified UDV against the profile of its actor and of every     │
│                        other actor, separating hearings seen at generation from held-out ones.   │
│ sample-profile-review  Draw a seeded, stratified sample of profile pairs as a CSV with empty     │
│                        judgments.                                                                │
│ score-profile-review   Compute support proportions with Wilson intervals from a filled review    │
│                        CSV.                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## `build-udvs`

Grava `<output_dir>/<run-name>.jsonl` e `<output_dir>/<run-name>_coverage.json`, e guarda
os embeddings em `cache_dir`, com o mesmo nome de arquivo dos scripts de referência (o cache existente
continua válido). `--ids` pode ser repetido; quando presente, `--limit` é ignorado. Com
`--actors-config`, grava também as falas por ator e as ligações UDV → ator (ver
[Atores e perfis](../README.md#atores-e-perfis-a-mesma-passada-das-udvs)).

<details>
<summary><code>uv run bookworm build-udvs --help</code></summary>

```text
 Usage: bookworm build-udvs [OPTIONS]

 Build UDV records (opinion to transcript evidence) from the LDS file.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --run-name             <str>               Basename of the output files. [required]           │
│    --config               <path>              UDV TOML config. [default: configs/udv.toml]       │
│    --limit                <int range> [x>=1]  Only the first N LDS records.                      │
│    --ids                  <int>               Only these hearing ids; repeat the option for      │
│                                               more.                                              │
│    --overwrite                                Replace the files of an existing run.              │
│    --actors-config        <path>              Actor speeches TOML config; also writes per-actor  │
│                                               speeches and the UDV to actor links from the same  │
│                                               pass over the transcripts.                         │
│    --help                                     Show this message and exit.                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

</details>

## `verify-udvs`

Recalcula tudo a partir do LDS, compara com os registros e com o arquivo de cobertura e
imprime um relatório JSON; `--baseline` acrescenta a diferença contra uma execução anterior.

<details>
<summary><code>uv run bookworm verify-udvs --help</code></summary>

```text
 Usage: bookworm verify-udvs [OPTIONS]

 Recompute and cross-check a UDV run against the LDS file.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --run-name        <str>   Basename of the run files. [required]                               │
│    --config          <path>  UDV TOML config. [default: configs/udv.toml]                        │
│    --baseline        <path>  Previous <run>.jsonl to diff against.                               │
│    --help                    Show this message and exit.                                         │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

</details>

## `export-hearing`

Grava, em `--output`, uma audiência de uma execução no JSON lido pela demonstração
web: transcrição, turnos com as sentenças e os offsets de cada uma, participantes, as UDV da
execução e, para cada opinião, as `--top-k` (padrão 8) unidades candidatas do participante mais
similares a ela, na unidade da configuração (sentenças em `udv_v1`, janelas de duas sentenças em
`udv_v2`), com os mesmos textos e rótulos de cache que `build-udvs` usou. Os embeddings vêm só do cache que `build-udvs` gravou para a execução
([ADR 0004](adr/0004-exports-read-only-the-run-cache.md)): o comando não carrega modelo, não
importa `torch` e não grava nada no cache, e um embedding ausente do cache faz o comando parar com
código 2, com o caminho do arquivo esperado. Com `--split-manifest`, o conjunto da audiência no
manifesto entra em `hearing.split`. O comando também para com código 2 se a execução foi construída
com outro encoder ou outra revisão, se o pipeline gravado na cobertura (unidade semântica, extensão da
citação, política de citação) ou o corte `embedding_threshold` diferem dos de `--config`, se a
audiência não pertence à execução ou se os registros da audiência não correspondem às opiniões do LDS. O formato está em
[data_model.md](data_model.md#json-de-demonstração-export-hearing).

<details>
<summary><code>uv run bookworm export-hearing --help</code></summary>

```text
 Usage: bookworm export-hearing [OPTIONS]

 Write one hearing of a UDV run, with ranked candidate units, as demo JSON.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --run-name               <str>               Basename of the run files. [required]            │
│ *  --hearing                <int>               Hearing id to export. [required]                 │
│ *  --output                 <path>              Path of the JSON to write. [required]            │
│    --config                 <path>              UDV TOML config. [default: configs/udv.toml]     │
│    --top-k                  <int range> [x>=1]  Candidate units kept per opinion. [default: 8]   │
│    --split-manifest         <path>              Split manifest that names the hearing split.     │
│    --verifier-report        <path>              Verifier report of the run; adds the verifier,   │
│                                                 question and translation signals of each UDV.    │
│    --help                                       Show this message and exit.                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

</details>

## `export-site`

Grava os dados da demonstração para todas as audiências de uma execução numa só
passada: `hearings/<id>.json`, que é exatamente o arquivo de `export-hearing` daquela audiência
(mesmo código, mesmo `--top-k` padrão, mesmo manifesto), e `index.json`, com um resumo de cada
audiência para a tela que lista as matérias. O LDS é lido uma vez, e os embeddings vêm só do cache,
como em `export-hearing`. Sem `--output`, o destino é `web/app/data` do projeto `bookworm` de onde o
pacote foi instalado em modo editável; fora dessa árvore de código, `--output` é obrigatório.
`bookworm/web/app/data/` é versionado com a exportação de `udv_v2`, que contém as transcrições inteiras, para que a demo rode sem o dataset. O
comando recusa um destino que já tenha `index.json` ou `hearings/`, a não ser com `--overwrite`, que
apaga o `index.json` antigo antes de gravar a primeira audiência e regrava os arquivos das audiências
da execução; arquivos de outras audiências não são apagados e ficam fora do índice. `index.json` é
gravado por último, então uma exportação interrompida, nova ou com `--overwrite`, deixa o diretório
sem índice. Nenhum arquivo leva data de criação, e duas exportações com as mesmas entradas geram os
mesmos bytes. A exportação de `udv_v2` com `--verifier-report`, `--profiles` e `--human-validation` grava
58.810.449 bytes no total, com 264 perfis de ator (`experiments/artifacts/web/export_site_udv_v2.json`).
Para `udv_v1`, sem perfis, são 206 arquivos de audiência com 46.836.254 bytes no total (mediana de
199.367; o maior, 2.388.858, é o da audiência 6, cuja transcrição tem 147.728 palavras) e um índice de
184.112 bytes. O formato está em
[data_model.md](data_model.md#diretório-de-demonstração-export-site), e a página que lê
esses arquivos, com o comando para servi-la, está descrita em [web/README.md](../web/README.md).

<details>
<summary><code>uv run bookworm export-site --help</code></summary>

```text
 Usage: bookworm export-site [OPTIONS]

 Write every hearing of a UDV run as demo JSON, plus an index.json, read only from the embedding
 cache.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --run-name                <str>               Basename of the run files. [required]           │
│    --output                  <path>              Directory to write; defaults to web/app/data of │
│                                                  the bookworm source tree.                       │
│    --config                  <path>              UDV TOML config. [default: configs/udv.toml]    │
│    --top-k                   <int range> [x>=1]  Candidate units kept per opinion. [default: 8]  │
│    --split-manifest          <path>              Split manifest that names each hearing split.   │
│    --overwrite                                   Replace the files of an existing export.        │
│    --verifier-report         <path>              Verifier report of the run; adds the verifier,  │
│                                                  question and translation signals of each UDV.   │
│    --profiles                <path>              Actor profiles JSONL; adds actors.json and one  │
│                                                  profiles/<actor>.json per profiled actor.       │
│    --actors-config           <path>              Hearing actors TOML config, used with           │
│                                                  --profiles to rebuild the speeches.             │
│                                                  [default: configs/hearing_actors.toml]          │
│    --profiles-run            <str>               Name of the profile run shown on the page;      │
│                                                  defaults to the file name.                      │
│    --human-validation        <path>              Final precision report of the run; adds the     │
│                                                  human judgments, the precision per tier and the │
│                                                  judgments per verifier band to index.json.      │
│    --help                                        Show this message and exit.                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

</details>

## `build-splits`

Grava `<output_dir>/<split_version>.json` (manifesto) e
`<output_dir>/<split_version>_report.json` (relatório) e imprime um resumo por conjunto. Se `udv_path`
existir, as UDVs entram nas contagens do relatório; se não existir, `udv_source` fica `null`.

<details>
<summary><code>uv run bookworm build-splits --help</code></summary>

```text
 Usage: bookworm build-splits [OPTIONS]

 Build the temporal split manifest and report from the LDS file.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --config        <path>  Split TOML config. [default: configs/splits.toml]                        │
│ --help                  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

</details>

## `verify-splits`

Refaz a extração de datas, os cortes e a atribuição a partir do LDS e confere o
manifesto e o relatório gravados.

<details>
<summary><code>uv run bookworm verify-splits --help</code></summary>

```text
 Usage: bookworm verify-splits [OPTIONS]

 Recompute and cross-check a split manifest against the LDS file.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --config        <path>  Split TOML config. [default: configs/splits.toml]                        │
│ --help                  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

</details>

## `filter-actor-speeches`

Lê o arquivo de falas por ator gravado por `build-udvs --actors-config` e mantém só as audiências
dos splits de `[split_filter].splits`, conferindo que o manifesto foi construído a partir do mesmo LDS.
Grava o arquivo filtrado e um arquivo de estatísticas, e conta as UDVs das audiências de
`eval_splits` ligadas a um ator com perfil. Detalhes em [profiles.md](profiles.md).

<details>
<summary><code>uv run bookworm filter-actor-speeches --help</code></summary>

```text
 Usage: bookworm filter-actor-speeches [OPTIONS]

 Keep only the hearings of the configured splits in the actor speeches file.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --config        <path>  Actor profiles TOML config. [required]                                │
│    --output        <path>  Filtered speeches JSONL (overrides config).                           │
│    --help                  Show this message and exit.                                           │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

</details>

## `generate-profiles`

Escreve um perfil por ator a partir do arquivo de falas filtrado, chamando um `ChatClient` (a
implementação com `transformers` precisa do extra `profiles`). Atores que já têm perfil com a mesma
versão de prompt e o mesmo modelo são pulados, então uma execução interrompida continua de onde
parou. `--dry-run` renderiza todos os prompts sem carregar modelo. Detalhes em
[profiles.md](profiles.md).

<details>
<summary><code>uv run bookworm generate-profiles --help</code></summary>

```text
 Usage: bookworm generate-profiles [OPTIONS]

 Write one LLM-written profile per actor from an actor speeches file.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --config                  <path>              Actor profiles TOML config. [required]          │
│    --speeches,--input        <path>              Actor speeches JSONL (overrides config).        │
│    --output                  <path>              Profiles JSONL (overrides config).              │
│    --model                   <str>               Hugging Face model id or local path (overrides  │
│                                                  config).                                        │
│    --actors                  <str>               Only these actors, by exact name; repeat for    │
│                                                  more.                                           │
│    --limit                   <int range> [x>=0]  Process at most N actors this run.              │
│    --dry-run                                     Render every prompt and report sizes, without a │
│                                                  model.                                          │
│    --help                                        Show this message and exit.                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

</details>

## `validate-profiles`

Para cada UDV verificada ligada a um ator com perfil, mede a sentença do perfil mais próxima da
opinião e a posição do perfil do próprio ator entre todos os perfis, separando as audiências que
entraram no prompt das que ficaram de fora. Detalhes em
[profile_validation.md](profile_validation.md).

<details>
<summary><code>uv run bookworm validate-profiles --help</code></summary>

```text
 Usage: bookworm validate-profiles [OPTIONS]

 Score each verified UDV against the profile of its actor and of every other actor, separating
 hearings seen at generation from held-out ones.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --config           <path>  Profile validation TOML config. [required]                         │
│    --overwrite                Replace existing pairs and report files.                           │
│    --help                     Show this message and exit.                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

</details>

## `sample-profile-review`

Sorteia, com semente fixa, pares estratificados por grupo e por faixa de score e grava um CSV com o
campo de julgamento vazio, para preenchimento à mão.

<details>
<summary><code>uv run bookworm sample-profile-review --help</code></summary>

```text
 Usage: bookworm sample-profile-review [OPTIONS]

 Draw a seeded, stratified sample of profile pairs as a CSV with empty judgments.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --config           <path>  Profile validation TOML config. [required]                         │
│    --overwrite                Replace an existing review sample.                                 │
│    --help                     Show this message and exit.                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

</details>

## `score-profile-review`

Lê o CSV preenchido e calcula a proporção de cada julgamento por grupo, com intervalos de Wilson.

<details>
<summary><code>uv run bookworm score-profile-review --help</code></summary>

```text
 Usage: bookworm score-profile-review [OPTIONS]

 Compute support proportions with Wilson intervals from a filled review CSV.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --config             <path>  Profile validation TOML config. [required]                       │
│ *  --annotations        <path>  Review CSV with the judgments filled in. [required]              │
│    --help                       Show this message and exit.                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

</details>

## Códigos de saída

Iguais em todos os comandos:

- `0`: nenhum problema.
- `1`: a verificação encontrou problemas, listados em `problems`, ou a geração de perfis terminou com
  falhas.
- `2`: erro de entrada: hash do LDS divergente (`DatasetIntegrityError`), TOML ausente ou inválido
  (`ConfigError`), arquivo de execução, manifesto ou relatório ausente ou ilegível, seleção de audiências
  vazia, audiência ou encoder que não correspondem à execução exportada, embedding ausente do cache
  numa exportação (`EmbeddingCacheMissError`), destino de `export-site` já exportado sem
  `--overwrite`, arquivo registrado no relatório do verificador ausente, matéria sem carimbo de
  publicação ou falta de corte elegível (`SplitError`). A mensagem sai em `stderr`, sem traceback.

