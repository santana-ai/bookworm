# bookworm

Biblioteca Python que liga cada opinião estruturada do dataset
[PublicHearingBR](https://huggingface.co/datasets/unicamp-dl/PublicHearingBR) a um trecho verificável
da transcrição da audiência, com offsets e um nível explícito que diz como a ligação foi feita.

A matéria jornalística atribui opiniões aos participantes sem dizer onde, na fala, cada opinião se
apoia, e a transcrição tem dezenas de milhares de palavras. Cada ligação que a biblioteca grava é uma
UDV (Unidade Deliberativa Verificável): a opinião, o participante, o trecho da transcrição, a posição
desse trecho e o critério que o escolheu. Na rodada de referência, `udv_v2`, o trecho é a citação
inteira, quando a opinião cita a fala, ou a janela de duas sentenças mais parecida; em `udv_v1`,
mantida como histórico, é uma sentença.

A biblioteca faz quatro coisas, todas conferidas por uma verificação independente que recalcula o
resultado a partir do dataset:

| o quê | comandos | problema que resolve |
|---|---|---|
| UDVs | `build-udvs`, `verify-udvs` | achar, nos turnos da própria pessoa, o trecho que corresponde à opinião: a citação, quando há, ou a unidade mais parecida no espaço de um encoder |
| Splits temporais | `build-splits`, `verify-splits` | separar treino, validação e teste pela data da matéria, para que nenhum método seja ajustado com audiências posteriores às que avalia |
| Atores e perfis | `build-udvs --actors-config`, `generate-profiles`, `validate-profiles` | juntar as falas de cada pessoa entre audiências, gerar um perfil por ator atrás de uma interface de LLM e conferir o perfil contra as UDVs do ator |
| Exportação da demo | `export-hearing`, `export-site` | gravar os JSON que a [página web](web/README.md) desenha, sem carregar modelo |

> [!NOTE]
> Uma UDV é uma ligação automática, não uma anotação humana, e não substitui a leitura da transcrição
> ([uso dos resultados](docs/limitations.md#uso-dos-resultados)).

## Instalação

Requer Python 3.12 e [uv](https://docs.astral.sh/uv/). O núcleo roda em CPU e depende de `numpy`,
`scikit-learn`, `pydantic`, `typer` e `jinja2`; dois extras acrescentam dependências pesadas, importadas
só quando a configuração pede.

```bash
cd bookworm && uv sync              # núcleo
uv sync --extra embeddings          # sentence-transformers e torch: o encoder Serafim dos artefatos publicados
uv sync --extra profiles            # transformers, accelerate e torch: gerar perfis com um modelo local
```

Em outro projeto `uv`, a biblioteca entra como dependência local editável com
`uv add --editable ../bookworm`; é assim que `experiments/` a usa. O dispositivo `auto` escolhe `mps` em
Apple Silicon, depois `cuda`, depois `cpu`.

O dataset não é versionado: `uv run python -m experiments.data.download`, dentro de `experiments/`, baixa
o `PublicHearingBR_LDS.jsonl` (206 audiências). Toda leitura pela CLI confere o sha256 do LDS e para com
`DatasetIntegrityError` (código de saída 2) se o arquivo for outro, para que nenhum artefato saia de uma
versão diferente do dataset sem aviso.

## Início rápido, em CPU

O exemplo usa o encoder TF-IDF, que roda em segundos sem GPU e sem extra. Ele não reproduz os artefatos
publicados (feitos com o Serafim), mas exercita o pipeline inteiro. Grave este `configs/udv_tfidf.toml`
dentro de `experiments/`, que já tem o dataset e os outros TOML:

```toml
[dataset]
lds_path = "dataset/PublicHearingBR_LDS.jsonl"
sha256 = "c4e392ab95ce22f6228eace16f15e9efe1c846724846b873dec420aec3d872c0"

[encoder]
kind = "tfidf"
max_features = 4096

[evidence]
embedding_threshold = 0.45

[run]
seed = 42
output_dir = "artifacts/udv"
cache_dir = "artifacts/cache/embeddings_tfidf"
```

```bash
uv run bookworm build-udvs --config configs/udv_tfidf.toml --run-name tfidf20 --limit 20
uv run bookworm verify-udvs --config configs/udv_tfidf.toml --run-name tfidf20
uv run bookworm export-hearing --config configs/udv_tfidf.toml --run-name tfidf20 --hearing 3 \
    --output demo/hearing3.json --split-manifest artifacts/splits/temporal_v1.json
```

Nas 20 primeiras audiências, `build-udvs` grava 268 registros em menos de um segundo (40
`quote_found`, 85 `semantic_match_high`, 130 `semantic_match_weak`, 13 `person_not_resolved`), e
`verify-udvs` termina com `"problems": {}` e código 0. Com o TF-IDF, a separação entre
`semantic_match_high` e `semantic_match_weak` não tem calibração e não deve ser interpretada. A rodada
publicada se confere sem modelo:

```bash
uv run bookworm verify-udvs --config configs/udv_v2.toml --run-name udv_v2
```

### Atores e perfis: a mesma passada das UDVs

`build-udvs --actors-config` recorta a transcrição de cada audiência uma vez e grava, na mesma
execução, as UDVs, as falas por ator e a ligação de cada UDV com o seu ator
([ADR 0005](docs/adr/0005-one-transcript-pass-for-udvs-and-actor-profiles.md)). Como as fusões de nomes
de `configs/hearing_actors.toml` foram revisadas nas 206 audiências, essa opção exige a execução
completa.

```bash
uv run bookworm build-udvs --config configs/udv_tfidf.toml --run-name tfidf \
    --actors-config configs/hearing_actors.toml
uv run bookworm filter-actor-speeches --config configs/actor_profiles.toml
uv run bookworm generate-profiles --config configs/actor_profiles.toml \
    --input artifacts/cache/hearing_actors/actors_multi_hearing_train.jsonl --dry-run --limit 3
```

O primeiro liga 2.104 das 2.203 UDVs a um dos 1.851 atores; o segundo mantém só as audiências do treino
(264 atores em 139 audiências); o terceiro renderiza os prompts sem carregar modelo. A geração de verdade
(sem `--dry-run`, com `--model`) precisa do extra `profiles`. Detalhes em [docs/actors.md](docs/actors.md)
e [docs/profiles.md](docs/profiles.md).

## Comandos

| comando | para quê |
|---|---|
| `build-udvs` | liga cada opinião do LDS a um trecho da transcrição e grava a rodada |
| `verify-udvs` | recalcula a rodada a partir do LDS e acusa qualquer registro divergente |
| `export-hearing`, `export-site` | gravam uma audiência, ou todas e o índice, no JSON da demo, lendo só o cache |
| `build-splits`, `verify-splits` | constroem e conferem o split temporal |
| `filter-actor-speeches` | restringe as falas por ator às audiências de alguns splits |
| `generate-profiles`, `validate-profiles` | escrevem um perfil por ator e o conferem contra as UDVs do ator |
| `sample-profile-review`, `score-profile-review` | sorteiam pares para revisão humana e pontuam a revisão preenchida |

Sem `--config`, os comandos de UDV e de exportação procuram `configs/udv.toml` e os de split,
`configs/splits.toml`; os comandos de perfil exigem `--config`. O que cada comando grava, a saída de
`--help` e os códigos de saída estão na [referência da CLI](docs/cli.md).

## Uso como biblioteca

A API pública é reexportada por `bookworm` (UDVs, splits, encoders, exportação) e por
`bookworm.actors` e `bookworm.profiles`. O exemplo abaixo, rodado dentro de `experiments/`,
constrói as UDVs das 20 primeiras audiências com TF-IDF, exporta uma audiência e refaz o split temporal:

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

As duas últimas linhas imprimem `{'train': 144, 'validation': 32, 'test': 30}` e `[]`. Os resultados
de `build_temporal_split` não dependem do encoder; as UDVs só entram nas contagens do relatório.

Pontos de entrada principais:

| Tarefa | Função ou classe |
| --- | --- |
| Ler o LDS com checagem de hash | `load_hearings` |
| Ler configuração | `load_udv_config`, `load_split_config`, `bookworm.actors.config.load_actors_config` |
| Construir UDVs | `build_udvs`, `build_hearing_udvs`, `bookworm.pipeline.run_pipeline` (UDVs e atores) |
| Verificar UDVs | `verify_udv_run`, `compare_with_baseline` |
| Encoders | protocolo `SentenceEncoder`, `TfidfEncoder`, `CachedEncoder`, `RunCacheEncoder` |
| Split temporal | `build_temporal_split`, `verify_split_run`, `article_date` |
| Exportar para a demonstração | `export_hearing`, `export_site`, `load_site_signals` |
| Perfis | `bookworm.profiles`: `run_generate_profiles`, `ChatClient`, `validate_profiles` |

## Documentação

| documento | conteúdo |
|---|---|
| [docs/cli.md](docs/cli.md) | cada comando, o que grava, `--help` e códigos de saída |
| [docs/configuration.md](docs/configuration.md) | formatos de `udv.toml` e `splits.toml`, e as chaves que ligam `udv_v2` |
| [docs/data_model.md](docs/data_model.md) | formato de todos os arquivos lidos e gravados |
| [docs/architecture.md](docs/architecture.md) | módulos, split temporal, checagens das verificações e paridade |
| [docs/actors.md](docs/actors.md), [docs/profiles.md](docs/profiles.md), [docs/profile_validation.md](docs/profile_validation.md) | falas por ator, perfis e a conferência dos perfis |
| [docs/testing.md](docs/testing.md) | `ruff`, `mypy` e `pytest`, marcadores de teste e variáveis de ambiente |
| [docs/limitations.md](docs/limitations.md) | limitações conhecidas das regras e uso dos resultados |
| [docs/adr/](docs/adr/README.md) | decisões de arquitetura, com o efeito medido de cada uma |
| [web/README.md](web/README.md) | a demo web: exportar, servir e o que a página mostra |

As regras de construção de uma UDV estão em
[`docs/methodology/udv.md`](../docs/methodology/udv.md#construção), e os resultados, em
[`docs/report.md`](../docs/report.md).

## Citação

Os metadados de citação estão em [`CITATION.cff`](../CITATION.cff), e o BibTeX do artigo do dataset, no
[README do repositório](../README.md#licença-e-citação).
