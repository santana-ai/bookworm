# Lint, tipos e testes

Como rodar as verificações da biblioteca, o que cada marcador de teste cobre e as variáveis de
ambiente que ligam os testes contra o dataset real e o cache de embeddings.

## Verificações

Rodado a partir de `bookworm/`:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

`ruff` usa as regras de `[tool.ruff.lint]` do `pyproject.toml`, `mypy` roda em modo estrito sobre `src`
e `tests`, e `pytest` mede cobertura por linha. Os testes de dataset precisam do LDS real e, para as
reconstruções byte a byte, do cache de embeddings de `experiments/`:

```bash
BOOKWORM_LDS_PATH=../experiments/dataset/PublicHearingBR_LDS.jsonl \
BOOKWORM_EMBEDDING_CACHE=../experiments/artifacts/cache/embeddings \
uv run pytest -m dataset
```

Os testes que dependem dos extras (`tests/unit/test_sentence_transformer.py`, parte de
`tests/unit/test_cli.py` e de `tests/unit/test_profiles_llm.py`) são pulados quando o extra não está
instalado; `uv sync --all-extras` os inclui.

## Marcadores

- `unit` e `integration`: aplicados pelo diretório (`tests/unit/`, `tests/integration/`) em
  `tests/conftest.py`. Os testes de `tests/unit/` usam só as fixtures sintéticas escritas à mão de
  `tests/fixtures/`, sem modelo e sem o dataset real. Os de `tests/integration/` comparam a biblioteca
  com os scripts e artefatos de `experiments/`; todos têm também `dataset` ou `model`, menos um, que roda
  o script de referência das falas por ator sobre as fixtures e é pulado sem `experiments/`.
- `dataset`: usa o LDS real indicado por `BOOKWORM_LDS_PATH`; o teste é pulado quando a variável não
  está definida ou o arquivo não existe, e falha quando o sha256 não bate.
- `model`: carrega o encoder Serafim do cache local do Hugging Face; é opt-in e fica fora da execução
  padrão (`addopts` usa `-m 'not model'`). Rodar com `uv run pytest -m model`.

Nenhum teste acessa a rede: uma fixture automática de `tests/conftest.py` define `HF_HUB_OFFLINE`,
`HF_DATASETS_OFFLINE` e `TRANSFORMERS_OFFLINE` e faz falhar qualquer resolução de nome ou conexão
TCP que não seja local (`tests/unit/test_network_guard.py` confere isso).

## Variáveis de ambiente

- `BOOKWORM_LDS_PATH`: o arquivo `PublicHearingBR_LDS.jsonl`.
- `BOOKWORM_EMBEDDING_CACHE`: diretório de cache de embeddings já existente (por exemplo
  `experiments/artifacts/cache/embeddings`). Com ele, `tests/integration/test_parity_udv_cached_rebuild.py`
  reconstrói `udv_v1` e `udv_v1_pre`, `tests/integration/test_export_hearing.py` exporta a audiência 70
  e `tests/integration/test_export_site.py` exporta as 206 audiências, só a partir do cache: o encoder
  desses testes falha em qualquer texto que não esteja no cache, o cache é aberto em modo somente
  leitura e o teste confere que nenhum arquivo do diretório foi criado ou alterado. Sem a variável,
  esses testes são pulados.
- `BOOKWORM_EXPERIMENTS_DIR`: o projeto com os scripts de referência (padrão: `experiments/` do
  repositório). `BOOKWORM_ARTIFACTS_DIR`: o diretório com as execuções publicadas (padrão:
  `<BOOKWORM_EXPERIMENTS_DIR>/artifacts`). Um artefato ausente faz o teste correspondente ser pulado; os
  arquivos grandes que ficaram fora do repositório estão listados, com o comando que regenera cada um,
  em `experiments/artifacts/MANIFEST_heavy.tsv`.
- `BOOKWORM_PARITY_FULL=1`: o teste `model` reconstrói as 206 audiências de `udv_v1` em vez das 20
  primeiras.

## O que os testes de dataset cobrem

O que o marcador `dataset` cobre: a paridade das datas e dos cortes de split com
`experiments.data.legacy_splits`, carregado pelo caminho do arquivo; a conferência, registro a registro, de tudo o
que não depende do encoder em `udv_v1.jsonl` e `udv_v1_pre.jsonl` e a igualdade byte a byte desses
dois arquivos depois de lidos e regravados; a verificação sem problemas dos dois e a lista exata de
problemas de `udv_v0`; a paridade byte a byte com `temporal_v1.json` (exceto `created_at`) e
`temporal_v1_report.json` (exceto `created_at` e `environment`); as falas por ator e o filtro por split
contra os scripts de `experiments/`; as propriedades do LDS citadas em
[data_model.md](data_model.md); os cinco defeitos injetados no manifesto real (audiência
trocada de conjunto, audiência removida, data alterada, fronteira alterada, contador do relatório
alterado); uma execução TF-IDF nas 20 primeiras audiências que precisa passar em todas as checagens e
ter as mesmas citações de `udv_v1`; a exportação da audiência 70 de `udv_v1`, com os sinais do
verificador quando o cache de tradução e os arquivos de score estão presentes; e a exportação das 206
audiências de `udv_v1` com `export_site` (contagens do índice, manchetes e tamanhos dos arquivos).

## Fixtures sintéticas

As fixtures sintéticas: `lds_mini.jsonl` cobre os casos de resolução de participante e de evidência,
inclusive um participante com dois turnos separados por `(Palmas.)`; `lds_turns.jsonl` cobre as regras
de citação por turno (prefixo repetido em dois turnos, sentenças idênticas em dois turnos, segunda
citação com prefixo mais longo que a primeira, citação entre aspas simples, prefixo curto repetido,
rubrica colada a uma pergunta, elisão `(...)` e sentença contida noutra sentença de outro turno);
`splits_mini.jsonl` tem 10 audiências com datas, menções ao dia da semana (inclusive uma que não
confere) e um `Atualizado em`. `StubEncoder` (em `tests/conftest.py`) associa textos a vetores
escolhidos à mão, o que permite fixar casos de fronteira como score exatamente igual ao limiar. O teste
`model` exige ids, tiers, `support_type`, texto, offsets e `quote_prefix` iguais e aceita scores que
diferem em até `1e-5`; ele não usa `BOOKWORM_EMBEDDING_CACHE`, e todo embedding é calculado pelo
modelo carregado.

