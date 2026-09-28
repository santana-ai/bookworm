# Como contribuir

O repositório tem dois projetos `uv` independentes, cada um com seu `pyproject.toml` e seu `uv.lock`:
`bookworm/` (a biblioteca) e `challenge/` (os experimentos). Uma mudança só entra quando lint, tipos e
testes passam no projeto que ela toca.

## Ambiente

Requer Python 3.12 e [uv](https://docs.astral.sh/uv/).

```bash
cd bookworm && uv sync            # núcleo em CPU
uv sync --all-extras              # inclui sentence-transformers, torch e transformers

cd ../challenge && uv sync
uv run python -m utils.download_dataset
```

O dataset vai para `challenge/dataset/` e nunca é versionado. Nenhum teste acessa a rede, e credenciais
não entram no repositório; o que um componente de LLM precisa fica atrás de uma interface e de variáveis
de ambiente.

## Verificações

Em `bookworm/`:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

Os testes marcados `dataset` leem o LDS real e o cache de embeddings de `challenge/`:

```bash
BOOKWORM_LDS_PATH=../challenge/dataset/PublicHearingBR_LDS.jsonl \
BOOKWORM_EMBEDDING_CACHE=../challenge/artifacts/cache/embeddings \
uv run pytest -m dataset
```

Em `challenge/`:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q
```

`mypy` roda em modo estrito nos dois projetos. Os marcadores de teste e as variáveis de ambiente estão
descritos em [`bookworm/README.md`](bookworm/README.md#lint-tipos-e-testes).

## Commits

- Um commit por mudança coerente, pequeno o bastante para ser revisto sozinho.
- Assunto em inglês, no imperativo, começando com maiúscula e sem ponto final (por exemplo, "Add the
  window2 unit to the UDV builder"). Quando a mudança é num único documento, o assunto pode começar pelo
  nome do arquivo ("PIPELINE.md: fix the UDV expansion").
- O corpo, quando existe, explica o porquê da mudança e o que ela não cobre.
- Artefatos regenerados entram no mesmo commit do código ou da configuração que os produziu.

## Decisões de arquitetura

Uma mudança de arquitetura da biblioteca (formato da UDV, regras de construção, contrato de uma
exportação) começa por um ADR em [`bookworm/docs/adr/`](bookworm/docs/adr/README.md), numerado em
sequência, com status, contexto, decisão e consequências. O formato e o índice ficam no README da
pasta; uma ADR aceita não é reescrita, e uma decisão posterior entra numa ADR nova que cita a anterior.

## Experimentos

- Toda configuração de experimento é um arquivo TOML em `challenge/configs/`, escrito antes da rodada.
  Mudanças feitas depois de ver resultados são registradas no próprio arquivo, com data e motivo.
- O conjunto de teste não é usado para escolher métodos, cortes ou hiperparâmetros; o comando que o lê
  exige `--final-test`.
- Todo número citado num documento precisa ter o artefato ou a célula que o produziu no repositório.
- Nenhum script preenche julgamento humano: o script gera a planilha com o campo vazio, e o
  preenchimento é manual.

## Convenções de código e texto

- Identificadores (variáveis, funções, classes, módulos) em inglês, nos dois projetos.
- Sem comentários no código; nomes e docstrings carregam a explicação. Quando um comentário for
  inevitável, ele é em inglês.
- Documentação, relatórios e as células de texto dos notebooks em português do Brasil.
- Em texto corrido, sem travessão: use vírgula, ponto e vírgula, dois pontos ou ponto.
- Todo conceito novo num documento é explicado pelo problema que ele resolve.
- Notebooks seguem a ordem: título, explicação curta, imports gerais, e depois uma seção `##` por
  investigação, com uma responsabilidade por célula de código.
