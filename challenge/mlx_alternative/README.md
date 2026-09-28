# Backend MLX para perfis e simulação de atores

Esta pasta roda o pipeline de perfis e de simulação de atores (`utils/generate_actor_profiles.py`,
`utils/evaluate_actor_simulation.py`, `utils/simulate_actors.py`) com modelos no formato MLX, em
Apple Silicon, sem alterar nenhum arquivo de `utils/` nem de `configs/`. O objetivo é comparar
modelos que cabem na máquina local e escolher um para a rodada completa.

## Como funciona

Os três scripts dependem do `transformers` em um só ponto: a classe que carrega o modelo
(`TransformersChatClient` no gerador de perfis, `SimulationModel` na avaliação e na simulação). O
`run.py` troca essas classes, só dentro do próprio processo, por equivalentes em MLX (`backend.py`)
e chama o `main()` original de cada script. Construção de perguntas, distratores, rotações, escolha
de k e γ, bootstrap, conferência da justificativa, retomada e fingerprints continuam sendo os do
código original.

Para cada modelo, o `run.py` gera cópias dos TOMLs de `configs/` em `runs/<id>/configs/`, trocando
só o nome do modelo e os caminhos de saída. Assim, cada modelo grava em `runs/<id>/`. As estatísticas
das etapas de preparação, que no pipeline original vão para arquivos versionados de `artifacts/`,
vão para `runs/shared/`. As falas por ator continuam em `artifacts/cache/`, fora do Git, como no
pipeline original.

## Modelos candidatos

Ficam em `config.yaml`, cada um identificado pelo `repo` do Hugging Face. A pasta local de cada
modelo é `<models_dir>/<repo>`: o padrão de `models_dir` é `~/.lmstudio/models`, a mesma
organização do LM Studio, e a variável de ambiente `MLX_MODELS_DIR` o substitui sem editar o
arquivo. O pipeline recebe e grava o `repo` (no campo `model` dos perfis, em `evaluation.json` e nas
fingerprints de retomada), e só o backend traduz esse id para a pasta local, então nenhuma saída
depende da máquina em que rodou.

Para baixar um modelo nesse lugar:

```
uv run --with "mlx-lm==0.31.3" python -m mlx_alternative.run download --model qwen38_27b
```

O download lê `HF_TOKEN` de `mlx_alternative/.env` (caminho em `env_file`), quando o arquivo
existe; o `.env` é ignorado pelo Git.

| id | modelo | tipo |
|---|---|---|
| `gemma4_31b` | Gemma 4 31B instruct, 8 bits | denso |
| `qwen38_27b` | Qwen3.8 27B, 8 bits | denso, atenção híbrida |
| `gemma4_26b_a4b` | Gemma 4 26B-A4B instruct, 8 bits | mistura de especialistas, cerca de 4B ativos |

Os três têm modo de raciocínio no chat template. `chat_template_kwargs: {enable_thinking: false}`
o desliga, porque a avaliação lê a letra da resposta no primeiro token gerado. O `mlx-lm` liga o
raciocínio por padrão quando o modelo suporta, então a opção precisa ficar explícita.

## Como rodar

Tudo roda de dentro de `challenge/`. O `mlx-lm` entra só no ambiente da execução, sem mudar o
`pyproject.toml`:

```
uv run --with "mlx-lm==0.31.3" python -m mlx_alternative.run smoke --model gemma4_31b
uv run --with "mlx-lm==0.31.3" python -m mlx_alternative.verify_backend --model gemma4_31b
uv run --with "mlx-lm==0.31.3" python -m mlx_alternative.run all --model gemma4_31b
uv run --with "mlx-lm==0.31.3" python -m mlx_alternative.benchmark
```

- `run smoke`: carrega o modelo e grava `runs/<id>/smoke.json`, com o fim do prompt montado pelo
  template (para conferir que não há bloco de raciocínio aberto), os ids das letras A a D, a massa
  das letras num prompt de múltipla escolha, uma resposta curta em português e o pico de memória.
- `verify_backend`: compara o backend com cálculos diretos. Confere os log-probs das letras contra
  o prefill padrão do `mlx-lm` e cada passo da geração gulosa contra o argmax de um forward
  completo do prompt com os tokens já gerados. Grava `runs/<id>/verify.json`.
- `run <etapa>`: `profiles`, `evaluate`, `simulate` ou `all`. Sem `--actors`, usa os atores de
  `short_run` (Arnaldo Jardim e Daniela Reinehr, que têm perguntas em `validation` e em `test`);
  `--all-actors` roda todos. `--k` vai para `utils.simulate_actors`. Os tempos
  de cada etapa vão para `runs/<id>/timings.jsonl`.
- `benchmark`: para cada modelo, mede carga, memória, velocidade de prefill e de geração num
  prompt de perfil longo (o ator com o maior prompt) e num prompt de múltipla escolha, e o ganho do
  reuso de prefixo. Projeta a duração da rodada completa com esses números e com os tempos da rodada
  curta. Grava `runs/benchmark.json` e `runs/benchmark.md`.

## Rodada sobre `udv_v2`

`config_udv_v2.yaml` refaz perfis, avaliação e simulação com o modelo da rodada completa,
`mlx-community/Qwen3.8-27B-8bit`, lendo os TOMLs de `configs/`, que apontam para
`artifacts/udv/udv_v2.jsonl`. As saídas vão para `runs/udv_v2/qwen38_27b/` e `runs/udv_v2/shared/`;
a rodada `udv_v1` em `runs/qwen38_27b/` não é lida nem sobrescrita. De dentro de `challenge/`:

```
uv run --with "mlx-lm==0.31.3" python -m mlx_alternative.run all --model qwen38_27b --all-actors --settings mlx_alternative/config_udv_v2.yaml
```

Se o modelo não estiver em `<models_dir>/mlx-community/Qwen3.8-27B-8bit`, `run download` com as
mesmas opções baixa o modelo antes. A revisão do modelo não está fixada (`revision: null`): a pasta
local da rodada `udv_v1` não guardou o commit do Hugging Face. Quando o commit for conhecido, ele
entra em `revision` antes da rodada, e `download` e `smoke.json` passam a registrá-lo.

As entradas da rodada podem ser conferidas antes e depois dela pelos dry runs, que não usam modelo:
`artifacts/actor_profiles/udv_v2_dry_run_profiles.json` (prompts dos perfis),
`artifacts/actor_simulation/udv_v2_dry_run_evaluation.json` (perguntas de múltipla escolha) e
`artifacts/actor_simulation/udv_v2_dry_run_simulation.json` (pedidos de simulação). Os comandos que
os gravam estão em `artifacts/udv/udv_v2_downstream_report.json`.

## Decisão entre os modelos

1. Rodar `smoke`, `verify_backend` e `run all` (rodada curta) para cada modelo, depois o
   `benchmark`.
2. Descartar o modelo cuja projeção de horas não cabe no prazo.
3. Entre os restantes, ler lado a lado os perfis dos dois atores em
   `runs/<id>/actor_profiles/actor_profiles_train.jsonl` (fundamentação nas falas, os quatro blocos,
   nenhuma ausência descrita) e comparar a massa das letras (`letter_mass`) em `evaluation.json`.
4. O acerto da rodada curta serve só como checagem de funcionamento: com duas pessoas, a amostra é
   pequena demais para comparar modelos.

## Limitações

- A quantização em 8 bits produz outro modelo. O campo `model` dos perfis e das avaliações grava o
  `repo` do modelo MLX, e é esse modelo que precisa aparecer no artigo.
- A decodificação gulosa e a leitura das letras foram reimplementadas sobre o `mlx-lm`. A guidance
  da condição 3 da múltipla escolha é calculada pelo próprio `utils/evaluate_actor_simulation.py` a
  partir dos log-probs das condições 0 e 2, sem passar pelo backend. A verificação confere a
  implementação, mas os números não coincidem bit a bit com uma rodada em `transformers`.
- A amostragem da geração de perfis usa o gerador aleatório do MLX. A semente da configuração é
  fixada antes de cada perfil, então os perfis se repetem entre rodadas no mesmo backend, mas não
  coincidem com os de uma rodada em `transformers` com a mesma semente.
- Reuso de prefixo (`backend.prefix_cache`): desligado por padrão. Quando ligado, os prompts que
  compartilham o começo (as quatro rotações e as condições da múltipla escolha) reaproveitam o
  cache desse começo. No Gemma 4 31B, isso mudou as probabilidades das letras, renormalizadas entre
  si, em até 0,05, sem trocar a letra de maior probabilidade em nenhum dos 8 prompts verificados. A
  diferença vem de o cálculo em bf16 sobre o cache não reproduzir exatamente o do prompt inteiro de
  uma vez; o próprio prefill padrão do `mlx-lm` já difere assim do forward único, em menor grau.
  Uma diferença desse tamanho pode trocar a resposta de perguntas quase empatadas, e por isso o
  padrão é o prefill completo. O `benchmark` mede quanto tempo o reuso economizaria.
- No Gemma 4 31B, com o prompt de múltipla escolha sem perfil (condição 0), a resposta gulosa
  começa por uma recusa por falta de material, e a massa das letras no primeiro token foi 2,4e-5
  no prompt do `smoke`. A leitura por letra sempre escolhe entre A e D, então, nesse caso, ela
  capta muito pouco da decisão do modelo. `letter_mass`, em `evaluation.json`, mede isso para cada
  condição.
