# Perfis de atores

Este documento descreve os comandos `bookworm filter-actor-speeches` e `bookworm generate-profiles`,
que transformam as falas de cada ator recorrente em um perfil escrito por um LLM. Eles portam, com o
mesmo comportamento, `challenge/utils/filter_actor_speeches.py` e
`challenge/utils/generate_actor_profiles.py` (ADR 0005).

## Problema

As UDVs dizem o que uma pessoa defendeu numa audiência, uma opinião por vez. Para comparar atores
entre audiências, buscar quem tem posições parecidas ou estimar o que alguém diria sobre um tema
novo, é preciso um resumo de tudo o que a pessoa falou. O perfil é esse resumo: um texto por ator,
escrito a partir só das falas da transcrição, para ser convertido em embedding numa etapa posterior.

Um perfil escrito com todas as audiências já leu as falas das audiências de teste. Se as opiniões
publicadas dessas audiências forem usadas para avaliar o perfil (`bookworm validate-profiles`), a
avaliação mede o que o perfil copiou, não o que ele permite estimar. Por isso a entrada da geração
passa antes por um filtro que mantém só as audiências dos splits configurados (por padrão só
`train` de `temporal_v1`).

## Etapas

1. `bookworm build-udvs --actors-config ...` grava as falas por ator (`actors_multi_hearing.jsonl`,
   um registro por ator com fala em pelo menos duas audiências), com os mesmos bytes do script
   `challenge/utils/build_actor_speeches.py`.
2. `bookworm filter-actor-speeches` mantém de cada ator só as audiências dos splits listados, sem
   alterar os turnos, descarta o ator que fica sem nenhuma e grava o arquivo filtrado e as contagens.
3. `bookworm generate-profiles` monta um prompt por ator, com as falas em ordem cronológica, e grava
   um perfil por linha.

## Como rodar

Os comandos leem a mesma configuração do `challenge/` (`configs/actor_profiles.toml`) e resolvem os
caminhos relativos a partir do diretório em que são executados. Dentro de `challenge/`:

```
uv run bookworm filter-actor-speeches --config configs/actor_profiles.toml
```

Antes de carregar um modelo, confira os prompts com `--dry-run`. Ele lê as falas e o LDS, renderiza
todos os prompts selecionados e imprime quantos são e quantos caracteres têm, sem carregar modelo e
sem gravar nada:

```
uv run bookworm generate-profiles --config configs/actor_profiles.toml \
  --input artifacts/cache/hearing_actors/actors_multi_hearing_train.jsonl --dry-run
```

Com `temporal_v1`, a saída tem 264 prompts, 6.596.900 caracteres no total e o maior prompt, de
Erika Kokay, com 283.255 caracteres. O `prompt_version` dos prompts atuais é `8428246d3fae`.

A geração de verdade precisa do extra `profiles` (`transformers`, `accelerate`, `torch`) e do id de
um modelo do Hugging Face ou do caminho de uma pasta local:

```
uv sync --extra profiles
uv run bookworm generate-profiles --config configs/actor_profiles.toml \
  --input artifacts/cache/hearing_actors/actors_multi_hearing_train.jsonl \
  --output artifacts/actor_profiles/actor_profiles_train.jsonl \
  --model <modelo>
```

O modelo também pode ficar em `name`, na seção `[model]` da configuração (vazio por padrão); sem ele,
o comando para com erro antes de ler as falas.

Opções de `generate-profiles`:

- `--config`: arquivo TOML (obrigatório).
- `--input` (ou `--speeches`): JSONL de falas por ator; substitui `[input].speeches_path`.
- `--output`: JSONL de perfis; substitui `[output].profiles_path`.
- `--model`: id ou pasta do modelo; substitui `[model].name`.
- `--actors`: só esses atores, pelo nome exato do campo `actor`; repita a opção para mais de um.
- `--limit`: no máximo N atores nesta execução.
- `--dry-run`: renderiza os prompts e imprime as contagens, sem modelo.

`filter-actor-speeches` aceita `--config` e `--output` (substitui `[split_filter].speeches_path`).

## Configuração

- `[input]`: `speeches_path` (falas por ator, entrada da geração), `lds_path` e `lds_sha256`. O LDS
  fornece a data (a da publicação da matéria, extraída como em `bookworm.data.dates.article_date`) e
  o `assunto` de cada audiência, e é conferido por SHA-256.
- `[output]`: `profiles_path`.
- `[split_filter]`: `manifest_path` (manifesto do split; o comando recusa um manifesto gerado de
  outro LDS), `splits` (nomes distintos entre `train`, `validation` e `test`), `speeches_path`
  (arquivo filtrado) e `stats_path` (contagens).
- `[prompts]`: `dir`, `system_profile` e `user_profile`. Sem `dir`, valem os prompts empacotados em
  `bookworm/profiles/prompts/`, cópias byte a byte dos de `challenge/prompts/actor_profile/`.
- `[model]`: `name`, `device_map` (repassado a `from_pretrained`; com `"auto"`, o `accelerate`
  distribui o modelo entre as GPUs e manda para a CPU o que não couber), `temperature`, `top_p`,
  `max_output_tokens` (vira `max_new_tokens`) e `seed`, fixada antes de cada geração.

## Formato dos arquivos

O arquivo filtrado tem o mesmo formato das falas por ator. As contagens (`stats_path`) têm três
blocos: `split_manifest` (caminho, SHA-256, versão, splits e número de audiências), `input` e
`output` (caminho, SHA-256, atores, audiências, turnos e distribuição de audiências por ator; o
`output` traz também `actors_without_split_hearings`).

Cada linha do arquivo de perfis (`bookworm.profiles.ProfileRecord`) tem, nesta ordem:

- `actor`: nome do ator, igual ao das falas; é por ele que a conferência liga o perfil às UDVs.
- `profile`: o texto do perfil.
- `model` e `prompt_version`: o nome passado a `from_pretrained` e os 12 primeiros caracteres
  hexadecimais do SHA-256 dos dois arquivos de prompt (nome e bytes de cada um, em ordem de nome).
- `n_statements` (turnos), `n_hearings` e `hearing_ids` (as audiências que entraram no prompt).
- `input_tokens` e `output_tokens`, no tokenizer do modelo.
- `generated_at` (UTC, em segundos) e `duration_seconds`.

## Retomada e falhas

O arquivo de perfis é aberto em modo append e cada perfil é gravado assim que fica pronto. Ao rodar
de novo com o mesmo `--output`, os atores que já têm linha são pulados, pelo nome; para regerar um
ator, apague a linha dele. Uma linha inválida (por exemplo, deixada por interrupção no meio da
escrita) é ignorada na retomada, com aviso, mas continua no arquivo; remova-a antes de consumir o
JSONL, porque `read_profiles` recusa o arquivo. Não rode duas execuções sobre o mesmo `--output`.

Erro em um ator não interrompe a execução: o ator fica fora do arquivo, e portanto é tentado de novo
na próxima rodada, e o resumo final lista os nomes em `failed_actors`. Contam como falha uma resposta
que chega a `max_output_tokens`, uma resposta com bloco `<think>` sem fechamento e uma resposta
vazia; um bloco `<think>` fechado no início da resposta é removido. Com qualquer falha, o comando
termina com código 1; erro de configuração ou de dados termina com código 2.

## Reprodutibilidade

- As falas por ator, o arquivo filtrado e as contagens são determinísticos. O teste com o marcador
  `dataset` (`tests/integration/test_parity_profile_split_filter.py`) refaz as falas a partir do LDS,
  roda o filtro e compara as contagens byte a byte com
  `challenge/artifacts/actor_profiles/train_speeches_stats.json`: 301 atores, 198 audiências e 6.323
  turnos na entrada; 264 atores, 139 das 144 audiências de treino e 4.598 turnos na saída, com 37
  atores sem audiência de treino. O mesmo arquivo confere que os prompts renderizados são idênticos
  aos do script do `challenge/` para os 264 atores.
- A geração depende do modelo, da semente e do hardware: a mesma semente não garante o mesmo texto
  entre GPUs ou versões do `transformers`. O registro guarda `model` e `prompt_version`, mas não a
  revisão do checkpoint; para fixá-la, passe uma pasta local em `--model`.
- O comando não compara o tamanho do prompt com a janela de contexto do modelo. Use o `--dry-run`
  para ver o maior prompt em caracteres e o `input_tokens` de cada linha para o tamanho real.
- O perfil é texto gerado por modelo; a fundamentação nas falas depende de o modelo seguir o prompt.
  A conferência contra as UDVs está em `bookworm validate-profiles`.
