# Documentação

Esta pasta guarda os documentos do projeto, do resumo de uma página ao relatório com a origem de cada
número. A documentação de uso da biblioteca fica em [`bookworm/docs/`](../bookworm/docs/), e a dos
experimentos, em [`experiments/README.md`](../experiments/README.md).

## Ordem de leitura

| # | documento | para quê | tamanho |
|---|---|---|---|
| 1 | [README do repositório](../README.md) | o que o projeto faz, os resultados principais e como conferir a rodada publicada | 1 página |
| 2 | [`pipeline.md`](pipeline.md) | o que cada experimento comparou e decidiu, e o pipeline recomendado `udv_v2` | médio |
| 3 | [`report.md`](report.md) | todos os experimentos, métricas, resultados, limitações e referências, com o artefato de cada número | longo |
| 4 | [`reproduce.md`](reproduce.md) | as 25 etapas em ordem, com comando, configuração e saída versionada, ambiente e artefatos pesados | médio |

Quem vai escrever sobre o projeto lê 1 a 3; quem vai rodar, 1 e 4.

## Metodologia

Cada documento explica uma etapa pelo problema que ela resolve, com as regras e as medidas que as
justificam. Os números de resultado são citados do relatório.

| documento | etapa |
|---|---|
| [`methodology/udv.md`](methodology/udv.md) | o registro UDV e as regras de construção: turnos, resolução da pessoa, sentenças, citação, similaridade, nível e offsets (`udv_v1`; as mudanças de `udv_v2` estão em [`pipeline.md`](pipeline.md#pipeline-recomendado-udv_v2)) |
| [`methodology/confidence.md`](methodology/confidence.md) | a pergunta do Trabalho 2, o levantamento da literatura de grounding e o experimento confidence_v2 |
| [`methodology/hearing_actors.md`](methodology/hearing_actors.md) | como separar a fala de cada pessoa entre audiências e identificar a mesma pessoa em audiências diferentes |
| [`methodology/actor_profiles.md`](methodology/actor_profiles.md) | o perfil de cada ator gerado só com as audiências de treino |
| [`methodology/actor_simulation.md`](methodology/actor_simulation.md) | a simulação de atores a partir do perfil e a avaliação por múltipla escolha |
| [`methodology/mlx_backend.md`](methodology/mlx_backend.md) | o backend MLX da rodada de perfis e simulação, e a escolha do modelo |

## Validação

| documento | conteúdo |
|---|---|
| [`validation/annotation_guide.md`](validation/annotation_guide.md) | o guia do anotador da validação humana: fluxo, colunas e critérios de cada rótulo |

O desenho da amostra e os resultados finais estão no [relatório, seção 7](report.md#7-validação-humana).

## Contexto e referência

| documento | conteúdo |
|---|---|
| [`vision.md`](vision.md) | a tese científica e a especificação de longo prazo; vai além do que esta versão implementa |
| [`roadmap.md`](roadmap.md) | o estado do projeto, o que falta e o checklist de entregas do desafio |
| [`challenge/brief.md`](challenge/brief.md) | o brief oficial do desafio Ideias em Rede |
| [`path_map.md`](path_map.md) | os caminhos da versão de pesquisa (tag `research-2026-09-28`) e os desta versão, para ler artefatos antigos |

## Uma fonte por assunto

| assunto | fonte | quem só aponta para ela |
|---|---|---|
| números de resultado | [`report.md`](report.md) | README, `pipeline.md`, metodologia |
| decisões dos experimentos | [`pipeline.md`](pipeline.md) | README, `report.md` |
| regras de construção da UDV | [`methodology/udv.md`](methodology/udv.md) | [`bookworm/docs/architecture.md`](../bookworm/docs/architecture.md) |
| comandos de reprodução | [`reproduce.md`](reproduce.md) | `experiments/README.md`, `report.md` |
| comandos e opções da CLI | [`bookworm/docs/cli.md`](../bookworm/docs/cli.md) | `bookworm/README.md` |
| formato dos arquivos | [`bookworm/docs/data_model.md`](../bookworm/docs/data_model.md) | `bookworm/README.md`, demo web |
| decisões de arquitetura | [`bookworm/docs/adr/`](../bookworm/docs/adr/README.md) | `pipeline.md`, `report.md` |
