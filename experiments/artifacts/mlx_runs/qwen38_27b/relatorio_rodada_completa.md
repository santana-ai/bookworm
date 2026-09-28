# Rodada completa do Qwen3.8 27B (8 bits, MLX)

Data: 27 e 28/09/2026. Modelo: `mlx-community/Qwen3.8-27B-8bit`, com o backend de `mlx_alternative/`
num Apple M4 Max de 128 GB. Código em `6025c77`, com o prompt de múltipla escolha e o system prompt
de simulação revisados nesse commit (`prompt_version` 4a5995d1da8a da simulação e a80b3a121fab dos
perfis). Perfis gerados só com as audiências de `train` do split `temporal_v1`; k e γ escolhidos em
`validation`; resultados em `test`.

No teste, o perfil aumentou o acerto da múltipla escolha de 30,7% para 42,6%, uma diferença de 11,9
pontos percentuais com intervalo de 95% de [4,9; 19,6]. Os trechos recuperados acrescentaram 2,0
pontos, com intervalo que inclui zero, e a guidance não entrou, porque γ = 1,0 teve o maior acerto na
validação. Os níveis de evidência não ordenaram o acerto como o desenho exige.

## Execução

| Etapa | Volume | Duração |
|---|---|---|
| Perfis | 264 atores, nenhuma falha; 1,70 milhão de tokens de entrada e 239 mil de saída | 6,2 h |
| Múltipla escolha, validação | 81 perguntas de 51 atores em 21 audiências | 2,8 h |
| Múltipla escolha, teste | 101 perguntas de 46 atores em 24 audiências | 2,0 h |
| Geração aberta | 53 pedidos | 2,1 h |

Das UDVs ligadas a atores com perfil, 21 de 102 em validação e 5 de 106 em teste foram descartadas
por não terem 3 outros atores na mesma audiência para servir de distratores. A avaliação e a geração
aberta demoraram mais que a projeção feita com a rodada curta, porque os perfis da rodada completa
são maiores (mediana de 610 palavras) e deixam os prompts mais longos.

## Múltipla escolha

Cada pergunta mostra 4 proposições da mesma audiência, uma do ator, e o acaso é 25%. A resposta é
lida pelas probabilidades das letras no primeiro token, com média das 4 rotações da ordem das opções.

| Condição | Acerto, teste | Probabilidade média da opção certa, teste | Acerto, validação |
|---|---|---|---|
| 0, nome e cargo | 30,7% | 0,295 | 35,8% |
| 1, perfil | 42,6% | 0,366 | 45,7% |
| 2, perfil e trechos (k = 3) | 44,6% | 0,402 | 49,4% |
| 3, condição 2 com guidance (γ = 1,0) | 44,6% | 0,402 | 49,4% |

| Diferença no teste | Acerto | Intervalo de 95% |
|---|---|---|
| 1 menos 0 | +11,9 pontos | [+4,9; +19,6] |
| 2 menos 1 | +2,0 pontos | [-2,9; +7,0] |
| 3 menos 2 | 0 | [0; 0] |

Os intervalos vêm de bootstrap pareado com 1.000 reamostragens de audiências inteiras, porque as
perguntas de uma audiência compartilham assunto e participantes.

Na validação, o acerto da condição 2 foi de 49,4%, 48,1% e 46,9% com k = 3, 5 e 10, e o da condição
3 foi de 49,4%, 48,1%, 46,9% e 43,2% com γ = 1,0, 1,5, 2,0 e 3,0. O acerto caiu à medida que k e γ
subiram, e os dois ficaram no primeiro valor da grade.

A massa das letras (soma das probabilidades de A, B, C e D no primeiro token) ficou entre 0,998 e
0,999 em média nas condições 0 a 2, com mínimo de 0,969 no teste. Com o prompt anterior ao
`6025c77`, o piloto de dois atores tinha dado entre 0,985 e 0,996 para este modelo; nos dois modelos
Gemma do piloto, a massa caía para até 0,00015 com o prompt anterior, e o efeito da revisão neles não
foi medido.

## Níveis de evidência

O desenho em `ACTOR_SIMULATION.md` considera que os níveis se sustentam se o acerto cair de DIRETA
para ESPECULATIVA. No teste:

| Nível | Condição 1: perguntas / acerto | Condição 2: perguntas / acerto |
|---|---|---|
| DIRETA | 22 / 54,5% | 22 / 54,5% |
| INDIRETA | 30 / 36,7% | 38 / 36,8% |
| ESPECULATIVA | 15 / 46,7% | 19 / 52,6% |
| SEM BASE | 34 / 38,2% | 22 / 40,9% |

DIRETA teve o maior acerto, mas ESPECULATIVA ficou acima de INDIRETA nas duas condições. Não foi
calculado intervalo para as diferenças entre níveis; com 15 a 38 perguntas por nível, elas não
permitem afirmar uma ordem, e os níveis não podem ser apresentados como calibração da confiança.

## Geração aberta

| | Abordagem 1, perfil | Abordagem 2, perfil e trechos |
|---|---|---|
| Falas geradas | 38 | 40 |
| Recusas (nível SEM BASE) | 15 | 13 |
| Níveis DIRETA / INDIRETA / ESPECULATIVA | 8 / 23 / 7 | 9 / 25 / 6 |
| Frases sem fundamento na conferência | 27 de 415 (6,5%) | 34 de 441 (7,7%) |
| Falas ou justificativas truncadas | 0 | 0 |
| Trechos de justificativa que não puderam ser lidos como JSON | 2 | 0 |

A conferência verifica que cada frase cita um item existente no material e que a cópia citada é
literal; ela não verifica se o item sustenta a frase, o que exige leitura humana.

A fala gerada só com o material da condição 0 (`baseline_speech`) começa com uma recusa em 38 dos 53
pedidos. A contagem é aproximada: ela procura expressões como "não há material" e "não possuo" no
texto, sem leitura das falas.

## Limitações

- O prompt da múltipla escolha e o system prompt foram revisados depois do piloto de 27/09, que usou
  perguntas de audiências de `test`. A revisão se baseou na massa das letras e na leitura das
  respostas, sem uso do acerto, como registrado em `ACTOR_SIMULATION.md`, e o artigo precisa
  declarar essa sequência.
- 28 dos 264 perfis passam do teto de 800 palavras do prompt de perfil; o maior tem 1.209. No
  piloto, os perfis deste modelo também traziam siglas expandidas, especulação entre parênteses e
  cortesia registrada como traço, o que o prompt proíbe. A frequência desses desvios nos 264 perfis
  não foi medida.
- Os resultados são de um único modelo, quantizado em 8 bits, e de uma única semente.
- O teste tem 101 perguntas em 24 audiências. O intervalo do efeito do perfil vai de 4,9 a 19,6
  pontos.

## Arquivos

Em `mlx_alternative/runs/qwen38_27b/`: `actor_profiles/actor_profiles_train.jsonl` (264 perfis),
`actor_simulation/evaluation.json`, `choice_validation.jsonl` e `choice_test.jsonl` (múltipla
escolha), `actor_simulation/simulations.jsonl` e `simulations_summary.json` (geração aberta),
`timings.jsonl` (tempos) e `full_run.log`. As saídas do piloto com o prompt anterior estão em
`archive_prompts_109a670/`.
