# Rodada curta do Gemma 4 31B (8 bits, MLX)

Data: 27/09/2026. Modelo: `lmstudio-community/gemma-4-31B-it-MLX-8bit`, rodado com o backend de
`mlx_alternative/` num Apple M4 Max de 128 GB. Atores: Arnaldo Jardim e Daniela Reinehr, os dois
indicados em `ACTOR_SIMULATION.md` por terem perguntas em `validation` e em `test`.

O backend MLX funciona e os perfis gerados são bons. A avaliação por múltipla escolha, com o prompt
atual, não serve com este modelo: quando recebe o perfil, o modelo responde "Nenhuma das
alternativas" em vez de uma letra, e o acerto das condições 1 a 3 passa a ser ruído. A causa está
no texto do prompt e aconteceria também com o backend `transformers`.

## Backend

A leitura das letras coincide com o prefill padrão do `mlx-lm`: a diferença nas probabilidades
renormalizadas entre as quatro letras foi 0,0 nos 8 prompts verificados. Na geração com guidance
(γ = 1,5), os 8 passos conferidos coincidem com a combinação calculada por forwards completos dos
dois prompts, e com γ = 1 a saída é idêntica à da geração sem guidance.

O reuso de prefixo entre prompts (`backend.prefix_cache`) ficou desligado. Ligado, ele mudou as
probabilidades renormalizadas das letras em até 0,05, sem trocar a letra de maior probabilidade em
nenhum dos 8 prompts. A diferença vem do cálculo em bf16 sobre o cache, que não reproduz exatamente
o do prompt inteiro de uma vez, e poderia trocar a resposta de perguntas quase empatadas.

## Perfis

Os dois perfis seguem o formato pedido: os quatro blocos com título fixo, itens começando por verbo
e nenhuma frase sobre o que a pessoa não disse. O de Arnaldo Jardim tem 316 palavras e registra as
posições sobre regras de despacho do ONS, renovação de concessões por norma legislativa e mistura de
biodiesel, além da coordenação da Frente Parlamentar do Etanol. O de Daniela Reinehr tem 106
palavras e omite o bloco "Forma de argumentar", o que está de acordo com o prompt, porque ela tem um
único turno de fala. Um item do perfil de Arnaldo Jardim parece inferência sem apoio nas
falas: "Relaciona seus pleitos a sua base eleitoral".

Os dois atores têm só uma audiência de treino, então os perfis são curtos por falta de material. O
mesmo vale para 79 dos 264 atores da rodada completa.

## Múltipla escolha

Para uma pergunta de teste (audiência de 17/04/2024, regulamentação da reforma tributária), o prompt
foi remontado nas três condições e o primeiro token gerado foi inspecionado:

| Condição | Resposta gulosa | Massa das letras, média no teste |
|---|---|---|
| 0, sem perfil | `A` | 0,997 |
| 1, perfil | "Nenhuma das alternativas. O material fornecido não contém informações sobre a audiência de 17/04/2024 ou sobre a regulamentação da reforma tributária." | 0,26 |
| 2, perfil e trechos | "Nenhuma das alternativas." | 0,00015 |

A massa das letras é a soma das probabilidades de A, B, C e D no primeiro token. O system prompt diz
que o material da conversa é a única fonte sobre a pessoa; como esse material não trata da
audiência de teste, o modelo recusa escolher, e recusa mais quanto mais material recebe. A leitura
por letra sempre escolhe entre A e D, então, nas condições 1 a 3, a letra escolhida sai de
probabilidades da ordem de 0,01%.

A amostra também é pequena demais para qualquer conclusão sobre acerto: 2 perguntas de validação e
3 de teste, com outras 2 descartadas por falta de distratores. Os três valores de k empataram, e
k = 3 foi escolhido por ser o primeiro da grade.

## Guidance e geração aberta

As falas geradas só com o prompt negativo da guidance (`negative_speech`) foram todas recusas, como
"Como não foi fornecido material com falas anteriores de Arnaldo Jardim, não possuo a base de dados
necessária para simular seu posicionamento". Nesse caso, com γ > 1, a guidance afasta a resposta da
recusa. O objetivo dela era afastar a resposta da opinião padrão do modelo, o que exige um prompt
negativo que produza uma fala genérica; `ACTOR_SIMULATION.md` já previa essa situação. Na seleção, γ = 1,0 teve o maior acerto, e as abordagens 2 e 3 produziram falas
idênticas.

As falas simuladas são plausíveis e em primeira pessoa, e a conferência da justificativa marcou de 0
a 1 frase sem fundamento por fala. Para Arnaldo Jardim na reforma tributária, o nível de evidência
foi SEM BASE e a saída foi a recusa prevista. Para Daniela Reinehr, no estudo da ANTT sobre a BR-101,
o nível foi ESPECULATIVA e a fala leva a acessibilidade, único tema do perfil dela, para dentro de
uma discussão sobre túneis e contenção de deslizamentos.

## Tempo

| Etapa | Rodada curta | Rodada completa | Estimativa |
|---|---|---|---|
| Perfis | 22 e 48 s por perfil (2,2 mil e 2,7 mil tokens de entrada); geração em torno de 13 tokens/s | 264 atores, 1,72 milhão de tokens de entrada (mediana de 4,6 mil, máximo de 66 mil) | 7 a 9 h |
| Avaliação | 76 s por pergunta de validação, 49 s por pergunta de teste | 81 perguntas de validação e 101 de teste | 3 a 4 h |
| Simulação | cerca de 120 s por pedido | 53 pedidos | 2 h |
| Total | | | 12 a 16 h |

A estimativa usa as velocidades medidas em prompts de 2 a 3 mil tokens. Os perfis de quem fala muito
têm prompts de até 66 mil tokens e devem ser mais lentos que a média.

## Uso

O backend e a geração de perfis podem ser usados como estão. A avaliação por múltipla escolha
precisa de uma de duas mudanças antes da rodada completa:

1. Mudar o pedido em `prompts/actor_simulation/ask_choice.md.j2` para exigir uma letra mesmo sem
   certeza. Isso altera o `prompt_version` gravado nas saídas e precisa constar no artigo.
2. Usar um modelo que não recuse. O Qwen3.8 27B e o Gemma 4 26B-A4B estão sendo baixados, e a
   comparação entre os três deve olhar primeiro o `letter_mass` das condições 1 e 2, depois o tempo.

Se os três modelos recusarem da mesma forma, a mudança 1 é necessária.

## Arquivos

Em `mlx_alternative/runs/gemma4_31b/`: `smoke.json`, `verify.json` e `verify_prefix_cache.json`
(checagens do backend), `actor_profiles/actor_profiles_train.jsonl` (perfis),
`actor_simulation/evaluation.json` e `choice_*.jsonl` (múltipla escolha),
`actor_simulation/simulations.jsonl` e `simulations_summary.json` (geração aberta) e `timings.jsonl`
(tempos por etapa).
