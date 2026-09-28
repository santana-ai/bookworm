# Comparação dos modelos MLX para perfis e simulação de atores

Data: 27/09/2026. Máquina: Apple M4 Max, 128 GB. Os três modelos rodaram com o mesmo código de
pipeline (`utils/`), trocando só o backend pelo de `mlx_alternative/`, na mesma rodada curta com
Arnaldo Jardim e Daniela Reinehr. Os números de velocidade e memória vêm de
`mlx_alternative/runs/benchmark.json`.

| id | Modelo | Tipo |
|---|---|---|
| `gemma4_31b` | `lmstudio-community/gemma-4-31B-it-MLX-8bit` | denso, 31B |
| `qwen38_27b` | `mlx-community/Qwen3.8-27B-8bit` | denso, 27B, atenção híbrida |
| `gemma4_26b_a4b` | `mlx-community/gemma-4-26b-a4b-it-8bit` | mistura de especialistas, cerca de 4B ativos |

Nenhum dos três atende bem às três partes do pipeline. O Gemma 4 31B escreve os perfis mais fiéis
às regras do prompt, mas recusa a múltipla escolha quando recebe o perfil. O Qwen3.8 27B responde a
múltipla escolha em todas as condições, mas os perfis dele quebram várias regras do prompt. O Gemma
4 26B-A4B é de 5 a 9 vezes mais rápido que os outros dois, responde com o perfil e recusa na maior
parte das vezes quando recebe também os trechos recuperados.

## Backend

Nos três modelos, a leitura das letras coincide com o prefill padrão do `mlx-lm` (diferença de 0,0
nas probabilidades renormalizadas), os 8 passos verificados da geração com guidance coincidem com a
combinação calculada por forwards completos, e a geração com γ = 1 é idêntica à geração sem
guidance. O modo de raciocínio ficou desligado nos três, e as letras A a D são um token cada.

## Múltipla escolha

A avaliação lê a resposta pelas probabilidades de A, B, C e D no primeiro token. A massa das letras
é a soma dessas quatro probabilidades: perto de 1, o modelo está de fato escolhendo uma letra; perto
de 0, ele ia começar a resposta por outra coisa, e a letra escolhida sai de probabilidades muito
pequenas.

| Condição | Gemma 4 31B | Qwen3.8 27B | Gemma 4 26B-A4B |
|---|---|---|---|
| 0, sem perfil | 0,997 | 0,996 | 1,000 |
| 1, perfil | 0,263 | 0,993 | 0,9995 |
| 2, perfil e trechos | 0,00015 | 0,985 | 0,151 |

Médias nas perguntas de teste da rodada curta. Na validação, a condição 2 teve 0,43 no Gemma 4
26B-A4B e 0,98 no Qwen3.8 27B.

Quando a massa é baixa, a resposta gulosa é uma recusa, como "Nenhuma das alternativas. O material
fornecido não contém informações sobre a audiência de 17/04/2024". O system prompt diz que o
material da conversa é a única fonte sobre a pessoa, e esse material nunca trata da audiência de
teste. Os dois Gemma recusam mais quanto mais material recebem; o Qwen escolhe uma letra nas três
condições.

Com 2 perguntas de validação e 3 de teste por modelo, as acurácias da rodada curta não permitem
comparar os modelos.

## Perfis

Os dois perfis de cada modelo foram lidos contra as regras de `prompts/actor_profile/system_profile.md`.
São dois perfis por modelo, de atores com uma audiência de treino cada, então o que segue são
exemplos de desvio, sem taxa medida.

| | Gemma 4 31B | Qwen3.8 27B | Gemma 4 26B-A4B |
|---|---|---|---|
| Palavras (Arnaldo Jardim / Daniela Reinehr) | 316 / 106 | 530 / 212 | 381 / 200 |
| Bloco "Forma de argumentar" para Daniela Reinehr, que tem um só turno | omitido | omitido | incluído |

Desvios encontrados:

- Gemma 4 31B: "Relaciona seus pleitos a sua base eleitoral", que parece inferência sem apoio nas
  falas.
- Qwen3.8 27B: sigla expandida ("SAF (Sustainable Aviation Fuel)"), o que o prompt proíbe;
  especulação entre parênteses ("provavelmente referindo-se à origem paulista"); cortesia registrada
  como traço ("agradecendo a presença do Ministro"); descrição do que a pessoa não fez ("opta por
  não aprofundar o tema"); e uma palavra em espanhol ("Pregunta").
- Gemma 4 26B-A4B: traços de "Forma de argumentar" tirados de um único turno ("Adota um tom de
  acolhimento") ou de uma única frase ("Adota um tom de busca por objetividade"), quando o prompt
  exige que o traço apareça em mais de um turno.

## Geração aberta e guidance

As falas geradas só com o prompt negativo da guidance foram recusas nos três modelos. Com γ > 1, a
guidance afasta a resposta dessa recusa; `ACTOR_SIMULATION.md` já registrava que, nesse caso, ela
deixa de corrigir a opinião padrão do modelo.

O γ escolhido na validação foi 1,0 no Gemma 4 31B, 1,5 no Gemma 4 26B-A4B e 3,0 no Qwen3.8 27B,
cada um a partir de 2 perguntas. Com γ = 3,0, a fala do Qwen na abordagem 3 perdeu o sentido:
"Presidente quero fazer uma pergunta lige pont Caro Ciocchi do ponto de vista mudanças dias depois
ons modific regras despacho [...] security electrical must operate grea". A escolha de γ por acerto,
com poucas perguntas, não detecta esse tipo de degradação.

No Qwen3.8 27B e no Gemma 4 26B-A4B, 2 dos 3 pedidos (Arnaldo Jardim na reforma tributária e
Daniela Reinehr na BR-101) receberam o nível SEM BASE, e a saída foi a recusa prevista. No Gemma 4
31B, só o da reforma tributária recebeu SEM BASE; o de Daniela Reinehr recebeu ESPECULATIVA, e a
fala leva a acessibilidade, único tema do perfil dela, para uma discussão sobre túneis e contenção
de deslizamentos. No pedido de
Arnaldo Jardim sobre transição energética em terminais marítimos, os três geraram falas em primeira
pessoa que retomam as posições do perfil (segurança do sistema elétrico, biocombustíveis).

## Velocidade e memória

| | Gemma 4 31B | Qwen3.8 27B | Gemma 4 26B-A4B |
|---|---|---|---|
| Memória após a carga | 32,6 GB | 28,6 GB | 26,8 GB |
| Pico com o prompt de 66 mil tokens | 50,3 GB | 42,1 GB | 33,9 GB |
| Prefill do prompt de 66 mil tokens | 157 tokens/s (7 min) | 207 tokens/s (5,3 min) | 1.019 tokens/s (1,1 min) |
| Geração com prompt curto | 13,6 tokens/s | 15,9 tokens/s | 73,7 tokens/s |
| Geração com prompt de 66 mil tokens | 10,4 tokens/s | 13,3 tokens/s | 47,8 tokens/s |
| Segundos por pergunta, validação / teste | 76 / 49 | 77 / 49 | 12 / 8 |
| Segundos por pedido de simulação | 123 | 90 | 13 |

O prompt de 66 mil tokens é o de Erika Kokay, o maior dos 264 perfis de treino.

## Tempo projetado da rodada completa

A rodada completa tem 264 perfis (1,72 milhão de tokens de entrada), 81 perguntas de validação, 101
de teste e 53 pedidos de simulação. A projeção usa a velocidade de prefill do prompt longo para toda
a entrada, a velocidade de geração do prompt longo para a saída, e os tempos por pergunta e por
pedido da rodada curta.

| Horas | Gemma 4 31B | Qwen3.8 27B | Gemma 4 26B-A4B |
|---|---|---|---|
| Perfis, com a saída média da rodada curta | 5,2 | 5,3 | 1,1 |
| Perfis, com 800 tokens de saída por perfil | 8,7 | 6,7 | 1,7 |
| Avaliação | 3,1 | 3,1 | 0,5 |
| Simulação | 1,8 | 1,3 | 0,2 |
| Total | 10 a 14 | 10 a 11 | 1,8 a 2,4 |

A saída média da rodada curta foi de 301, 546 e 414 tokens por perfil, para dois atores com pouco
material; atores que falam em várias audiências recebem perfis de até 800 palavras, então a linha
de 800 tokens está mais perto do esperado. A avaliação e a simulação também devem demorar mais que o
projetado, porque perfis maiores deixam os prompts maiores do que os da rodada curta.

Ligar o reuso de prefixo (`backend.prefix_cache`) reduziria o tempo da múltipla escolha por 1,5 a
1,8, mas muda as probabilidades renormalizadas das letras em até 0,05, como registrado no relatório
da rodada curta do Gemma 4 31B.

## Opções

1. Qwen3.8 27B em tudo. A múltipla escolha funciona como desenhada. Os perfis precisam de leitura,
   porque trazem siglas expandidas, especulação e cortesia, e a grade de γ precisa perder os valores
   altos ou ter as falas lidas antes da escolha. Cerca de 10 a 11 h.
2. Perfis com o Gemma 4 31B e múltipla escolha e simulação com o Qwen3.8 27B. Junta os perfis mais
   fiéis ao prompt com a avaliação que funciona. O pipeline aceita perfis de um modelo e simulação
   de outro, porque o arquivo de perfis grava o modelo que os gerou, mas o `run.py` usa hoje um
   único modelo por pasta de rodada e precisaria aceitar os perfis de outra pasta. O artigo precisa
   registrar os dois modelos. Cerca de 10 a 13 h.
3. Gemma 4 26B-A4B em tudo, com uma mudança no pedido de múltipla escolha
   (`prompts/actor_simulation/ask_choice.md.j2`) para exigir uma letra mesmo sem certeza. É a única
   opção que deixa tempo para repetir a rodada, mas a mudança de prompt altera o `prompt_version` e
   não foi testada; o efeito dela na recusa da condição 2 precisa ser medido antes.

## Arquivos

Em `mlx_alternative/runs/`: `benchmark.json` e `benchmark.md` (velocidade, memória e projeções) e,
para cada modelo, `<id>/smoke.json`, `<id>/verify.json`, `<id>/actor_profiles/`,
`<id>/actor_simulation/` e `<id>/timings.jsonl`. O relatório da rodada curta do Gemma 4 31B está em
`gemma4_31b/relatorio_rodada_curta.md`.
