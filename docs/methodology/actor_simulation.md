# Simulação de atores a partir do perfil

Desenho e implementação do uso do perfil textual de cada ator ([`actor_profiles.md`](actor_profiles.md)) no system
prompt de outro modelo, de modo que ele responda como essa pessoa responderia, tanto sobre temas
que ela já tratou quanto sobre temas novos. As abordagens abaixo são comparadas com o mesmo
protocolo de avaliação. O código está em `experiments.actors.simulation` (partes comuns),
`experiments.actors.evaluate_simulation` (avaliação por múltipla escolha) e `experiments.actors.simulate`
(geração aberta); como rodar, o formato das saídas e as escolhas de implementação que este desenho
deixava em aberto estão no fim do documento.

## O perfil para simulação

O prompt anterior (`prompts/actor_profile/system_profile_old.md`) foi escrito para gerar um texto
que seria convertido em embedding: prosa corrida, começando pelo conteúdo mais distintivo (proteção
contra truncamento do encoder), preferindo posição concreta a rótulo abstrato e cortando o que
serviria para qualquer participante. Para orientar um modelo em tema novo, essas regras tiram do
perfil justamente o que se aplica fora dos temas já tratados: os critérios que a pessoa usa para
decidir.

O prompt atual (`prompts/actor_profile/system_profile.md`, lido pela config) mantém as seções
"Fonte única" e "Condução e cortesia" e organiza a saída em até quatro blocos com título fixo:

- `## Posições`: o que a pessoa defende, critica, propõe e cobra, e de quem, com o motivo nas
  palavras dela quando ela o declara. Sem data, exceto quando o item registra mudança de posição
  entre audiências. Responde às perguntas sobre temas que ela já tratou.
- `## Critérios e valores`: os princípios que a pessoa enuncia de forma geral e os motivos que ela
  usa para justificar mais de uma posição, cada um ligado às posições que justifica. Valor deduzido
  de uma posição é proibido. Orienta a resposta em tema novo.
- `## Alinhamentos declarados`: o que a pessoa diz sobre si e sobre o próprio lado (governo,
  oposição, bancada, frente, categoria). Nada deduzido do partido, das posições ou de elogios.
- `## Forma de argumentar`: comportamento observável (tipo de apoio usado, como trata convidados),
  só se aparecer em mais de um turno e nenhuma fala o contradisser; tom que muda entre audiências é
  registrado por audiência. Descrito em poucas frases; o estilo propriamente dito é passado ao
  modelo por trechos reais de fala (ver abaixo), porque estilo só descrito tende a virar caricatura.

Um bloco sem apoio nas falas é omitido. O teto de 800 palavras vale para a soma dos blocos. As
decisões de desenho estão em [`actor_profiles.md`](actor_profiles.md). A troca de prompt foi decidida pela leitura de
poucos perfis gerados durante o desenvolvimento, não por uma medida da avaliação abaixo.

## Componentes

- **Perfil:** fixo por pessoa, vai no system prompt.
- **RAG:** para cada pergunta, recupera os k trechos de fala da pessoa mais próximos do tema.
  Resolve a perda de detalhe do perfil, que comprime todas as falas em no máximo 800 palavras.
  Busca só em turnos de audiências de `train` (senão recupera a própria fala da audiência avaliada),
  com o encoder Serafim já usado na UDV e restrita aos turnos da pessoa (filtro de falante). Os
  trechos vão na mensagem do usuário, junto com a pergunta, e não no system prompt: mudam a cada
  pergunta enquanto o perfil é fixo, são material de consulta e não regra de comportamento (muitos
  têm frases no imperativo), ficam perto da pergunta a que se referem, e o system prompt fixo por
  pessoa permite reaproveitar o cálculo do prefixo entre perguntas. Essa escolha não foi medida.
- **Classifier-free guidance (CFG):** resolve a tendência do modelo de responder com a própria
  opinião padrão e com o que já sabe da pessoa pelo nome, mesmo com o perfil no prompt. Combina a
  distribuição com e sem o material da pessoa:

  ```
  logp_final = logp_negativo + γ · (logp_completo − logp_negativo)
  ```

  Com γ = 1 o resultado é o do prompt completo; com γ > 1 a resposta vai mais na direção que o
  material empurra. O prompt negativo é o mesmo prompt (instruções, nome, pergunta) sem o material
  cujo efeito se quer amplificar: sem perfil e sem trechos amplifica os dois; só sem o perfil
  amplifica o que o perfil acrescenta além dos trechos. A referência é Sanchez et al. 2023, "Stay on
  topic with Classifier-Free Guidance" (arXiv 2306.17806). A CFG entra só na múltipla escolha, na
  condição 3 (ver Avaliação); o motivo de a geração aberta não usá-la está na seção seguinte.

## As abordagens

1. **Só o perfil:** perfil no system prompt.
2. **Perfil + RAG:** perfil no system prompt, trechos recuperados e pergunta na mensagem do
   usuário.

Na múltipla escolha, as duas são as condições 1 e 2, e a condição 3 combina a 2 com o baseline por
CFG. As abordagens são comparadas com um baseline sem perfil (condição 0 da avaliação). O partido
fica fora do prompt em todas: com ele, o modelo preenche o que falta com o estereótipo do partido.

A geração aberta tinha uma terceira abordagem, a 2 gerada com CFG a cada token, com o prompt da
condição 0 como negativo. Ela saiu depois do piloto de 27/09/2026 (Arnaldo Jardim e Daniela
Reinehr, três modelos, três pedidos de audiências de `test` por modelo), por dois resultados:

- As falas geradas só com o prompt negativo foram recusas nos três modelos. A CFG supõe que o
  prompt negativo produza a opinião padrão do modelo sobre o tema; com uma recusa no lugar, a
  guidance afasta a fala da recusa, e a opinião padrão fica sem correção.
- γ era o escolhido pelo acerto da múltipla escolha, em que a CFG age sobre um único token, e no
  piloto cada modelo o escolheu com 2 perguntas de validação. Com γ = 3,0, escolhido para um dos
  modelos, a fala da abordagem 3 perdeu o sentido ("Presidente quero fazer uma pergunta lige pont
  Caro Ciocchi do ponto de vista mudanças dias depois ons modific regras despacho [...]"), e a
  escolha por acerto não detecta essa degradação.

A decisão veio da leitura de saídas geradas para audiências de `test`, sem uso das UDVs nem do
acerto, e o artigo precisa declarar isso. A fala com o material da condição 0 continua sendo gerada
(ver "Fala") para verificar, na rodada completa, se a recusa se repete.

## Template de simulação

Todas as chamadas partem do mesmo system prompt e as condições da avaliação diferem só no material
incluído (perfil, exemplos, trechos), porque, se as instruções mudassem junto, a diferença entre
condições vizinhas e o prompt negativo da CFG mediriam também o efeito da troca de instrução. As
chamadas são quatro: a múltipla escolha, que é a da avaliação, o nível de evidência e, na geração
aberta, a fala e a justificação, uma separada da outra.

### System prompt

```
[system]
Você simula como {nome}, {papel}, se posiciona em audiências públicas da
Câmara dos Deputados. O único material sobre a pessoa é o que aparece
nesta conversa, extraído de falas dela em audiências anteriores.

- Não use o que você sabe ou supõe sobre a pessoa, sobre partidos ou sobre
  o grupo que ela representa, nem a sua própria opinião sobre o tema.
- Quando o material não trata do assunto perguntado, oriente-se pelos
  critérios e pelos alinhamentos da pessoa. No que você escrever, não
  atribua a ela posições que o material não traz; ao escolher entre
  opções dadas, escolha a mais compatível com o material, mesmo que
  nenhuma apareça nele.
- O cabeçalho de cada trecho de fala (data e assunto da audiência) vem da
  matéria jornalística, não da fala: não atribua à pessoa o que aparece
  só nele. Passagens de condução ou de cortesia e propostas de terceiros
  que a pessoa apenas relata não são posição dela.

PERFIL
## Posições
P1. {item}
P2. {item}
## Critérios e valores
C1. {item}
...

EXEMPLOS DE COMO A PESSOA FALA
[E1] {data} · {assunto da audiência de origem}
"{trecho}"
[E2] {data} · {assunto da audiência de origem}
"{trecho}"
```

- `{nome}`: o campo `actor` do perfil.
- `{papel}`: na múltipla escolha, o `actor.role` da UDV avaliada, que é o `cargo` registrado na
  matéria da audiência de teste, sem o parêntese de partido e UF: "Deputado (PSD-GO) e relator do
  Projeto de Lei 2357/23" vira "Deputado e relator do Projeto de Lei 2357/23", e siglas de órgão
  entre parênteses, como "(Capes)", ficam. O cargo de deputado vem nesse formato ("Deputado
  (PT-SP)"), e sem a limpeza o partido entraria no prompt de todas as condições. Quando o cargo traz
  a função da pessoa na própria audiência de teste (relator, presidente de comissão), essa informação
  entra no acerto da condição 0 e se cancela na diferença entre condições. Na geração aberta, o
  papel é o cargo da UDV de treino mais recente ligada à pessoa, com a mesma limpeza; sem UDV
  ligada, a frase fica só com o nome.
- `PERFIL`: o campo `profile`, com o "- " de cada item trocado por um id formado pela inicial do
  bloco e pela ordem do item nele (P em Posições, C em Critérios e valores, A em Alinhamentos
  declarados, F em Forma de argumentar). Os ids são o que a justificação cita. Na condição 0 o
  perfil é trocado por "Nenhum." e as instruções ficam como estão.
- `EXEMPLOS`: só na geração aberta. São dois trechos fixos por pessoa, de 60 a 150 palavras em
  frases inteiras, tirados de turnos de fala (não de presidência) de duas audiências de treino
  diferentes, a partir da segunda frase do turno, porque os cumprimentos se concentram na primeira,
  e sorteados com a semente da configuração. Com uma só audiência de treino, os dois vêm dela; sem
  turno que sirva, a seção sai. Os exemplos ficam fora da múltipla escolha porque ali a resposta é
  uma letra, em que o estilo não pesa, e porque são fala real com posições: na condição 1
  funcionariam como trechos não escolhidos pelo assunto, e a diferença entre as condições 0 e 1
  deixaria de medir só o perfil. A avaliação, portanto, não mede o efeito dos exemplos, que só
  aparece na leitura das falas geradas. As regras de escolha não foram medidas.

### Mensagem do usuário

```
[user]
AUDIÊNCIA DE {data}, SOBRE: {assunto}

TRECHOS DE FALAS ANTERIORES DE {nome}
[T1] {data} · {assunto da audiência de origem}
"{trecho}"
...
[Tk] {data} · {assunto da audiência de origem}
"{trecho}"

{pedido}
```

Na múltipla escolha, `{data}` e `{assunto}` são os da audiência de teste; na geração aberta, o
assunto é o tema pedido e a data é a do pedido. As datas seguem o formato do template do perfil. A
seção de trechos só existe nas condições 2 e 3 e na abordagem 2:

- a consulta é `{assunto}`, e a busca é por frase, entre as frases dos turnos da pessoa nas
  audiências de `train`, com o Serafim da UDV;
- entram as k frases de maior cosseno, no máximo uma por turno, e cada trecho mostra a frase com a
  anterior e a seguinte do mesmo turno;
- os trechos vão em ordem de data, para que uma mudança de posição apareça na sequência em que
  ocorreu, e são numerados depois de ordenados; o modelo não vê a similaridade;
- trecho de turno de presidência leva a marca `[presidência da sessão]`, como no template do perfil.

k sai de `validation`; as demais escolhas não foram medidas.

### Pedidos

**Múltipla escolha** (avaliação, condições 0 a 3):

```
Uma destas afirmações foi feita por {nome} nesta audiência, que não aparece no material desta conversa. Qual é a mais provável?
A) {proposição}
B) {proposição}
C) {proposição}
D) {proposição}
Responda só com a letra, mesmo sem certeza.
```

A resposta é lida no primeiro token gerado pelo assistente, pelo softmax dos logits dos quatro
tokens de letra, cujos ids são conferidos no tokenizer antes da rodada (cada letra precisa ser um
token só).

A redação anterior, "Qual destas afirmações {nome} fez nesta audiência? [...] Responda só com a
letra.", pedia um fato que o material não contém por construção, já que o system prompt diz que o
material da conversa é a única fonte sobre a pessoa e a audiência avaliada nunca está nele. Num
piloto com dois atores (Arnaldo Jardim e Daniela Reinehr) e três modelos, dois deles responderam
com recusas ("Nenhuma das alternativas. O material fornecido não contém informações sobre a
audiência [...]") nas condições com material, e a soma das probabilidades das letras caiu. A
redação atual declara que uma das opções é da pessoa e que a audiência não está no material, e
pede a letra mesmo sem certeza. As três informações são iguais em todas as condições. O efeito da
troca ainda não foi medido.

A mesma contradição existia no system prompt, na regra "sem atribuir a ela posições que o material
não traz": na múltipla escolha nenhuma opção aparece no material, e escolher qualquer uma contraria
a regra lida ao pé da letra. No piloto, a soma das probabilidades das letras nos dois modelos que
recusaram foi de 0,997 e 1,000 na condição 0 e caiu conforme o material aumentou (Gemma 4 31B: 0,263
na condição 1 e 0,00015 na 2), o que é compatível com essa leitura. A regra passou a valer para o que
o modelo escreve, onde impede que a fala e a justificativa tragam posições inventadas, e ganhou uma
instrução para a escolha entre opções dadas: a mais compatível com o material, mesmo que nenhuma
apareça nele. Como o system prompt é o mesmo em todas as chamadas, a troca também muda a geração
aberta, e não foi medida em nenhuma das duas.

**Nível de evidência** (condições 1 a 3 e abordagens 1 e 2):

```
Com o material desta conversa, classifique a base para estimar a posição
de {nome} sobre esse assunto:
DIRETA: um item de Posições ou um trecho de fala trata desse mesmo assunto.
INDIRETA: nenhum trata desse assunto, mas há posições ou trechos sobre
assunto vizinho (a mesma política, o mesmo setor, o mesmo órgão).
ESPECULATIVA: só itens de Critérios e valores ou de Alinhamentos
declarados se aplicam.
SEM BASE: nada no material se aplica.
Responda só com o rótulo.
```

Na avaliação, esta chamada não leva as opções, para que o nível não dependa dos distratores, e roda
em cada condição com o material dela. Na geração aberta ela vem antes da fala, e com SEM BASE a fala
não é gerada: a saída é "o material não permite estimar". A recusa precisa ser uma saída prevista,
senão o modelo inventa uma posição, e fica numa chamada própria, antes da fala, porque o rótulo
decide se a fala é gerada.

**Fala** (geração aberta, abordagens 1 e 2):

```
Escreva, em primeira pessoa, a fala de {nome} nesta audiência sobre esse
assunto, com no máximo 200 palavras, sem cumprimentos nem agradecimentos
e sem os ids do material. Escreva só a fala.
```

Cada pedido gera também a fala com as mensagens na forma da condição 0: perfil "Nenhum.", sem
exemplos e sem trechos, e sem o pedido do nível antes. É o prompt negativo que a CFG usaria, e a
fala serve para verificar se ele produz uma recusa, como no piloto (ver "As abordagens"). O teto de
200 palavras é uma escolha, não uma medida.

**Justificação** (geração aberta, abordagens 1 e 2): a mesma mensagem da fala, com o
pedido abaixo no lugar do pedido da fala.

```
FALA SIMULADA
{fala}

Para cada frase da fala simulada, escreva uma linha JSON:
{"frase": "<frase copiada da fala>", "apoio": ["P2", "T3"], "copias": {"T3": "<parte do trecho T3 copiada literalmente>"}}
Em "apoio", use os ids do perfil (P, C, A, F), dos exemplos (E) e dos
trechos (T); para cada exemplo ou trecho citado, "copias" traz a parte
usada. Frase sem apoio no material leva "apoio": []. Escreva só as linhas.
```

A conferência é feita no código, frase a frase: "frase" precisa casar com a fala simulada, e cada
cópia com o exemplo ou trecho do mesmo id, pela regra de casamento de citação da UDV; cada id
precisa existir no material da chamada. Contam como sem fundamento a frase com "apoio" vazio, com id
inexistente ou com cópia não encontrada, e a frase da fala que não aparece em nenhuma linha. A
conferência mostra que o item citado existe e que a cópia é literal, mas não que ele sustenta a
frase nem que o modelo o usou, porque a justificação é escrita depois da fala. Sustentação de
sentido exige leitura humana e fica fora da conferência.

Decodificação gulosa em todas as chamadas; a CFG entra só na múltipla escolha, pela combinação dos
log-probs das condições 0 e 2 (ver Avaliação).

## Avaliação

### Pergunta de múltipla escolha

Cada UDV de teste ligada a um ator com perfil vira uma pergunta: nome e papel do ator, data e
assunto da audiência de teste, e 4 proposições, a do ator e 3 de outros atores da mesma audiência,
com o pedido de múltipla escolha acima. O acaso é 25%. Pelo relatório de
`experiments.actors.filter_speeches` (`artifacts/actor_profiles/train_speeches_stats_udv_v2.json`; o mesmo
número em `train_speeches_stats.json`, da rodada `udv_v1`), são 106 das 359 UDVs de teste, de 48
atores, em 27 das 30 audiências de teste.

A resposta aberta não serve como medida principal porque o assunto da audiência já carrega a
posição central: na audiência 1 (teste), o assunto é "Acusações de censura contra Alexandre de
Moraes por exigir bloqueio de contas na rede social X" e a UDV `udv-1-0-0` é "Acusou Alexandre de
Moraes de censura ao solicitar o bloqueio de contas na rede social X". Um juiz daria crédito a quem
repetisse o assunto. Na múltipla escolha o assunto é igual para as 4 opções.

A resposta é lida pelos logits dos tokens A/B/C/D no primeiro passo de geração, sem gerar texto,
o que dá a probabilidade de cada opção. Cada pergunta roda nas 4 rotações da ordem das opções, com
média, para cancelar o viés de posição.

### Condições

| Condição | O que o modelo recebe | O que mede |
|---|---|---|
| 0. Sem perfil (baseline) | nome + papel | acerto só com nome e papel, sob a mesma instrução das outras condições |
| 1. Perfil | 0 + perfil do ator | ganho do perfil sobre o baseline |
| 2. Perfil + RAG | 1 + trechos de treino do ator recuperados pelo assunto | ganho dos trechos sobre o perfil |
| 3. Perfil + RAG + CFG | 2, combinada com a 0 via γ | ganho da guidance sobre a 2 |

Cada condição acrescenta um componente à anterior, então a diferença entre duas condições vizinhas
é o efeito do componente acrescentado.

Na condição 3, o prompt negativo da CFG é o prompt da condição 0 (sem perfil e sem trechos), então
o que a guidance amplifica é tudo o que o material da pessoa acrescenta ao baseline. Na múltipla
escolha, isso não custa passagens extras: as condições 0 e 2 já calculam os dois log-probs de cada
opção, e a condição 3 é a combinação deles com γ.

### Níveis de evidência

Os níveis DIRETA/INDIRETA/ESPECULATIVA só se sustentam se o acerto na múltipla escolha cair de
DIRETA para ESPECULATIVA. O nível é pedido numa chamada separada e cruzado com o acerto.

### Escolha de γ e de k

γ e k (número de trechos recuperados) são escolhidos nas UDVs das audiências de `validation`, que
não entram nos perfis, e nunca no teste.

## Como rodar

Dentro de `experiments/`:

```
./run_simulation_pipeline.sh
```

O script roda, em ordem, `run_actor_profiles.sh` (perfis só com as audiências de `train`, ver
[`actor_profiles.md`](actor_profiles.md)), `experiments.actors.evaluate_simulation` e `experiments.actors.simulate`, todos com o
modelo da variável `MODEL` (o mesmo padrão de `run_actor_profiles.sh`, trocável por
`MODEL=<modelo> ./run_simulation_pipeline.sh`). O padrão, `meta-llama/Llama-3.3-70B-Instruct`, é só um
exemplo e não é o modelo da rodada versionada, que usou `mlx-community/Qwen3.8-27B-8bit` pelo backend MLX. Os argumentos passados ao script vão só para o
gerador de perfis (`--actors`, `--limit`); as duas etapas seguintes usam todos os atores com linha
em `artifacts/actor_profiles/actor_profiles_train.jsonl`. O script para no primeiro erro, inclusive
quando um perfil falha, e rodar de novo retoma de onde parou. Uma falha que se repete, como um
perfil que chega a `max_output_tokens` (a semente é fixada antes de cada geração, então a resposta
se repete), para o script de novo no mesmo ator: tire o ator com `--actors` ou aumente
`max_output_tokens` em `configs/actor_profiles.toml`.

As etapas também rodam separadas:

```
uv run python -m experiments.actors.evaluate_simulation --model <modelo> [--actors ...] [--config ...]
uv run python -m experiments.actors.simulate --model <modelo> [--actors ...] [--requests pedidos.jsonl] \
  [--k N]
```

`experiments.actors.evaluate_simulation` monta as perguntas de `validation` e de `test` e confere, antes
de carregar o modelo, que nenhum perfil nem as falas de treino dos atores contêm audiência desses
dois splits. Em `validation`, mede as condições 0 e 1 e a condição 2 para cada k de `k_grid`,
escolhe k e depois γ (condição 3, em `guidance_grid`); em seguida roda `test` só com o k e o γ
escolhidos, junto com os níveis de evidência.

`experiments.actors.simulate` gera a fala das abordagens 1 e 2 com o k gravado em `evaluation.json`, ou
com `--k`. Sem `--requests`, faz um pedido por par (ator,
audiência de `requests_split`) com UDV ligada, com a data e o assunto da audiência. Com
`--requests`, lê um JSONL com `actor`, `date` (DD/MM/AAAA) e `topic`, que é o caminho para temas
novos. Cada pedido gera também `baseline_speech`, a fala feita com o material da condição 0.
Trocar k depois refaz todas as linhas de `simulations.jsonl`.

Para uma rodada curta, os atores precisam ter perguntas em `validation` e em `test`, porque k e γ
são escolhidos em `validation`; sem pergunta em um dos dois splits, a avaliação para com erro antes
de carregar o modelo. Arnaldo Jardim e Daniela Reinehr têm perguntas nos dois. Uma rodada com
`--actors` reescreve as saídas só com esses atores; para não sobrescrever uma rodada completa, use
uma cópia da config com outra pasta em `[output]` e passe `--config`.

Tudo fica em `configs/actor_simulation.toml`:

- `[input]`: perfis, falas de treino (recuperação e exemplos), falas sem filtro (dono do turno de
  evidência de cada UDV), UDVs, manifesto do split e LDS (data e assunto de cada audiência,
  conferido por SHA-256). O encoder é o de `configs/udv.toml`.
- `[output]`: pasta das saídas.
- `[prompts]`: pasta dos templates Jinja2 (`prompts/actor_simulation/`: system prompt, mensagem do
  usuário e os quatro pedidos). Editar um template muda o `prompt_version` gravado nas saídas.
- `[model]`: `name` (vazio por padrão), `device_map` e o teto de tokens de cada chamada gerada
  (nível, fala e justificação).
- `[roles]`: partidos retirados do papel.
- `[evaluation]`: split de escolha e split de avaliação, grades de k e de γ, número de
  reamostragens do bootstrap e semente.
- `[generation]`: split dos pedidos padrão, número e tamanho dos exemplos e semente do sorteio.

## Formato das saídas

Em `artifacts/actor_simulation/`:

- `choice_validation.jsonl` e `choice_test.jsonl`: uma linha por pergunta, com a UDV avaliada
  (`udv_id`, `hearing_id`, `tier`), o ator, `cargo` (como está na UDV), `role` (sem partido), data,
  assunto, `options`, `option_udv_ids` e `answer` (posição da proposição do ator em `options`).
  `logprobs` guarda, por condição, os log-probs das quatro letras (log-softmax sobre o vocabulário
  inteiro) em 4 listas, uma por rotação, já na ordem de `options`; a condição 2 tem uma entrada por
  k. `excerpts` registra a origem de cada trecho recuperado (audiência, turno, frase e cosseno) e,
  em `test`, `levels` traz o nível das condições 1 e 2 com a resposta crua.
- `evaluation.json`: modelo, versão dos prompts, ids dos tokens de letra, encoder, SHA-256 das
  entradas e as regras usadas. Em `selection`, as contagens de `validation`, o acerto das
  condições 0 e 1, o acerto da condição 2 por k e da condição 3 por γ, e o k e o γ escolhidos. Em
  `evaluation`, as contagens de `test`, o acerto e a probabilidade média da opção certa por
  condição, a diferença de acerto entre condições vizinhas com intervalo de 95% por bootstrap
  pareado de audiências inteiras (as perguntas de uma audiência compartilham assunto e falantes) e
  o acerto por nível de evidência. As contagens (`counts`) trazem UDVs ligadas, perguntas, ids das
  descartadas por falta de distratores, atores, audiências, perguntas por `tier` e quantos papéis
  perderam o partido.
- `letter_mass`, nas duas seções de `evaluation.json`: a leitura por letra sempre escolhe entre A,
  B, C e D, mesmo quando o modelo ia começar a resposta por outro token (um preâmbulo, uma
  marcação). A soma das probabilidades das quatro letras no vocabulário inteiro, com média e
  mínimo por condição, mostra quanto da decisão do modelo essa leitura captura.
- `simulations.jsonl`: uma linha por pedido, com papel, exemplos e trechos usados (com o texto),
  `baseline_speech` e, por abordagem, o nível, a saída (a fala ou "o material não permite
  estimar"), a fala e a justificação cruas (`truncated` quando chegam ao teto de tokens) e `check`,
  a conferência frase a frase.
- `simulations_summary.json`: por abordagem, a distribuição dos níveis, as recusas, as falas e
  justificações truncadas, as frases, as frases sem fundamento e os objetos JSON inválidos ou que
  não puderam ser lidos.

## Retomada

As linhas são gravadas uma a uma. Ao rodar de novo, uma pergunta ou um pedido é reaproveitado
quando a impressão digital coincide: modelo, versão dos prompts, tetos de tokens, encoder, arquivo
de falas de treino, texto do perfil do ator e a própria pergunta (opções e papel) ou o próprio
pedido (papel, exemplos e k). Acrescentar perfis de outros atores não refaz as linhas já
calculadas. No fim, o JSONL é reescrito só com as linhas da rodada atual. Uma linha quebrada por
interrupção é ignorada, com aviso no log. Não rode duas instâncias sobre a mesma pasta de saída.

## Escolhas de implementação não medidas

O desenho acima deixava estas escolhas em aberto. Nenhuma delas foi medida:

- **Distratores.** Proposições de 3 outros atores da mesma audiência, distintos pelo nome
  normalizado da UDV, sem as UDVs cujo turno de evidência é da própria pessoa. Um texto que a
  matéria também atribui à pessoa nessa audiência não entra, e um texto repetido entre outros
  atores entra uma vez só, para que as 4 opções sejam distintas: sem essa regra, algumas perguntas
  mostravam a mesma proposição em duas letras. Atores e proposições são sorteados com a semente e o
  id da UDV; pergunta sem 3 outros atores é descartada e contada.
- **Ordem das opções.** Sorteada, e as 4 rotações são cíclicas, de modo que cada opção passa uma
  vez por cada letra.
- **Pontuação.** Em cada rotação, as probabilidades das quatro letras são renormalizadas entre si;
  a probabilidade de cada opção é a média das 4 rotações, e a resposta é a opção de maior
  probabilidade.
- **CFG na múltipla escolha.** A combinação usa os log-probs das letras sobre o vocabulário
  inteiro, com a mesma fórmula do `transformers`, antes da renormalização entre as letras.
- **Escolha de k e γ.** k pelo acerto da condição 2 em `validation`; γ pelo acerto da condição 3
  com esse k, para que a diferença entre as condições 2 e 3 meça só a guidance. O empate vai para a
  maior probabilidade média da opção certa e depois para o primeiro valor da grade. As grades da
  config são um ponto de partida.
- **Níveis de evidência.** Só em `test` e só com o k escolhido; a condição 3 tem o material da
  condição 2 e usa o nível dela. O rótulo é o primeiro de DIRETA, INDIRETA, ESPECULATIVA e SEM BASE
  que abre a resposta; resposta sem rótulo conta como `unparsed`.
- **Papel.** A limpeza usa a lista de partidos de `[roles]`, tirada dos cargos de deputados em
  `udv_v1`, com ou sem UF no parêntese.
- **Trechos.** A busca é entre as frases de pelo menos 4 palavras (o filtro da UDV), e a vizinhança
  de cada frase vem da lista completa de frases do turno, sem as marcações de cena. Um ator com
  menos turnos do que k recebe menos trechos.
- **Exemplos.** Começam na segunda frase do turno e acumulam frases inteiras até 150 palavras; o
  turno só entra se chegar a 60. Quando menos de duas audiências de treino têm turno que sirva, os
  exemplos saem do que houver, e pode sobrar um só.
- **Pedidos padrão da geração aberta.** Um por par (ator, audiência de `test`) com UDV ligada.
- **Conferência da justificação.** Os objetos JSON são lidos em qualquer ponto da resposta, e não
  só um por linha, porque modelos escrevem o objeto em várias linhas ou dentro de blocos de código,
  e o formato errado apareceria como falta de fundamento. Uma frase da fala casa com o objeto cuja
  "frase" a contém inteira (sem a pontuação final e sem aspas iniciais), e só tem fundamento se
  todos os objetos casados estão sem problema. A cópia é conferida sem as aspas das pontas, pela
  regra de prefixo da UDV: precisa ter pelo menos 6 palavras, e só as 6 a 10 primeiras são
  comparadas com o exemplo ou trecho, então o resto da cópia não é conferido. Cada problema tem um
  código (`empty_support`, `unknown_id`, `copy_missing`, `copy_too_short`, `copy_not_found`).

## Pendências

A rodada completa com `mlx-community/Qwen3.8-27B-8bit` está em `artifacts/mlx_runs/qwen38_27b/`
(relatório em `relatorio_rodada_completa.md`, resumo na seção 6.3 do [relatório](../report.md)). Ela
foi feita sobre `udv_v1`. As configurações de `configs/` apontam agora para `udv_v2`; sobre ela, só as
perguntas e os pedidos foram reconstruídos sem modelo (`artifacts/actor_simulation/udv_v2_dry_run_evaluation.json`
e `udv_v2_dry_run_simulation.json`): no teste, 98 das 101 perguntas são idênticas às de `udv_v1` e 3
mudam só o nível; os pedidos de simulação são idênticos (`question_overlap` e
`simulation_requests_dry_run` de `artifacts/udv/udv_v2_downstream_report.json`). O que continua em
aberto:

- A rodada do modelo sobre `udv_v2`, com o comando de [`mlx_backend.md`](mlx_backend.md), seção "Rodada sobre
  `udv_v2`".

- As contagens de perguntas de `validation` e de `test` (UDVs ligadas, perguntas, descartadas por
  falta de 3 outros atores) saem em `counts` de `evaluation.json` e dependem do conjunto de perfis
  da rodada; citar só os números da rodada completa.
- Contar as recusas entre as `baseline_speech` da rodada completa, que é o número do artigo para a
  retirada da CFG da geração aberta. A recusa é texto livre, então a contagem é por leitura: um
  script gera o arquivo com um campo de julgamento vazio por pedido, e o campo é preenchido à mão.
- Confirmar, em script versionado, o ganho do filtro de falante na recuperação com Serafim.
- Opcional: medir a troca de prompt na condição 1, em `validation`, com perfis gerados por
  `system_profile.md` e por `system_profile_old.md`.
- O Apêndice A do artigo traz cópia literal de `system_profile.md`; qualquer edição do prompt exige
  copiar de novo.
