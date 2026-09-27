# Perfis por ator

Este documento descreve a etapa que transforma as falas de cada ator recorrente (arquivo
`actors_multi_hearing.jsonl`, construído como descrito em `HEARING_ACTORS.md`) em um texto de perfil
escrito por um LLM. O perfil vai no system prompt de outro modelo, que responde como a pessoa
responderia (plano em `ACTOR_SIMULATION.md`); aqui o objetivo é só gerar, para cada ator, um texto
denso, específico e fundamentado exclusivamente nas falas.

## Como rodar

O gerador carrega o modelo com `from_pretrained` do `transformers` (`AutoTokenizer` e
`AutoModelForCausalLM`), uma vez, no início. Cada ator vira uma única geração, com todas as suas
falas no prompt, montado pelo chat template do próprio modelo.

Dentro de `challenge/`:

```
uv sync
uv run python -m utils.generate_actor_profiles --model <modelo>
```

`<modelo>` é o id de um repositório do Hugging Face ou o caminho de uma pasta local com o modelo
salvo. Ele também pode ficar em `name`, na seção `[model]` de `configs/actor_profiles.toml` (vazio
por padrão); sem ele o script para com erro antes de carregar qualquer coisa. Os perfis de todos os
atores do arquivo de entrada vão para `artifacts/actor_profiles/actor_profiles.jsonl`, uma linha por
ator, escrita incrementalmente.

Outras flags (todas opcionais; o padrão vem de `configs/actor_profiles.toml`):

```
uv run python -m utils.generate_actor_profiles \
  --model <modelo> \
  --output artifacts/actor_profiles/actor_profiles.jsonl \
  --actors "Erika Kokay" "Jorge Solla" \
  --limit 10
```

`--actors` seleciona atores pelo nome exato do campo `actor`; `--limit` corta a lista.
`--input` aponta para outro JSONL com o mesmo schema (ex.: o arquivo de falantes de uma audiência) e
`--config` para outro arquivo de configuração.

## Perfis só com as audiências de treino

Um perfil gerado com as 206 audiências já leu as falas das audiências de teste. Se as opiniões
publicadas dessas audiências forem usadas para avaliar o perfil, a avaliação mede o que o perfil
copiou, não o que ele permite estimar. Para esse uso, o perfil é gerado só com as audiências de
`train` do split `temporal_v1`; as de `validation` não entram.

```
./run_actor_profiles.sh
```

O modelo vem da variável `MODEL` do script (um exemplo, trocável por `MODEL=<modelo>
./run_actor_profiles.sh` ou por `--model <modelo>` na linha de comando).

O script roda, em ordem: `utils.download_dataset` (só se o LDS não estiver em `dataset/`),
`utils.build_actor_speeches`, `utils.filter_actor_speeches` e `utils.generate_actor_profiles`. Os
argumentos passados ao script vão para o gerador (`--model`, `--actors`, `--limit`). A saída é
`artifacts/actor_profiles/actor_profiles_train.jsonl`, separada da saída com todas as audiências,
porque a retomada pula atores pelo nome e misturaria as duas versões.

`utils.filter_actor_speeches` lê a seção `[split_filter]` de `configs/actor_profiles.toml`, mantém
de cada ator só as audiências dos splits listados em `splits`, sem alterar os turnos, e descarta o
ator que fica sem nenhuma. O resultado vai para `actors_multi_hearing_train.jsonl` em
`artifacts/cache/` (não versionado, como o arquivo de origem) e as contagens, com o SHA-256 do
manifesto do split e dos dois arquivos, para `artifacts/actor_profiles/train_speeches_stats.json`.
Com `temporal_v1`, dos 301 atores ficam 264, em 139 das 144 audiências de treino, com 4.598 dos
6.323 turnos; 37 atores não falam em nenhuma audiência de treino e 79 ficam com uma só. Os campos
`has_party_header` e `party_uf` continuam com o valor calculado sobre todas as audiências, porque o
arquivo de origem não guarda o partido por audiência.

O mesmo relatório conta o que sobra para avaliar nos splits de `eval_splits` (`test`), que não podem
coincidir com os de `splits`. Uma UDV de teste é ligada ao ator dono do seu turno de evidência
(`hearing_id` e `evidence.speaker_turn`) no arquivo sem filtro; UDV sem turno de evidência fica sem
ligação. Com `udv_v1`, 116 dos 264 atores também falam no teste, e 106 das 359 UDVs de teste se ligam
a 48 deles, em 27 das 30 audiências (17 `quote_found`, 87 `semantic_match_high`, 2
`semantic_match_weak`).

## Configuração

Tudo fica em `configs/actor_profiles.toml`:

- `[input]`: caminho do JSONL de falas por ator e do LDS (o LDS fornece a data e o assunto de cada
  audiência, conferido por SHA-256 como nos demais scripts).
- `[output]`: caminho do JSONL de perfis.
- `[split_filter]`: manifesto do split, splits mantidos (`splits`), splits de avaliação
  (`eval_splits`), arquivo de UDVs e caminhos do arquivo filtrado e das contagens, lidos só por
  `utils.filter_actor_speeches`.
- `[prompts]`: pasta e nomes dos dois arquivos de prompt.
- `[model]`: `name` (id do Hugging Face ou pasta local, vazio por padrão); `device_map`, repassado
  ao `from_pretrained` (com `"auto"`, o `accelerate` distribui o modelo entre as GPUs disponíveis e
  manda para a CPU o que não couber nelas); e os parâmetros de geração `temperature`, `top_p`,
  `max_output_tokens` (vira `max_new_tokens`) e `seed`, fixada antes de cada geração. O `dtype` fica
  no padrão do `transformers`, que é o do checkpoint.

## Formato do output

Uma linha JSON por ator:

- `actor`: nome do ator, igual ao do arquivo de entrada.
- `profile`: o texto do perfil.
- `model`, `prompt_version`: o `name` passado ao `from_pretrained` e o hash (12 hex) dos dois
  arquivos de prompt, para saber com que versão do prompt cada perfil foi gerado.
- `n_statements` (turnos de fala), `n_hearings` e `hearing_ids`.
- `input_tokens` e `output_tokens`: tamanho do prompt e da resposta no tokenizer do modelo.
- `generated_at` (UTC) e `duration_seconds`.

## Retomada e falhas

A escrita é incremental e o arquivo de saída é aberto em modo append. Ao rodar de novo com o mesmo
`--output`, os atores que já têm linha válida são pulados; para regerar um ator, apague a linha dele
(ou o arquivo). Erro em um ator não derruba o processo: a falha é registrada no log, o ator fica de
fora do arquivo (portanto será tentado de novo na próxima rodada) e o resumo final lista os nomes que
falharam. Uma resposta que chega a `max_output_tokens` sem terminar conta como falha. Uma linha
parcial deixada por interrupção abrupta é ignorada na retomada, com aviso no log, mas permanece no
arquivo; remova a linha quebrada antes de consumir o JSONL. Não rode duas instâncias sobre o mesmo
`--output`: a retomada é lida só no início do processo e as escritas podem se intercalar.

## Editar os prompts

Os prompts moram em `prompts/actor_profile/` e podem ser editados sem tocar no código:

- `system_profile.md`: regras do perfil (fundamentação, condução e cortesia, blocos do perfil,
  escrita).
- `system_profile_old.md`: versão anterior, que gerava prosa corrida para ser convertida em
  embedding. Não é lida pelo script; para gerar com ela, `system_profile = "system_profile_old.md"`
  na seção `[prompts]` e um `--output` novo, porque a retomada pula atores pelo nome.
- `user_profile.md.j2`: template Jinja2 com as falas; recebe `actor_label` (o nome do ator) e
  `hearings` (cada um com `date`, `assunto` e `turns`, e cada turno com `role` e `text`).

Jinja2 não interpreta `{` e `}` soltos, então chaves dentro das falas não quebram a renderização.
Qualquer edição muda o `prompt_version` gravado nos perfis seguintes. Para iterar, gere poucos atores
contrastantes com `--actors` para um arquivo descartável e compare com a versão anterior.

## Decisões de desenho do prompt

- **Quatro blocos com título fixo.** `## Posições`, `## Critérios e valores`, `## Alinhamentos
  declarados` e `## Forma de argumentar`, nesta ordem, com itens começando pelo verbo; bloco sem
  apoio nas falas é omitido. As posições orientam o modelo que simula a pessoa em temas que ela já
  tratou; os outros três blocos, em temas novos. Dentro de cada bloco, o que se repete em mais
  audiências vem primeiro.
- **Motivo junto da posição.** Quando a pessoa diz por que defende algo, o item de posição traz o
  motivo com as palavras dela. O bloco de critérios fica com os princípios enunciados de forma geral
  e os motivos que justificam mais de uma posição, e proíbe deduzir valor de posição ("defende o
  SUS" não autoriza "valoriza o papel do Estado").
- **Alinhamento só declarado.** O bloco de alinhamentos contém o que a pessoa diz sobre si (cargo,
  organização, trajetória, cidade) e sobre o próprio lado (governo, oposição, bancada, frente), nunca
  o que se deduziria do partido, das posições ou de quem ela elogia. É o mesmo motivo que deixa o
  partido fora do prompt de simulação.
- **Forma de argumentar observável.** Descreve comportamento (tipo de apoio, como trata
  convidados), sem adjetivo de personalidade. Um traço só entra se aparecer em mais de um turno e
  nenhuma fala o contradisser; uma frase dirigida a uma pessoa ou a um caso não vira traço geral; tom
  que muda entre audiências é registrado por audiência. As três restrições vieram de perfis gerados
  durante o desenvolvimento do prompt, em que "conversa fiada", dita sobre um prefeito, virou traço
  geral, e "identifica problemas estruturais em vez de culpar pessoas" era contrariado por falas de
  outra audiência.
- **Fundamentação como restrição explícita.** O system prompt proíbe conhecimento prévio sobre a
  pessoa, partidos e temas, além de proibir expandir ou abreviar siglas, completar nome, cargo ou
  órgão de pessoas citadas e atribuir data a fato que a fala menciona sem data. Dados citados nas
  falas entram como afirmação da pessoa, não como fato. Falas escassas ou protocolares geram só os
  blocos que elas sustentam, com um ou dois itens cada, sem preencher lacunas.
- **Proibição de conteúdo negativo.** O modelo que simula a pessoa leria "não menciona temas de
  saúde" como posição (a primeira iteração produziu essa frase para um ator que só fez saudações); o
  prompt proíbe descrever ausências.
- **Posição de terceiros não é posição do ator.** Em falas de presidência de sessão a pessoa
  resume e questiona propostas de convidados; a primeira iteração atribuiu a um deputado a proposta
  de duplicação de rodovia que era da concessionária convidada. O prompt manda atribuir ao ator
  somente o que ele mesmo defende.
- **Turnos de presidência marcados, não removidos.** A construção do arquivo de entrada já corta os
  turnos de presidência com menos de 50 palavras (`HEARING_ACTORS.md`); os que restam misturam
  condução e opinião. O template marca esses turnos com `[presidência da sessão]` e o system prompt
  manda ignorar a condução e aproveitar só o que é posição. A regra vale para qualquer turno: pedidos
  de tempo ou inscrição, citação de requerimentos, cumprimentos, agradecimentos e elogios também são
  desconsiderados, porque apareciam nos perfis como se fossem conteúdo.
- **Data e assunto por audiência.** Cada bloco de falas vem com a data da matéria e o `assunto` do
  LDS, o que permite ao modelo ancorar mudanças de posição no tempo ("em 14/05/2024 passou a
  defender") e desfazer referências vagas ("este projeto de lei"). Os itens do perfil não levam
  data, exceto os que registram mudança de posição; nos perfis gerados durante o desenvolvimento do
  prompt, a data por item vinha às vezes com a data de outra audiência. Como o `assunto` vem da matéria e
  não da fala, o system prompt restringe o cabeçalho a situar as falas no tempo e no tema e proíbe
  atribuir à pessoa o que só aparece nele (sem essa regra, o assunto "atos de 8 de janeiro" virou
  posição de um deputado que não citou a data).
- **Comprimento proporcional ao material**, com teto rígido de 800 palavras somando os blocos. O teto
  alto faz o perfil de quem fala muito carregar o máximo de informação (posições, critérios,
  propostas, alvos, números); a proporcionalidade impede que ator com duas falas protocolares ganhe
  perfil inflado.

## Limitações

- O perfil é texto gerado por modelo, e a fundamentação nas falas depende de o modelo seguir a
  instrução. A inspeção da amostra encontrou e corrigiu vazamentos (ver acima), mas não há
  verificação automática de que cada frase do perfil tem apoio nas falas.
- O script não compara o tamanho do prompt com a janela de contexto do modelo: o prompt vai inteiro
  para o `generate`, sem corte. O `input_tokens` de cada linha registra o tamanho real, para conferir
  contra a janela do modelo usado.
