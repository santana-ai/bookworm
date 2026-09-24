# Perfis por ator

Este documento descreve a etapa que transforma as falas de cada ator recorrente (arquivo
`actors_multi_hearing.jsonl`, construído como descrito em `HEARING_ACTORS.md`) em um texto de perfil
escrito por um LLM. O perfil será convertido em embedding numa etapa posterior, fora do escopo deste
passo; aqui o objetivo é só gerar, para cada ator, um texto denso, específico e fundamentado
exclusivamente nas falas.

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

## Configuração

Tudo fica em `configs/actor_profiles.toml`:

- `[input]`: caminho do JSONL de falas por ator e do LDS (o LDS fornece a data e o assunto de cada
  audiência, conferido por SHA-256 como nos demais scripts).
- `[output]`: caminho do JSONL de perfis.
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

- `system_profile.md`: regras do perfil (fundamentação, presidência de sessão, escrita).
- `user_profile.md.j2`: template Jinja2 com as falas; recebe `actor_label` (o nome do ator) e
  `hearings` (cada um com `date`, `assunto` e `turns`, e cada turno com `role` e `text`).

Jinja2 não interpreta `{` e `}` soltos, então chaves dentro das falas não quebram a renderização.
Qualquer edição muda o `prompt_version` gravado nos perfis seguintes. Para iterar, gere poucos atores
contrastantes com `--actors` para um arquivo descartável e compare com a versão anterior.

## Decisões de desenho do prompt

- **Prosa corrida, sem estrutura fixa.** Cabeçalhos e fórmulas repetidas em todos os perfis
  ("Temas: ...", "Posições: ...") criariam texto idêntico entre atores diferentes e aproximariam os
  embeddings por forma, não por conteúdo. O prompt proíbe listas, títulos e frases de abertura ou
  fechamento genéricas, e manda começar pelo conteúdo mais distintivo, o que também protege contra
  truncamento em encoders com janela curta.
- **Fundamentação como restrição explícita.** O system prompt proíbe conhecimento prévio sobre a
  pessoa, partidos e temas, além de proibir completar cargo ou órgão de pessoas citadas; o que a
  pessoa declara sobre si pode entrar. Dados citados nas falas entram como afirmação da pessoa, não
  como fato. Falas escassas ou protocolares geram um perfil de duas a quatro frases, sem preencher
  lacunas.
- **Proibição de conteúdo negativo.** Dizer o que a pessoa "não menciona" injeta termos espúrios no
  embedding (a primeira iteração produziu "não menciona temas de saúde" para um ator que só fez
  saudações); o prompt proíbe descrever ausências.
- **Posição de terceiros não é posição do ator.** Em falas de presidência de sessão a pessoa
  resume e questiona propostas de convidados; a primeira iteração atribuiu a um deputado a proposta
  de duplicação de rodovia que era da concessionária convidada. O prompt manda atribuir ao ator
  somente o que ele mesmo defende.
- **Turnos de presidência marcados, não removidos.** A construção do arquivo de entrada já corta os
  turnos de presidência com menos de 50 palavras (`HEARING_ACTORS.md`); os que restam misturam
  condução e opinião. O template marca esses turnos com `[presidência da sessão]` e o system prompt
  manda ignorar a condução e aproveitar só o que é posição.
- **Data e assunto por audiência.** Cada bloco de falas vem com a data da matéria e o `assunto` do
  LDS, o que permite ao modelo ancorar mudanças de posição no tempo ("em maio de 2024 passou a
  defender") e desfazer referências vagas ("este projeto de lei").
- **Comprimento proporcional ao material**, com teto rígido de 800 palavras. O teto alto faz o perfil
  de quem fala muito carregar o máximo de informação distintiva (posições, propostas, alvos, números,
  datas); a proporcionalidade impede que ator com duas falas protocolares ganhe perfil inflado; e a
  ordem por distintividade limita a perda se o encoder da etapa de embedding truncar o texto.

## Limitações

- O perfil é texto gerado por modelo, e a fundamentação nas falas depende de o modelo seguir a
  instrução. A inspeção da amostra encontrou e corrigiu vazamentos (ver acima), mas não há
  verificação automática de que cada frase do perfil tem apoio nas falas.
- O script não compara o tamanho do prompt com a janela de contexto do modelo: o prompt vai inteiro
  para o `generate`, sem corte. O `input_tokens` de cada linha registra o tamanho real, para conferir
  contra a janela do modelo usado.
