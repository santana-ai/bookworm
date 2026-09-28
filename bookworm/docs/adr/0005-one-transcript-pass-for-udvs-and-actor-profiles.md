# ADR 0005: uma passada pela transcrição para UDVs e perfis de atores, com o perfil conferido contra as UDVs

- Status: aceita
- Data: 2026-09-26
- Código afetado: `src/bookworm/pipeline.py` (novo), `src/bookworm/actors/` (novo),
  `src/bookworm/profiles/` (novo), `src/bookworm/udv/build.py`, `src/bookworm/config.py` e
  `src/bookworm/cli.py`
- Porta, com paridade testada: `challenge/utils/build_actor_speeches.py`,
  `challenge/utils/filter_actor_speeches.py` e `challenge/utils/generate_actor_profiles.py`
- Nota de 2026-09-28: na versão 1.0, `src/bookworm/cli.py` virou o pacote `src/bookworm/cli/`
  (`udv_commands.py` e `profile_commands.py` para os comandos desta decisão), e os scripts de
  `challenge/utils/` estão nos caminhos de [`docs/path_map.md`](../../../docs/path_map.md). A decisão não mudou.

## Contexto

As UDVs e os perfis de atores saem do mesmo texto: os turnos de fala da transcrição, recortados pelo
mesmo padrão de cabeçalho (`O SR.`/`A SRA.`). Até aqui, eram dois caminhos independentes:

- `build-udvs` recorta os turnos de cada audiência, resolve cada participante de
  `metadados.envolvidos` para os seus turnos e liga cada opinião da matéria a um trecho desses turnos;
- `challenge/utils/build_actor_speeches.py` recorta os turnos de novo, agrupa todos os turnos de
  pessoa (não só os dos envolvidos) por nome normalizado, entre audiências, aplica as fusões revisadas
  de nomes e grava um arquivo de falas por ator, que `generate_actor_profiles.py` transforma em um
  perfil escrito por um LLM.

Isso trazia três problemas:

1. A transcrição era lida e segmentada duas vezes, por dois códigos que precisam concordar sobre o que
   é um turno. Uma correção num deles (como a da ADR 0002) não chegava ao outro.
2. Não havia ligação entre uma UDV e o ator do perfil. A UDV guarda o nome do envolvido como a matéria
   o escreve; o perfil guarda o nome de exibição do ator, escolhido entre as grafias dos cabeçalhos.
   Casar os dois depois, por nome, repetiria a resolução de nomes com outro critério.
3. Não havia como saber se um perfil está correto. A UDV é conferida contra a transcrição: cada
   opinião que a matéria atribui a alguém precisa ter um trecho da fala dessa pessoa que a sustente. O
   perfil é outro resumo da mesma fala e não tinha conferência equivalente.

## Decisão

### Uma passada

`bookworm.pipeline` percorre cada audiência uma vez. Para cada audiência, `split_into_turns` roda uma
única vez e os mesmos objetos `Turn` alimentam:

- a resolução de participantes e a construção das UDVs (o código de `bookworm.udv.build` passa a
  aceitar os turnos já recortados, sem mudar nenhum registro);
- a coleta das falas por ator (`bookworm.actors`), com a mesma política do script do `challenge/`:
  papel de presidência pelos nomes de `[speakers].chair_names`, descarte de chaves que não são pessoa,
  de rubricas, de turnos vazios e de turnos de presidência com menos de `chair_min_words` palavras, e
  as fusões de `[merges]`;
- a ligação UDV → ator: para cada participante resolvido, os turnos casados são os mesmos objetos que
  entraram nas falas por ator, então a chave do ator sai do próprio turno, sem nova comparação de
  nomes.

`bookworm build-udvs` ganha a opção `--actors-config`. Sem ela, o comando grava exatamente os mesmos
arquivos de antes. Com ela, grava também, na mesma execução, as falas por ator, a lista de nomes
ambíguos, as contagens e o arquivo de ligações UDV → ator.

### Perfis atrás de uma interface

`bookworm filter-actor-speeches` mantém, no arquivo de falas por ator, só as audiências dos splits
configurados (em `challenge/configs/actor_profiles.toml`, só `train`). A geração de perfis
(`bookworm generate-profiles`) lê esse arquivo filtrado e chama um `ChatClient`, um `Protocol` com um
único método. A implementação com `transformers` fica no extra `profiles`; os testes usam um
cliente falso, sem modelo e sem rede. O comportamento é o de
`challenge/utils/generate_actor_profiles.py` e `filter_actor_speeches.py`, sem alteração: mesmos
prompts, mesma versão de prompt, mesmos parâmetros e mesmo formato de saída. Esta ADR não roda a
geração.

### O perfil conferido contra as UDVs

A conferência do perfil (`bookworm validate-profiles`) é a mesma pergunta que a UDV faz à matéria,
com os papéis trocados: dada uma opinião verificada do ator (uma UDV com evidência na transcrição,
nos níveis configurados), existe uma sentença do perfil que a sustente? Para cada UDV ligada a um ator
com perfil:

- o perfil é segmentado em sentenças pelo mesmo segmentador das falas;
- um `SentenceEncoder` (o mesmo `Protocol` das UDVs, TF-IDF no núcleo, Serafim no extra `embeddings`)
  dá o cosseno entre a proposição da UDV e cada sentença; o score do par é o maior deles, e a sentença
  correspondente é gravada como a evidência no perfil;
- o mesmo score é calculado contra o perfil de todos os outros atores, e a posição do perfil verdadeiro
  nessa ordem dá a identificação do ator (acerto no topo e MRR), uma métrica que não depende de corte
  nem de rótulo humano.

Os pares são separados por split da audiência da UDV:

- `train`: a audiência entrou no prompt; mede cobertura, isto é, se o perfil guardou as posições que a
  matéria registrou;
- `test` (ou `validation`): a audiência não entrou no prompt; mede se o perfil permite estimar a
  posição do ator numa audiência que ele não leu.

A conferência recusa um perfil cujo `hearing_ids` contenha audiência fora dos splits usados na
geração, e nunca mistura os dois grupos de pares num mesmo número.

O julgamento humano segue a regra das UDVs: `bookworm sample-profile-review` sorteia, com semente
fixa, pares estratificados por split e por faixa de score e grava um CSV com o campo de julgamento
vazio (`sustentada`, `compativel`, `contradita`, `sem_relacao`), preenchido à mão.

## Consequências

- Uma única execução de `build-udvs --actors-config` produz tudo o que a geração de perfis precisa, e
  a paridade com os artefatos versionados de `challenge/artifacts/hearing_actors/` e
  `challenge/artifacts/actor_profiles/train_speeches_stats.json` é testada com o marcador `dataset`.
- A geração dos atores e dos perfis não muda: política de turnos, fusões, nome de exibição, filtro por
  split, prompts, parâmetros de geração e formato dos registros são os dos scripts de `challenge/`, e
  os arquivos de falas por ator saem byte a byte iguais aos do script. A ligação com as UDVs fica só no
  arquivo de ligações, que traz o nome de exibição do ator (a chave de junção com os perfis) e a chave
  normalizada.
- Opiniões de envolvidos que não foram resolvidos para nenhum turno (`person_not_resolved`) não têm
  ator e ficam fora da conferência; a contagem delas é gravada no relatório.
- O cosseno máximo mede proximidade de conteúdo, não implicação. Ele serve para ordenar e identificar
  atores e para escolher a amostra humana; a afirmação de que um perfil sustenta uma opinião vem do
  julgamento humano. Um verificador de implicação pode substituir o encoder atrás da mesma interface
  sem mudar o formato dos arquivos.
- Nada disso muda a UDV: `build-udvs` sem `--actors-config` continua reproduzindo `udv_v1` byte a
  byte.
