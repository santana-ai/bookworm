# Falas por ator

Esta etapa junta tudo o que cada pessoa disse nas audiências do PublicHearingBR, audiência por
audiência, e liga cada UDV ao ator que a disse. Ela porta `challenge/utils/build_actor_speeches.py`
para a biblioteca e roda na mesma passada pela transcrição que constrói as UDVs
([ADR 0005](adr/0005-one-transcript-pass-for-udvs-and-actor-profiles.md)).

## O problema

Um perfil de ator precisa da fala completa de uma pessoa em todas as audiências de que ela
participou. A matéria não serve para isso: ela cita poucas pessoas e resume o que disseram. A
transcrição tem a fala inteira, mas dividida em turnos, e o nome de quem fala aparece com grafias
diferentes entre audiências (com e sem acento, com e sem sobrenome, dentro do cabeçalho de
presidência).

Até esta etapa havia dois problemas práticos:

- as UDVs e as falas por ator recortavam a transcrição em turnos com dois códigos separados, que
  precisavam concordar sobre o que é um turno;
- nada ligava uma UDV (que guarda o nome do envolvido como a matéria escreve) ao registro do ator (que
  guarda o nome escolhido entre as grafias dos cabeçalhos). Casar os dois depois, por nome, repetiria a
  resolução de nomes com outro critério.

Agora `split_into_turns` roda uma vez por audiência e os mesmos objetos `Turn` alimentam a
resolução dos envolvidos das UDVs e a coleta das falas por ator. Os turnos que a UDV atribuiu a um
envolvido são os mesmos que entraram no registro do ator, então a ligação UDV → ator sai do próprio
turno, sem comparar nomes de novo.

## Como rodar

A etapa é uma opção de `build-udvs`. Dentro de `challenge/`:

```bash
uv run bookworm build-udvs --run-name udv_v1 --config configs/udv.toml \
  --actors-config configs/hearing_actors.toml
```

Sem `--actors-config`, o comando grava exatamente os mesmos arquivos de antes. Com a opção, a mesma
execução grava também:

| Arquivo | Caminho |
|---|---|
| Falas de quem aparece em uma única audiência | `[speeches].single_hearing_path` da configuração de atores |
| Falas de quem aparece em duas ou mais | `[speeches].multi_hearing_path` |
| Pares de nomes ambíguos | `[speeches].ambiguous_names_path` |
| Contagens da rodada | `[speeches].stats_path` |
| Ligações UDV → ator | `<run.output_dir>/<run>_actor_links.jsonl` |

Os caminhos da configuração de atores são relativos ao diretório onde o comando roda, como os das
outras configurações. O comando para com código 2, antes de calcular qualquer coisa, se o sha256 do
LDS na configuração de atores for diferente do da configuração de UDV, ou se algum desses arquivos já
existir e `--overwrite` não tiver sido passado.

Todas as entradas de `[merges]` precisam casar com pelo menos um turno mantido; se alguma não casar,
o comando para com código 2 e lista as entradas. Por isso uma seleção parcial (`--limit`, `--ids`) com
a configuração completa de fusões costuma falhar: para rodar só algumas audiências, use uma cópia da
configuração sem as fusões que não se aplicam a elas.

Em Python, sem gravar arquivos:

```python
from pathlib import Path

from bookworm import load_hearings
from bookworm.actors.config import load_actors_config
from bookworm.actors.speeches import collect_actor_speeches

config = load_actors_config(Path("configs/hearing_actors.toml"))
hearings = load_hearings(Path("dataset/PublicHearingBR_LDS.jsonl"), config.expected_sha256)
speeches = collect_actor_speeches(hearings, config)
print(len(speeches.multi_hearing), speeches.stats["turns"]["kept"])
```

`bookworm.pipeline.run_pipeline` faz a passada completa (UDVs, falas e ligações) e é o que
`build-udvs` chama.

## Configuração

O arquivo é o mesmo formato de `challenge/configs/hearing_actors.toml`; a tabela `[measurement]`, usada
só pelo script de medição do `challenge/`, é ignorada.

| Chave | Uso |
|---|---|
| `[dataset].lds_path`, `[dataset].sha256` | Caminho do LDS e hash esperado. O hash precisa ser igual ao da configuração de UDV. O caminho só é copiado para as contagens: o LDS lido é o da configuração de UDV. |
| `[speakers].chair_names` | Nomes de cabeçalho (`PRESIDENTE`, `PRESIDENTA`) que marcam um turno de presidência. |
| `[speakers].non_person_keys` | Chaves normalizadas que não são uma pessoa (intérprete, avisos de tradução); seus turnos são descartados. |
| `[speakers].chair_min_words` | Turnos de presidência com menos palavras que isso são descartados. |
| `[speeches]` | Os quatro caminhos de saída da tabela acima. |
| `[[merges.groups]]` | `canonical` e `aliases`: chaves que passam a valer como a chave canônica em todas as audiências. Um alias não pode estar em dois grupos. |
| `[[merges.reassignments]]` | `key`, `hearing_id` e `canonical`: troca a chave só naquela audiência. Um par (`key`, `hearing_id`) não pode aparecer duas vezes. |

## A política, turno a turno

Para cada turno de `split_into_turns`:

1. O nome da pessoa é o do cabeçalho; num turno de presidência, vale o nome entre parênteses
   (`A SRA. PRESIDENTE (Erika Kokay. PT - DF)` dá `Erika Kokay`). O mesmo vale para qualquer
   cabeçalho cujo parêntese não seja partido e UF (`ANTONIO DA SILVA JESUS (TOINHO DO JUDÔ)` dá
   `TOINHO DO JUDÔ`). A chave é esse nome sem acentos, em
   maiúsculas e com espaços normalizados.
2. O papel é `chair` quando o nome do cabeçalho está em `chair_names`, e `speaker` nos demais.
3. O turno é descartado, nesta ordem, se a chave estiver em `non_person_keys`, se a fala for só
   rubricas entre parênteses (`(Palmas.)`), se a fala for vazia ou se for de presidência com menos de
   `chair_min_words` palavras.
4. Se sobreviver, confere-se que `transcricao[start_char:end_char]` é igual ao texto do turno, e a
   chave passa pelas reatribuições da audiência e depois pelos grupos de fusão.

Cada chave final vira um registro. O nome de exibição (`actor`) é a grafia mais frequente entre as
que têm alguma letra minúscula; se nenhuma tiver, entre todas; empates vão para a grafia mais longa e
depois para a primeira em ordem alfabética. `party_uf` lista, na ordem em que aparecem, os textos de
partido e UF dos cabeçalhos. Os nomes de exibição precisam ser únicos, porque são eles que ligam o
registro às UDVs e aos perfis; se dois registros tiverem o mesmo nome, a etapa para com erro.

Depois de todas as audiências, os pares de chaves cujos conjuntos de palavras são iguais ou um contido
no outro vão para o arquivo de nomes ambíguos, para revisão manual. Os pares confirmados como a mesma
pessoa entram em `[merges]` numa próxima rodada.

## Ligação UDV → ator

Para cada UDV, os turnos considerados são os que a resolução de envolvidos da UDV atribuiu à pessoa
(`matched_turns`). Desses, contam-se os que sobreviveram à política acima, agrupados pela chave final
do ator. A ligação fica com a chave de mais turnos (empate: a primeira em ordem alfabética), e
`linked_turns` é o número de turnos dessa chave. `actor` é nulo quando a pessoa não foi resolvida para
nenhum turno (`matched_turns` 0, as UDVs `person_not_resolved`) ou quando nenhum dos turnos dela
sobreviveu à política (`linked_turns` 0).

## Duas regras de ligação

Para avaliar um perfil de ator, cada UDV de uma audiência de teste precisa apontar para o registro do
ator que a disse. O repositório tem duas regras que fazem essa ligação, e a questão era se elas
escolhem atores diferentes para a mesma UDV, o que faria as contagens de avaliação dependerem da regra.

- **Turnos atribuídos** (a regra desta etapa, `bookworm.pipeline.resolve_link`, usada no arquivo
  `<run>_actor_links.jsonl`): entre os turnos que a resolução de envolvidos atribuiu à pessoa da UDV,
  fica o ator com mais turnos mantidos pela política.
- **Turno de evidência** (a regra do bloco `evaluation` de `filter-actor-speeches`, portada de
  `challenge/utils/filter_actor_speeches.py`; ver [Perfis de atores](profiles.md#formato-dos-arquivos)):
  fica o ator dono do turno `evidence.speaker_turn` da UDV. UDV sem evidência, ou cujo turno de
  evidência foi descartado pela política, fica sem ligação.

Sobre `udv_v1` (2.203 UDVs) e o LDS completo:

- a regra dos turnos atribuídos liga 2.104 UDVs e a do turno de evidência, 2.099;
- nas 2.099 UDVs que as duas ligam, o ator é o mesmo; não há nenhuma divergência;
- o turno de evidência está sempre entre os turnos atribuídos à pessoa;
- as 5 ligações a mais da primeira regra (`udv-6-1-0` e `udv-6-1-1`, Aureo Ribeiro; `udv-58-2-0`,
  Soraya Santos; `udv-117-0-0`, Alfredo Gaspar; `udv-195-2-2`, `KRISZTIAN KATONA`) são UDVs cujo turno
  de evidência foi descartado pela política, mas que têm outros turnos mantidos da mesma pessoa; nas
  5, o nome do ator corresponde ao nome da UDV;
- no split `test` de `temporal_v1`, as duas regras dão 106 UDVs ligadas a atores com perfil (duas ou
  mais audiências, pelo menos uma de treino), 48 atores e 27 audiências, os mesmos números de
  `challenge/artifacts/actor_profiles/train_speeches_stats.json`.

Como a primeira regra contém a segunda e nunca escolhe outro ator, o arquivo de ligações continua com
ela. O bloco `evaluation` continua com a regra do turno de evidência, para reproduzir byte a byte as
contagens do `challenge/`; em `udv_v1` test as duas coincidem.

Em 6 UDVs o nome de exibição do ator ligado não é o nome que a matéria usa. As duas regras escolhem o
mesmo ator nelas, porque a diferença vem de antes da ligação, da forma como o nome do ator é formado a
partir do cabeçalho do turno:

- `udv-40-2-0`, `udv-40-2-1` e `udv-40-2-2` (matéria: "Antônio da Silva Jesus (Toninho do Judô)") vão
  para `TOINHO DO JUDÔ`. O cabeçalho é `ANTONIO DA SILVA JESUS (TOINHO DO JUDÔ)`: o nome civil casa
  com o da matéria, e o nome do ator é o que está entre parênteses, grafado sem o "n" na transcrição.
- `udv-146-2-0` (matéria: "Ângela Gomes", representante da Federação das Associações de Cannabis
  Terapêutica) vai para `ANGELA ABOIN`. O cabeçalho da audiência 146 é `ANGELA ABOIN GOMES`, que
  contém as duas palavras do nome da matéria; a fusão `ANGELA ABOIN GOMES` → `ANGELA ABOIN` de
  `[merges]` junta esse registro ao da audiência 126, sobre cannabis medicinal, onde a mesma pessoa
  aparece como `ANGELA ABOIN` e também fala da filha com autismo tratada com cannabis. O nome de
  exibição fica com a grafia canônica, sem "Gomes".
- `udv-206-5-0` e `udv-206-5-1` (matéria: "Maria do Rosário Tripodi", representante do Ministério da
  Educação) vão para `ZARA FIGUEIREDO`. O cabeçalho é `MARIA DO ROSARIO FIGUEIREDO TRIPODI (ZARA
  FIGUEIREDO)`, e a presidência anuncia a fala como a de Zara Figueiredo, do Ministério da Educação:
  o nome civil casa com o da matéria, e o nome do ator é o que está entre parênteses. É um ator de
  uma única audiência, então não entra nos perfis.

Nos três casos o turno atribuído é da pessoa citada na matéria. O que difere é a grafia escolhida para
o registro: quando o cabeçalho traz um nome entre parênteses que não é partido e UF, a política usa
esse nome como chave (passo 1 da política), e uma fusão usa a chave canônica. Quem liga perfis a UDVs pelo
nome de exibição, em vez de usar o arquivo de ligações, perde esses casos.

Esses números são fixados por `tests/integration/test_link_rules.py` (marcador `dataset`), que refaz
as duas regras a partir do LDS, de `udv_v1.jsonl` e de `temporal_v1.json`.

## Números da rodada sobre o LDS completo

Com `challenge/configs/hearing_actors.toml` e o LDS de sha256
`c4e392ab95ce22f6228eace16f15e9efe1c846724846b873dec420aec3d872c0`:

- 17.264 turnos; 13.190 mantidos e conferidos contra a transcrição; descartados 32 de chaves que não
  são pessoa, 36 só de rubricas, 5 vazios e 4.001 de presidência com menos de 50 palavras;
- 37 grupos de fusão (40 aliases, 130 turnos) e 1 reatribuição (1 turno);
- 1.851 atores: 1.550 em uma única audiência e 301 em duas ou mais; 81 pares de nomes ambíguos (1 com
  as mesmas palavras, 80 com um nome contido no outro);
- das 2.203 UDVs, 2.104 têm ator; 90 são de pessoas não resolvidas e 9 são de pessoas resolvidas cujos
  turnos foram todos descartados (8 UDVs de 3 pessoas cujos turnos atribuídos são só rubricas, 1 de
  uma pessoa cujos turnos atribuídos são de chaves que não são pessoa). Em 355 ligações nem todos os
  turnos atribuídos sobreviveram; 726 UDVs apontam para um ator
  de duas ou mais audiências, e as ligações cobrem 827 atores distintos.

Esses números são fixados pelos testes de `tests/integration/test_parity_actor_speeches.py` (marcador
`dataset`). Os mesmos testes conferem que os quatro arquivos gravados pela biblioteca são byte a byte
iguais aos que `utils.build_actor_speeches` grava com a mesma configuração, que as contagens e os pares
ambíguos são iguais aos versionados em `challenge/artifacts/hearing_actors/`, e que a passada única
reconstrói `udv_v1.jsonl` byte a byte a partir do cache de embeddings.

Os formatos dos arquivos estão em [Modelo de dados](data_model.md#falas-por-ator).
