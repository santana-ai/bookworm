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
   (`A SRA. PRESIDENTE (Erika Kokay. PT - DF)` dá `Erika Kokay`). A chave é esse nome sem acentos, em
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
