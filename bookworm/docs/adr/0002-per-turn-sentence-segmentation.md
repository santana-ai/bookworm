# ADR 0002: segmentação de sentenças por turno e escolha da ocorrência da citação

- Status: aceita
- Data: 2026-09-22
- Código afetado: `challenge/utils/udv_pipeline.py`, `challenge/utils/build_udvs.py`,
  `challenge/utils/verify_udvs.py`
- Decisão seguinte que altera a mesma busca de citação: ADR 0003 (trechos entre aspas simples)
- Medição: `challenge/utils/measure_turn_segmentation.py`, saídas em
  `challenge/artifacts/udv/turn_segmentation_summary.json` e
  `challenge/artifacts/udv/turn_segmentation_quote_changes.jsonl`
- Código anterior preservado em `backup/udv_pipeline_2026-09-21.py`, `backup/build_udvs_2026-09-21.py`
  e `backup/verify_udvs_2026-09-21.py` (o código que gerou `artifacts/udv/udv_v0.jsonl` em 21/09/2026)

## Contexto

A UDV liga cada opinião estruturada do LDS a um trecho da fala da própria pessoa na transcrição. Até
esta decisão, a fala de uma pessoa era a concatenação de todos os turnos atribuídos a ela, unidos por
um espaço (`resolve_person_speech`), e três operações rodavam sobre esse texto concatenado: a divisão
em sentenças candidatas (`split_sentences`), a busca do prefixo da citação (`find_opinion_quote_match`)
e a sentença que contém o prefixo (`enclosing_sentence`). Isso produzia quatro defeitos conhecidos.

1. Sentença atravessando turnos. O separador de sentenças só quebrava depois de `.`, `!` ou `?`
   seguidos de espaço. Quando um turno terminava de outro jeito (por exemplo numa rubrica como
   `(Palmas.)`), o fim desse turno e o começo do turno seguinte da mesma pessoa viravam uma sentença
   só, embora entre eles houvesse falas de outras pessoas. Em `udv-1-1-2` a evidência gravada é
   exatamente isso: o fim do turno 15 de Glenn Greenwald (`...sem processo nenhum.(Palmas.)`) colado ao
   começo do turno 59 (`Agradeço, mais uma vez, à Comissão...`). É o único registro de `udv_v0` sem
   offsets, porque esse texto não existe em lugar nenhum da transcrição.
2. Rubrica colada à sentença dentro do mesmo turno. Pelo mesmo motivo, `nenhum.(Palmas.) Agradeço` ou
   `Pode ser?(Pausa.) Vou deixar` formavam uma sentença só mesmo dentro de um turno, e uma rubrica
   precedida de espaço (`Muito bem. (Risos.) Então`) ficava grudada na sentença seguinte.
3. Ocorrência errada do prefixo. Quando o prefixo confiável (6 palavras ou mais) ocorre mais de uma vez
   na fala, a evidência era sempre a primeira ocorrência. Em `udv-65-1-0` o prefixo
   `A Justiça do Trabalho é fundamental` ocorre nos turnos 6 e 18; a opinião diz "em um país tão
   desigual", e é a ocorrência do turno 18 (`...fundamental neste País, um país tão desigual...`) que
   corresponde a ela, mas os offsets apontavam o turno 6.
4. Só a primeira citação que casa era usada. Se a primeira citação da opinião casava com um prefixo
   curto (3 ou 4 palavras) e uma citação posterior casava com 6 ou mais palavras, a opinião ficava sem
   `quote_found`.

Além disso, `get_embedding_model` carregava o encoder sem fixar a revisão, ao contrário de
`build_udvs`, e `verify_udvs` não conferia se o texto de uma evidência sem offsets existe na
transcrição, por isso não acusava o defeito 1.

## Decisão

1. Segmentação por turno. As sentenças candidatas de uma pessoa passam a ser divididas dentro de cada
   turno atribuído a ela, separadamente, e concatenadas na ordem dos turnos
   (`split_turn_sentences`). Cada sentença guarda o índice do turno de onde veio. Nenhuma sentença
   pode atravessar a fronteira entre dois turnos.
2. Busca de citação dentro de um turno. O prefixo da citação é procurado no texto de cada turno
   (espaços normalizados), nunca no texto concatenado; a sentença que contém o prefixo é calculada
   dentro do turno da ocorrência (`find_prefix_occurrences`, `enclosing_turn_sentence`).
3. Offsets no turno de origem. `locate_sentence_span` continua com a mesma assinatura, mas passa a
   receber só o turno de onde a sentença veio (`locate_turn_sentence_span`).
4. Escolha da ocorrência. Quando um prefixo confiável ocorre mais de uma vez nos turnos da pessoa,
   vence a ocorrência cuja sentença tem a maior sobreposição de palavras com o texto da opinião:
   índice de Jaccard entre os conjuntos de tokens `\w+` em minúsculas e sem acentos. Empate mantém a
   primeira ocorrência na ordem dos turnos. A regra é determinística e não usa o encoder. Para
   prefixos curtos (que não viram `quote_found` e só servem para corroborar a escolha do encoder), a
   ocorrência continua sendo a primeira: trocar o critério de corroboração é uma questão em aberto
   separada e não entra nesta decisão.
5. Todas as citações. Todas as citações da opinião são tentadas e vence a que casa com mais palavras de
   prefixo; empate mantém a citação que aparece primeiro na opinião.
6. Separador de sentença depois de parêntese. `SENTENCE_BOUNDARY_PATTERN` passa a quebrar também
   depois de um parêntese fechado cujo conteúdo termina em `.`, `!` ou `?` e que é seguido de espaço,
   exceto a elisão `(...)`:
   `(?<=[.!?])\s+|(?<=[.!?]\))(?<!\(\.\.\.\))\s+`. A regra vale para todo o código que usa o padrão,
   inclusive `split_sentences` e `enclosing_sentence`.
7. Revisão fixada. `get_embedding_model` deixa de ter nome e revisão do encoder escritos no código:
   lê os dois da seção `[encoder]` de `configs/udv.toml` (ou de outro arquivo passado como argumento),
   o mesmo arquivo que `build_udvs` usa, então o notebook e o runner carregam o mesmo encoder na mesma
   revisão (hoje `a01887015444f7599669c509447c5bdbce958916`) sem uma segunda cópia do valor que
   possa divergir.
8. Novas checagens em `verify_udvs`. Para todo registro com evidência, inclusive os de offsets nulos,
   o texto da evidência precisa ocorrer (com espaços normalizados) dentro de um único turno do ator
   (`evidence_not_in_single_actor_turn`). Todo registro com evidência e offsets nulos passa a ser
   acusado (`evidence_offsets_missing`): com a segmentação por turno, todas as sentenças candidatas e
   todas as sentenças de citação confiável são localizadas no turno de origem (seção "Offsets"), então
   um offset nulo só aparece por regressão, e as checagens de turno abaixo não rodam quando o turno é
   nulo. As checagens existentes passam a recalcular a citação, a sentença e o turno com a lógica
   nova, e duas checagens novas conferem que o turno dos offsets é o turno de origem da sentença
   (`quote_turn_mismatch`, `semantic_turn_mismatch`).
9. O relatório de cobertura de `build_udvs` ganha a seção `pipeline`, que descreve a segmentação, o
   padrão de fronteira e as regras de escolha de citação e de ocorrência usadas na rodada.

As funções que operam sobre a fala concatenada (`resolve_person_speech`, `split_sentences`,
`find_quote_match`, `find_opinion_quote_match`, `enclosing_sentence`, `locate_sentence_span`)
continuam importáveis e com o mesmo formato de retorno, porque outros scripts e notebooks as usam;
`build_udvs` e `verify_udvs` deixam de usá-las para a evidência.

## Medições

Todas as medições vêm de `uv run python -m utils.measure_turn_segmentation` (em `challenge/`), que
carrega `backup/udv_pipeline_2026-09-21.py` como código anterior e compara com o código atual nas 206
audiências, sem encoder. O relatório guarda o SHA-256 dos dois arquivos de pipeline e do próprio
script (seção `code`). A comparação de citações usa só os padrões de aspas duplas
(`DOUBLE_QUOTE_PATTERNS`, registrado em `quotes.new_quote_patterns`), para isolar o efeito desta
decisão; o efeito das aspas simples, acrescentadas depois, está na ADR 0003. As mudanças de evidência
semântica dependem do encoder e não foram medidas aqui; elas saem da próxima rodada completa.

### Segmentação

- 1.020 pessoas com turno localizado, 776 delas com mais de um turno.
- Segmentação anterior: 114.969 sentenças candidatas, 350 delas atravessando a fronteira entre dois
  turnos, em 244 pessoas.
- Só a segmentação por turno, com o separador antigo: 115.096 sentenças; a lista muda para 244
  pessoas, exatamente as que tinham sentença atravessando turnos.
- Segmentação por turno com o separador novo: 115.599 sentenças; em relação à segmentação por turno
  com o separador antigo, o separador novo muda a lista de 275 pessoas. Dessas 275, 109 já tinham a
  lista alterada pela segmentação por turno e 166 mudam só por causa do separador; 135 pessoas mudam
  só pela segmentação por turno.
- No total, 410 das 1.020 pessoas têm lista de sentenças diferente (135 + 109 + 166; nenhuma das 109
  alteradas pelos dois passos volta à lista original), em 185 das 206 audiências: 1.073 sentenças
  antigas deixam de existir e 1.703 novas aparecem.

### Separador depois de parêntese

Nos 11.185 turnos atribuídos a alguma pessoa resolvida há 785 ocorrências de um parêntese fechado
cujo conteúdo termina em `.`, `!` ou `?`, seguido de espaço. O conteúdo mais frequente é `Pausa.`
(371), `Palmas.` (269) e `Risos.` (94); 12 são a elisão `(...)`; as outras 39 são rubricas como
`Manifestação na plateia.`, `Exibe documento.` e `O orador se emociona.`.

- As 773 que não são elisão: 761 vêm depois de `.`, `?` ou `!` (571, 149 e 41), e 771 são seguidas
  de letra maiúscula. As 14 restantes estão listadas em
  `non_elision_examples_not_between_punctuation_and_uppercase` e também separam sentenças ou falas:
  rubricas consecutivas (`(Risos.)(Pausa.) Informo`), citação que fecha aspas antes da rubrica,
  rubrica no começo do turno, `importante(riso.) Peço` (ponto ausente antes da rubrica) e dois casos
  na audiência 170 em que a rubrica é seguida de `(Não identificado)-`, a fala de outra pessoa que o
  detector de turnos não separa.
- As 12 elisões estão todas dentro de texto citado ou lido e continuam a mesma sentença (lista em
  `elision_examples`): 11 são seguidas de minúscula e a única seguida de maiúscula é um nome próprio
  no meio da frase (`O presidente da Petrobras, (...) Castello Branco, comemorou`).

Variantes medidas sobre a segmentação por turno (contagem por pessoa):

| Variante | Sentenças | Pessoas com lista alterada |
| --- | --- | --- |
| separador antigo | 115.096 | 0 |
| quebra depois de qualquer `.)`, `!)` ou `?)` | 115.608 | 279 |
| quebra só se a palavra seguinte começa com maiúscula | 115.599 | 276 |
| quebra exceto na elisão `(...)` (adotada) | 115.599 | 275 |

A variante adotada separa as 773 rubricas e não quebra nenhuma das 12 elisões. A variante sem exceção
quebraria as 12 citações no meio. A variante que exige maiúscula quebraria a elisão seguida de nome
próprio e deixaria colada a fala de `(Não identificado)` nos dois casos da audiência 170.

### Citações

Sobre as 2.113 opiniões de pessoas resolvidas:

- O código anterior reproduz os 260 registros `direct_quote` de `udv_v0` (prefixo e texto iguais nos
  260), o que valida a comparação.
- Nenhum prefixo casado pelo código anterior atravessava a junção entre turnos (0), então a busca por
  turno sozinha não tira nenhuma citação.
- Prefixos confiáveis: 260 antes, 263 agora. Prefixos curtos: 244 antes, 241 agora.
- 3 opiniões entram em `quote_found` porque uma citação posterior casa com prefixo de 10 palavras
  enquanto a primeira casava com 2, 3 ou 4: `udv-1-6-0`, `udv-6-0-0` e `udv-183-1-1`. Isso confirma
  as 3 opiniões registradas no ROADMAP.
- Nenhuma opinião sai de `quote_found`.
- 4 registros que já eram `quote_found` mudam:
  - `udv-65-1-0` muda de ocorrência (turno 6 para turno 18), como esperado. Entre os 263 prefixos
    confiáveis, 3 ocorrem mais de uma vez na fala e só esse muda de ocorrência pela regra de
    sobreposição.
  - `udv-24-1-0` muda de prefixo: a primeira citação casa com 6 palavras e a segunda com 10, então a
    regra passa a gravar a sentença da segunda. Esse caso não estava previsto. A opinião afirma a
    dificuldade de acesso a filosofia e sociologia, assunto da primeira citação, e a sentença nova tem
    sobreposição menor com ela (Jaccard 0,2157 contra 0,3654).
  - `udv-48-2-0` e `udv-82-2-1` mantêm prefixo e ocorrência, mas o texto encurta por causa do
    separador novo. No primeiro, a evidência incluía a sentença anterior, colada por `(Palmas.)`. No
    segundo, incluía as quatro perguntas seguintes e a exclamação final (`Cabeça de rádio!`), coladas
    por `(A oradora toca um pandeiro.)`; a evidência nova é só a primeira pergunta.
- 7 dos 263 prefixos confiáveis vêm de uma citação que não é a primeira da opinião.
- 1 prefixo curto muda por causa da regra de todas as citações (`udv-47-0-3`, de 3 para 4 palavras
  em outra citação); se ele passa a corroborar a escolha do encoder depende da rodada com encoder.
- Os detalhes de cada um desses 8 registros estão em `turn_segmentation_quote_changes.jsonl`.

### Offsets

- Todas as 115.599 sentenças candidatas são encontradas no próprio turno de origem por
  `locate_sentence_span`, e as 263 sentenças de citação confiável também
  (`quotes.new_trusted_located_in_source_turn`).
- Em 85 sentenças candidatas a busca encontra uma posição anterior à da sentença no turno
  (`offset_search.earlier_position_by_kind`):
  - em 76 é o começo de outra sentença candidata com o mesmo texto (frase repetida no turno);
  - em 5 é o começo de uma sentença candidata diferente e mais longa que começa com o mesmo texto: em
    3 é a mesma frase com uma rubrica colada (`Nós queremos paz no campo.(Palmas.)`), em 1 falta o
    espaço antes da sentença seguinte (`...autonomia financeira.Onde fazer?`) e em 1 a frase continua
    depois de reticências (`Eu quero saber quem é..."A gente mesmo", não.`);
  - em 4 é o meio de uma sentença candidata mais longa.

  O texto do trecho é o mesmo nos 85 casos; muda o ponto da transcrição. Nos 9 casos que não são
  frase repetida, o trecho gravado não coincide com nenhuma sentença candidata inteira: começa numa
  fronteira de sentença e termina dentro dela (5) ou começa no meio de uma sentença (4). Os 9 estão
  listados em `offset_search.earlier_position_examples_not_identical`.
- Nas 263 evidências de citação confiável isso não acontece (0 casos): o offset sempre aponta a
  sentença da ocorrência escolhida.

### Rodada existente

`uv run python -m utils.verify_udvs --run-name udv_v0` com a lógica nova sai com código 1, como
esperado para um artefato gerado pelo código anterior. As checagens novas
`evidence_not_in_single_actor_turn` e `evidence_offsets_missing` acusam exatamente `udv-1-1-2`. As
outras acusações são os registros cuja evidência a lógica nova produz de outro jeito; entre
parênteses, o que vem desta decisão e o que vem das aspas simples (ADR 0003), porque a verificação usa
o código atual inteiro:

- `evidence_not_person_sentence`: 4 (`udv-1-1-2`, `udv-45-0-2`, `udv-169-2-0`, `udv-170-4-0`), cujo
  texto deixou de ser sentença candidata;
- `semantic_but_trusted_quote_findable`: 17 (3 desta decisão, as opiniões `udv-1-6-0`, `udv-6-0-0` e
  `udv-183-1-1`, que entram em `quote_found`; 14 da ADR 0003);
- `short_quote_support_mismatch` e `short_quote_prefix_mismatch`: 15 cada (2 desta decisão,
  `udv-1-6-0` e `udv-6-0-0`, que hoje estão gravadas como `semantic_with_short_quote`; 13 da ADR
  0003);
- `quote_prefix_mismatch`: 1 (`udv-24-1-0`);
- `quote_text_mismatch`: 3 (`udv-48-2-0`, `udv-65-1-0`, `udv-82-2-1`);
- `quote_turn_mismatch`: 1 (`udv-65-1-0`).

Não há acusação de contadores do relatório nem de `semantic_turn_mismatch`. Em `dev20` a mesma
verificação acusa `udv-1-1-2` (`evidence_not_in_single_actor_turn`, `evidence_not_person_sentence`,
`evidence_offsets_missing`) e os dois registros das 20 primeiras audiências que entram em
`quote_found` (`udv-1-6-0`, `udv-6-0-0`). Em `dev20` as aspas simples só acrescentam um prefixo curto
em 3 opiniões (`udv-1-7-0`, `udv-1-7-1`, `udv-2-0-2`), que não cai na sentença gravada em nenhuma
delas, então não geram acusação.
`udv_v0_repro`, a rodada refeita com o código anterior, recebe as mesmas acusações de `udv_v0` e não
tem nenhuma evidência diferente dela.

## Consequências

- `udv_v0.jsonl`, `dev20.jsonl` e qualquer rodada gerada antes desta decisão continuam como estão,
  mas não passam mais em `verify_udvs`, que confere a lógica atual. Uma rodada completa nova é
  necessária, com o encoder, e é ela que vai medir quantas evidências semânticas mudam; 4 mudam com
  certeza, porque o texto gravado deixou de ser candidato.
- O cache de embeddings é indexado pela lista de sentenças da audiência, que muda em 185 das 206
  audiências; a próxima rodada recalcula as sentenças dessas audiências. Os embeddings das opiniões
  não mudam.
- O corte de 0,47 foi calibrado com pares confirmados por citação nas 20 primeiras audiências, com a
  lógica anterior. Duas opiniões dessas audiências entram em `quote_found`, então o conjunto de pares
  positivos muda e a calibração precisa ser refeita numa versão nova do notebook de UDV.
- Scripts e notebooks que importam `SENTENCE_BOUNDARY_PATTERN`, `split_sentences`,
  `enclosing_sentence` ou `resolve_hearing_people` (`measure_case_insensitive_quotes.py`,
  `generate_udv_manual_review.py`, `udv_v05.ipynb`) passam a usar o separador novo e, no caso de
  `resolve_hearing_people`, a segmentação por turno. As saídas gravadas por eles foram produzidas com o
  código em `backup/`; reexecutá-los com o código atual pode dar números diferentes, o que não foi
  medido aqui.
- Limitações que ficam:
  - A regra de todas as citações prefere o prefixo mais longo mesmo quando a primeira citação já é
    confiável e tem sobreposição maior com a opinião (`udv-24-1-0`, único caso nas 206 audiências).
  - A corroboração por prefixo curto continua usando a primeira ocorrência.
  - Os 9 casos em que o offset de uma sentença candidata não aponta a própria sentença nem uma
    repetição idêntica dela (5 no começo de uma sentença mais longa, 4 no meio de outra sentença)
    seguem possíveis na camada semântica; se algum deles vira evidência depende do encoder.
  - A fala marcada como `(Não identificado)-` continua dentro do turno da pessoa anterior: a
    segmentação por turno depende dos turnos que `split_into_turns` detecta.

## Alternativas consideradas

- Manter a fala concatenada e inserir um separador artificial entre turnos (por exemplo um ponto
  final). Resolveria o defeito 1, mas altera o texto da fala, deixa a busca de citação ainda capaz de
  casar um prefixo que atravessa a junção se o separador estiver dentro dele, e não dá o turno de
  origem de cada sentença, que é o que permite fixar os offsets no turno certo.
- Calcular os offsets diretamente da posição de cada sentença no turno, sem busca por regex. Seria
  exato também nos 85 casos de texto repetido no turno; a busca restrita ao turno de origem foi
  mantida porque reaproveita `locate_sentence_span`, é exata nas 263 citações confiáveis e erra o ponto
  (sem errar o texto) em 85 das 115.599 sentenças candidatas.
- Escolher a ocorrência do prefixo pelo encoder (maior similaridade de embedding entre a opinião e a
  sentença). Seria coerente com a camada semântica, mas faria a evidência de citação depender do
  modelo e da versão do encoder; a sobreposição de palavras é determinística e resolve o caso que
  motivou a mudança.
- Usar a primeira citação confiável em vez da citação com mais palavras de prefixo. Produziria as
  mesmas 3 entradas em `quote_found` e manteria `udv-24-1-0` como está. A regra adotada privilegia o
  casamento literal mais longo (10 palavras contra 6 nesse registro), e a diferença entre as duas
  regras nas 206 audiências é esse único registro. Fica registrada como alternativa a reavaliar na
  validação humana das citações.
- Quebrar depois de qualquer parêntese fechado precedido de `.`, `!` ou `?`, sem exceção, ou só quando
  a palavra seguinte começa com maiúscula. As duas foram medidas e descartadas pelos casos descritos em
  "Separador depois de parêntese".
