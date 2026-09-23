# ADR 0003: trechos entre aspas simples contam como citação

- Status: aceita
- Data: 2026-09-22
- Código afetado: `challenge/utils/udv_pipeline.py` (`SINGLE_QUOTE_PATTERN`, `QUOTE_PATTERNS`,
  `extract_quotes`, `find_opinion_turn_quote_match`), `challenge/utils/build_udvs.py` (seção
  `pipeline` do relatório de cobertura), `challenge/utils/build_quote_benchmark.py` e
  `challenge/configs/quote_benchmark.toml` (máscara do benchmark de citações mascaradas)
- Medição: `uv run python -m utils.measure_quote_patterns` (em `challenge/`), saídas em
  `challenge/artifacts/udv/quote_patterns_summary.json` e
  `challenge/artifacts/udv/quote_patterns_changes.jsonl`
- Decisão anterior sobre a mesma busca: ADR 0002

## Contexto

A evidência de nível `quote_found` vem de um trecho da opinião marcado como citação cujo prefixo de 6
ou mais palavras aparece num turno da própria pessoa. Até esta decisão, só `“…”` e `"…"` eram
reconhecidos como citação (`QUOTE_PATTERN`). O LDS também usa aspas simples ASCII (`'…'`) para
marcar fala literal, como em `udv-154-0-0`: "Disse: 'É necessário debater a promoção da saúde da
mulher para esse ciclo da vida, [...]'", cujo trecho entre aspas é idêntico a uma sentença da
própria pessoa na transcrição. Essas opiniões nunca viravam `quote_found` e caíam na camada
semântica. Havia dois efeitos derivados no benchmark de citações mascaradas (`masked_quotes_v1`), que
serve de entrada para a calibração do corte de embeddings:

1. a opinião cuja única citação está entre aspas simples ficava fora do benchmark e fora das consultas
   de citação confiável da calibração;
2. a máscara removia `‘…’` mas não `'…'`, então uma opinião mantida no benchmark podia continuar com
   texto entre aspas simples depois da máscara, e a checagem `masked_with_quote_marks_left` não via
   isso, porque só procurava as marcas configuradas.

## Números do train usados na decisão

A decisão foi tomada olhando só o split `train` do manifesto `temporal_v1`
(`by_split.train` em `quote_patterns_summary.json`); os números de validação e teste mais abaixo foram
calculados depois dela, só para registro.

- 1.456 opiniões de pessoas resolvidas; 62 têm pelo menos um trecho entre aspas simples com 10 ou mais
  caracteres (66 trechos, 5 deles dentro de um trecho entre aspas duplas).
- Prefixo mais longo encontrado nos turnos da pessoa, por trecho entre aspas simples: 10 palavras em
  5, 6 palavras em 5, 5 em 1, 4 em 8, 3 em 8, 1 ou 2 em 13, nenhum em 26. São 10 de 66 (15,2%) com
  prefixo confiável, contra 174 de 635 (27,4%) nos trechos entre aspas duplas.
- Com as aspas simples, 10 opiniões entram em `quote_found` (173 para 183 prefixos confiáveis); todas
  estavam em `semantic_match_high` em `udv_v0`, e nenhum dos 173 casamentos confiáveis existentes
  muda. 22 opiniões ganham um prefixo curto que não tinham e 3 trocam de prefixo curto.
- As 10 estão em `quote_patterns_changes.jsonl` com o prefixo, a sentença localizada e a sobreposição
  de palavras com a opinião. Em 8 delas o trecho entre aspas é uma fala introduzida por verbo de
  elocução (`Disse: '...'`, `Observou que '...'`); em `udv-39-3-0` é uma expressão de 6 palavras
  (`o poder moderador das Forças Armadas`) e em `udv-197-1-0` é uma fala citada dentro de um trecho
  entre aspas duplas cujo prefixo não aparece na transcrição.
- Nenhuma opinião do train tem `'` ou `’` entre duas letras ou dígitos (`with_apostrophe_inside_word`
  = 0), então as fronteiras de palavra do padrão não descartam nenhum trecho no LDS atual; elas existem
  para que um apóstrofo dentro de palavra (`d'água`) nunca abra uma citação.
- Trechos entre `‘…’` com 10 ou mais caracteres: 6 no train, nenhum com prefixo confiável (1 com 1
  palavra, 1 com 3, 4 sem prefixo).

O critério que dá confiança a `quote_found` é o casamento literal de 6 ou mais palavras num turno da
própria pessoa, não o tipo de aspas. Com o mesmo critério, as aspas simples produzem no train o mesmo
tipo de evidência que as duplas, e nenhum casamento existente é perdido ou trocado.

## Decisão

1. `SINGLE_QUOTE_PATTERN = (?<!\w)'([^']{10,})'(?!\w)` passa a ser um padrão de citação, com o mesmo
   tamanho mínimo das aspas duplas. `QUOTE_PATTERNS` reúne os dois padrões; `DOUBLE_QUOTE_PATTERNS`
   guarda só o de aspas duplas, para medições que precisam isolar o efeito da mudança.
2. `extract_quotes` junta os casamentos de todos os padrões e os ordena pela posição na opinião. Um
   trecho entre aspas simples dentro de um trecho entre aspas duplas também é extraído, logo depois do
   trecho que o contém. As regras de escolha da ADR 0002 não mudam: vence o trecho com mais palavras de
   prefixo e o empate fica com o que aparece primeiro na opinião. Com `DOUBLE_QUOTE_PATTERNS`, a saída
   é a mesma de antes.
3. `‘…’` continua fora dos padrões de citação: no train não há nenhum caso com prefixo confiável, então
   não há evidência para a mudança.
4. A máscara do benchmark passa a remover também `'…'`, com as mesmas fronteiras de palavra
   (`masking.word_bounded_quote_pairs` em `quote_benchmark.toml`). A checagem de sobra passa a procurar
   qualquer caractere de aspas (`QUOTE_CHARACTERS`, configurado ou não) e a conferir que nenhum trecho
   extraído pelos padrões de citação continua na opinião mascarada.
5. O mínimo de 4 palavras depois da máscara foi recalculado no train com a regra nova: 183 opiniões
   com prefixo confiável; 36 ficam com 0 palavras, 2 com 1, 3 com 2 (verbo de elocução seguido de
   `que`) e nenhuma com 3. Qualquer mínimo de 3 a 4 descarta as mesmas 41 linhas; o empate continua
   resolvido pelo mínimo de palavras de uma sentença candidata (4), e o texto da regra em
   `quote_benchmark.toml` foi atualizado.

## Efeito medido

Nas 206 audiências, depois da decisão (`quote_patterns_summary.json`):

| Split | Prefixos confiáveis, só aspas duplas | Com aspas simples | Entram em `quote_found` | Prefixo curto novo | Prefixo curto trocado |
| --- | --- | --- | --- | --- | --- |
| train | 173 | 183 | 10 | 22 | 3 |
| validation | 49 | 53 | 4 | 3 | 0 |
| test | 41 | 41 | 0 | 6 | 0 |

- As 14 opiniões que entram em `quote_found` estavam todas em `semantic_match_high` em `udv_v0`;
  nenhuma sai de `quote_found` e nenhum prefixo confiável muda de texto ou de ocorrência.
- As 277 sentenças de citação confiável são localizadas no turno de origem
  (`trusted_all_patterns_located_in_source_turn`).
- Em 1 dos 3 prefixos curtos trocados no train (`udv-200-0-0`), o trecho entre aspas simples e o
  trecho entre aspas duplas casam com o mesmo número de palavras (4), e vence o que aparece primeiro na
  opinião, que é o de aspas simples.
- `verify_udvs` em `udv_v0` passa a acusar essas 14 opiniões em `semantic_but_trusted_quote_findable`
  e mais 13 registros em `short_quote_support_mismatch` e `short_quote_prefix_mismatch`: neles, um
  prefixo curto entre aspas simples cai na sentença gravada naquela rodada (a escolha do encoder), e,
  com essa mesma sentença, a regra atual os classificaria como `semantic_with_short_quote`.
- Benchmark de citações mascaradas (`masked_quotes_v1_report.json`), reconstruído com a regra nova:
  277 opiniões consideradas e 210 mantidas (train 142, validação 40, teste 28). Das 14 opiniões que
  entram em `quote_found`, 9 ficam no benchmark e 5 são descartadas porque a opinião é só a citação
  (`udv-63-0-1`, `udv-63-1-1`, `udv-73-4-1`, `udv-154-0-0` no train, `udv-128-3-2` na validação).
  Como nenhum prefixo confiável existente muda, as demais linhas mantêm alvo, candidatos e índices.
- Com a máscara antiga (só `masking.quote_pairs`, sem `'…'`), 13 das 210 linhas mantidas ficariam com
  aspas simples na opinião mascarada (`quote_characters_left_without_word_bounded_pairs_kept`): 8 das 9
  linhas novas, em que o trecho entre aspas simples é a própria citação que define o alvo, e 5 linhas
  em que ele é outro trecho da opinião (`udv-39-8-0`, `udv-52-3-1`, `udv-69-2-0`, `udv-189-0-1` no
  train, `udv-111-1-1` no teste). A nona linha nova, `udv-197-1-0`, tem o trecho entre aspas simples
  dentro de aspas duplas, que a máscara antiga já removia. Por isso a máscara e o padrão de citação
  mudam juntos. Com a máscara nova, nenhuma linha mantida tem caractere de aspas nem trecho citado
  depois da máscara.

## Consequências

- `udv_v0`, `dev20` e `udv_v0_repro` não foram regenerados. A próxima rodada completa, com o encoder,
  deve ter 14 opiniões a mais em `quote_found` e pode mudar o `support_type` de até 34 registros com
  prefixo curto novo ou trocado; o número exato depende da sentença que o encoder escolher.
- As entradas da calibração mudam: 183 consultas de citação confiável no train (eram 173 só com
  aspas duplas) e 142 opiniões mascaradas no train.
- `measure_case_insensitive_quotes.py`, `generate_udv_manual_review.py` e `udv_v05.ipynb` usam
  `extract_quotes` e passam a ver as aspas simples; as saídas gravadas por eles vêm do código anterior.
- Limitações que ficam:
  - A corroboração por prefixo curto já aceitava prefixos pouco distintivos, e as aspas simples
    acrescentam outros: nos 13 registros acima, o prefixo tem 1 palavra em 1 (`minutinhos`), 2 em 3
    (`populismo penal`, `tempo recorde`, `por dentro`), 3 em 4 e 4 em 5.
  - A validação humana das citações aceitas continua pendente, agora também para as 14 opiniões desta
    decisão.
  - `‘…’` fica fora dos padrões enquanto não houver caso no train que justifique a inclusão.

## Alternativas consideradas

- Só mascarar `'…'` no benchmark, sem tratá-las como citação. Resolveria a sobra na opinião mascarada,
  mas deixaria 14 opiniões com fala literal localizada fora de `quote_found` e fora do benchmark.
- Um único regex com alternativas para aspas duplas e simples. Um trecho entre aspas simples dentro de
  aspas duplas seria engolido pelo casamento externo; no train são 5 trechos assim, e `udv-197-1-0` só
  entra em `quote_found` pelo trecho interno.
- Aspas simples sem fronteira de palavra. No LDS atual daria o mesmo resultado (nenhum apóstrofo
  dentro de palavra), mas deixaria um apóstrofo abrir uma citação em dados futuros.
