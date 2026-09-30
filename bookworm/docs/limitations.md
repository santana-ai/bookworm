# Limitações conhecidas e uso dos resultados

O que as regras da biblioteca garantem e o que não garantem, com os casos medidos em `udv_v1`. As
limitações dos experimentos (validação humana, verificador, simulação) estão no
[relatório, seção 10](../../docs/report.md#10-limitações).

## Limitações

- **Ocorrência de um prefixo repetido.** Um prefixo confiável que aparece mais de uma vez na fala do
  participante é resolvido pela sobreposição de palavras com a opinião; um prefixo curto usa sempre a
  primeira ocorrência, e mudar esse critério é uma questão separada (ADR 0002). Em `udv_v1`, 15
  registros com `quote_prefix` têm o prefixo repetido nos turnos do participante: 3 `direct_quote`
  (`udv-44-3-0` e `udv-192-0-1`, com as duas ocorrências no mesmo turno e a primeira escolhida, e
  `udv-65-1-0`, em que a segunda ocorrência vence) e 12 `semantic_with_short_quote`, fixados em
  `tests/integration/test_known_limitations.py`.
- **Citação com mais palavras de prefixo.** A regra prefere o prefixo mais longo mesmo quando a
  primeira citação já é confiável e tem sobreposição maior com a opinião; nas 206 audiências isso muda
  um registro, `udv-24-1-0` (ADR 0002).
- **Garantia de uma citação localizada.** `quote_found` garante que um prefixo de 6 a 10 palavras da
  citação aparece num turno da própria pessoa: as 10 primeiras palavras, ou a citação inteira quando
  ela tem de 6 a 9 palavras; se esse primeiro degrau não é encontrado, as 6 primeiras. As palavras da
  citação depois do prefixo encontrado não são conferidas. Em `udv_v1`, o prefixo de `direct_quote`
  tem 10 palavras em 122 registros, 6 em 152 e 7 em 3 (citações de exatamente 7 palavras encontradas
  inteiras). Na validação humana de `udv_v1` (`experiments/artifacts/udv/udv_v2_precision_final.json`),
  o trecho de 35 citações sorteadas no teste sustenta a afirmação inteira em 25 e parte dela em 6.
- **Prefixos curtos pouco distintivos.** A corroboração de `semantic_with_short_quote` aceita prefixos
  de 1 a 5 palavras; em `udv_v1` são 1 de 1 palavra, 7 de 2, 37 de 3, 65 de 4 e 1 de 5.
- **Posição de um texto repetido no turno.** O offset é a primeira posição do texto no turno de
  origem. Nas 115.599 sentenças candidatas, essa posição é anterior à da própria sentença em 85 casos:
  76 frases repetidas no turno e 9 em que o texto aparece antes, dentro de outra sentença (ADR 0002).
  Nas 2.105 evidências de `udv_v1`, o texto aparece uma única vez no turno de origem e o trecho gravado
  começa e termina em fronteiras de sentença desse turno, então nenhum desses casos ocorre (teste em
  `tests/integration/test_known_limitations.py`).
- **Turnos que o detector não separa.** A segmentação por turno depende dos cabeçalhos que
  `split_into_turns` reconhece; uma fala marcada como `(Não identificado)-` continua dentro do turno da
  pessoa anterior.
- **Encoder.** O Serafim tem `max_seq_length` de 128 tokens, então sentenças mais longas são truncadas
  antes de virar vetor. O lock da biblioteca resolve `sentence-transformers` 6.1.0 e `torch` 2.14.0,
  enquanto os artefatos foram produzidos com 5.6.1 e 2.13.0; recalcular embeddings pode dar scores
  ligeiramente diferentes, e o teste `model` aceita até `1e-5`.
- **Limiar.** O corte 0,45 separa `high` de `weak` e não foi validado como nível de confiança; com
  TF-IDF, a divisão não tem calibração.
- **Chave do cache de embeddings.** O separador `\x1e` entre textos não é escapado, então `["a", "b"]` e
  `["a\x1eb"]` geram a mesma chave. Mantido por compatibilidade com o cache existente.
- **Data do split.** A data usada é a de publicação da matéria. Em `temporal_v1`, 155 das 156 matérias
  que nomeiam o dia do debate confirmam a data com defasagem de 0 ou 1 dia; a audiência 111 cita um dia
  que não confere com o dia da semana. As fronteiras escolhidas têm 8 e 76 dias de intervalo.
- **O que o split temporal não isola.** 55 dos 879 nomes distintos aparecem em mais de um conjunto, 24
  deles em treino e teste; quem precisar medir generalização para participantes inéditos precisa de um
  split por ator. Os `tier` também não se distribuem igualmente: 80 das 90 opiniões de pessoa não
  resolvida estão no treino, e 6 das 8 sem evidência estão no teste.
- **Relatório do split.** `temporal_v1_report.json` conta as UDV de `udv_v0`, com que foi gerado; as
  contagens por `tier` desse relatório não refletem `udv_v1`.

## Uso dos resultados

- Uma UDV liga uma opinião publicada a um trecho da transcrição por uma regra automática; ela não é
  anotação humana nem substitui a leitura da transcrição original. `provenance` registra como cada
  ligação foi feita (`weak` para casamento de citação, `model` para similaridade).
- Uma opinião sem evidência (`no_evidence`, `person_not_resolved` ou similaridade baixa) é uma opinião
  para a qual o método não encontrou trecho correspondente; a biblioteca não classifica essas opiniões
  como alucinação da matéria.
- A biblioteca descreve o que é observável nos textos; ela não infere intenção política de
  participantes, jornalistas ou veículos.
- Os rótulos NLI do dataset (arquivo que a biblioteca não usa) dizem se uma opinião é inferível a
  partir de até quatro trechos recuperados da fala; eles não cobrem a transcrição inteira.

