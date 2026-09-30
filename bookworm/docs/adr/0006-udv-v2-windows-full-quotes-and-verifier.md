# ADR 0006: udv_v2, janelas de duas sentenças, citação inteira e verificador como confiança

- Status: aceita
- Data: 2026-09-28
- Código afetado: `src/bookworm/udv/quotes.py` (extensão da evidência de citação),
  `src/bookworm/udv/windows.py` (novo: unidades de janela), `src/bookworm/udv/build.py`,
  `src/bookworm/udv/evidence.py`, `src/bookworm/udv/verify.py`,
  `src/bookworm/udv/coverage.py`, `src/bookworm/config.py`;
  `challenge/utils/calibrate_udv_v2.py` (novo), `challenge/utils/udv_verifier.py` (segundo corte),
  `challenge/utils/udv_v2_analysis.py` (novo), `challenge/configs/udv_v2.toml`,
  `challenge/configs/udv_v2_verifier.toml`, `challenge/configs/udv_v2_calibration_verifier.toml`
- Decisões anteriores sobre a mesma construção: ADR 0002 (sentenças por turno), ADR 0003 (aspas
  simples)
- Documento de referência: `challenge/PIPELINE.md`

## Contexto

`udv_v1` tem três limitações medidas.

1. **A evidência de citação cobre só a primeira sentença.** `quote_found` guarda a sentença que contém
   o prefixo de 6 ou mais palavras da citação. Das 277 citações escolhidas em `udv_v1`, 139 têm mais de
   uma parte de sentença, e só 82 evidências contêm as 3 últimas palavras da citação
   (`quote_evidence.udv_v1` de `challenge/artifacts/udv/udv_v2_analysis.json`). Nas outras, o resto da
   fala citada fica fora da evidência.
2. **A unidade semântica é uma sentença.** No E2 (`retrieval_v1_report.json`, só train e validação),
   janelas de sentenças consecutivas do mesmo turno acham o trecho citado mais vezes que sentenças
   isoladas no benchmark de citações mascaradas.
3. **O nível semântico não mede se a sentença sustenta a opinião.** O cosseno escolhe a sentença; ele
   não separa bem sentença que sustenta de sentença que não sustenta. No confidence_v2, a probabilidade
   do verificador E3x (`laya_en_en:learned_cross_model_with_pt`) separa melhor que o cosseno na
   validação do benchmark NLI (ROC AUC 0,876 contra 0,730, p de Holm 0,012).

## Decisão

### 1. Resolução de pessoa: sem mudança

A resolução por nome contido continua igual à de `udv_v1`. O casamento aproximado de nomes (E6) fica
desligado até a planilha de revisão de nomes ser preenchida, e a atribuição da fala do intérprete de
Libras aos participantes surdos não é implementada. As duas ficam registradas como pendências em
`challenge/PIPELINE.md`, com o potencial medido.

### 2. Evidência de citação inteira

Vale só para casamentos confiáveis (prefixo de 6 ou mais palavras, regra de `udv_v1`), com a mesma
citação, o mesmo turno e a mesma ocorrência escolhidos em `udv_v1`. Tudo acontece no texto do turno
com espaços normalizados, o mesmo texto em que o prefixo foi achado.

1. `Q` é a citação escolhida (espaços normalizados) e `s` é o início do prefixo no texto do turno.
2. **Fim da citação por sufixo.** Para `k` em 6, 4 e 3, nesta ordem: o sufixo são as últimas `k`
   palavras de `Q`, sem a pontuação final da última palavra; ele só é tentado se `Q` tiver pelo menos
   `k` palavras e o sufixo tiver pelo menos 6 caracteres. O sufixo é procurado com o mesmo padrão do
   prefixo (`prefix_pattern`) a partir de `s`; vale a primeira ocorrência que termina depois do fim do
   prefixo e cujo fim `e` fica a no máximo `2 × len(Q)` caracteres de `s`. O primeiro `k` com
   ocorrência válida decide.
3. **Sem sufixo, contagem de sentenças.** Se nenhum sufixo for achado, conta-se o número `n` de
   partes de `Q` pelo mesmo padrão de fronteira de sentença da segmentação; a evidência cobre a parte
   do turno que contém o prefixo e as seguintes até somar `n` partes, sem passar do fim do turno.
4. A evidência é o conjunto de partes de sentença do turno que cobrem `[s, e)` (as mesmas partes que
   `enclosing_turn_sentence` junta), então ela sempre contém a sentença de `udv_v1`, começa numa
   fronteira de sentença e nunca sai do turno.
5. Os offsets são localizados como em `udv_v1` (primeira ocorrência do texto no turno de origem), e
   `normalize_whitespace(transcrição[start_char:end_char])` é igual ao texto da evidência.

`quote_prefix`, `support_type`, `score` (nulo), nível e proveniência não mudam.

### 3. Unidade da busca semântica: janela de 2 sentenças

A unidade candidata passa a ser `window2` do E2: 2 sentenças candidatas consecutivas do mesmo turno,
passo 1; um turno com uma única sentença candidata dá uma unidade com ela. O texto codificado pelo
encoder é o das sentenças juntadas por espaço, exatamente o texto da unidade no E2. A evidência
guarda o trecho da transcrição do início da primeira sentença ao fim da última (localizadas em
sequência dentro do turno), com espaços normalizados; se houver entre as duas um fragmento que a
segmentação descartou (menos de 4 palavras ou rubrica), ele aparece no texto da evidência.

**Regra de escolha, aplicada aos resultados do E2 em train e validação, sem teste:** a unidade adotada
precisa superar `sentence` no ganho sobre o acaso (`acc_at_1_minus_random`) com p de Holm < 0,05 em
algum benchmark, no train e na validação, e não pode ser significativamente pior que `sentence` em
nenhum benchmark e split. A regra foi escrita depois de ler os resultados do E2, e isso fica
registrado.

| unidade (serafim_335m) | masked_quotes train | masked_quotes validação | nli train | nli validação |
|---|---|---|---|---|
| window2 contra sentence | +0,097 (Holm 0,007) | +0,179 (Holm 0,017) | +0,007 (Holm 0,252) | +0,014 (Holm 0,410) |
| window3 contra sentence | +0,200 (Holm 0,0003) | +0,254 (Holm 0,017) | -0,020 (Holm 0,019) | -0,019 (Holm 0,410) |

Fonte: `comparisons.<bench>.<split>.units_within_retriever.serafim_335m` de
`challenge/artifacts/experiments/retrieval/retrieval_v1/retrieval_v1_report.json`.

`window3` tem ganho maior no benchmark mascarado, mas é pior que `sentence` no benchmark NLI do
train com p de Holm 0,019. `window2` é a única unidade que passa pela regra. Não há teste direto entre
`window2` e `window3` no E2; a escolha não afirma que `window2` é melhor que `window3`.

### 4. Corte do cosseno recalibrado no train

O cosseno de uma janela tem outra distribuição que o de uma sentença, então 0,45 não vale mais. O corte
é recalculado por `challenge/utils/calibrate_udv_v2.py`, que reusa as funções de
`calibrate_threshold.py` com janelas no lugar de sentenças:

- só audiências do train do manifesto `temporal_v1`; a checagem de vazamento continua e falha se uma
  audiência de validação ou teste entrar em qualquer consulta, codificação ou sorteio de negativo;
- positivos: opinião sem máscara contra a janela que contém a sentença alvo da citação confiável (a de
  maior cosseno, se houver mais de uma);
- negativos `legacy_random`: a mesma opinião contra uma janela sorteada de outra audiência do train;
  negativos `hard_negative`: contra uma janela sorteada da mesma pessoa que não contém a sentença alvo;
- **regra adotada, a mesma de `udv_v1`: `legacy_random`**, ponto médio do quantil 0,25 dos positivos e
  do quantil 0,75 dos negativos, com intervalo de 95% por bootstrap de audiência (1.000 réplicas,
  semente 42). `hard_negative`, `masked_hard_negative` e `masked_top1_youden` são relatados para
  comparação e não escolhem o corte.

### 5. Verificador como confiança de toda UDV com evidência

Toda UDV com evidência recebe a probabilidade do primário do E3x, calculada sobre o texto da evidência
de `udv_v2` (premissa) e a proposição (hipótese), com `challenge/utils/udv_verifier.py` sem mudança no
modelo: mesmos scorers, mesma tradução NLLB (só textos que faltam no cache), mesmo ajuste no train do
benchmark NLI. Dois cortes são registrados.

- **Corte do benchmark (`train_threshold`)**: 0,7478, o de `final_test.json`, ajustado com premissas de
  4 chunks recuperados.
- **Corte de premissa UDV (`udv_threshold`)**: recalculado só no train com os mesmos pares da calibração
  do cosseno (item 4): a probabilidade do primário nos positivos e nos negativos `legacy_random`, com a
  mesma regra de ponto médio dos quantis 0,25 e 0,75. A variante com os negativos `hard_negative` é
  relatada e não é adotada. Como os pares são fixos depois do sorteio (cada negativo precisa ser
  traduzido e pontuado), o bootstrap por audiência reamostra os pares sorteados em vez de sortear
  negativos novos.

A probabilidade, as duas decisões e a proveniência (primário, corte, artefato do corte, arquivo de
entrada) ficam num arquivo lateral versionado, `challenge/artifacts/udv/udv_v2_verifier.jsonl`, uma
linha por UDV. `UdvRecord` não muda: um campo novo mudaria o esquema estrito do registro e a
serialização de `udv_v1`, que os testes de paridade comparam byte a byte.

### 6. Níveis

Os níveis continuam os mesmos cinco (`quote_found`, `semantic_match_high`, `semantic_match_weak`,
`no_evidence`, `person_not_resolved`) e são atribuídos pelas mesmas regras, com o corte do item 4. A
decisão do verificador é um campo separado no arquivo lateral e não muda o nível.

## Consequências

- `udv_v1` continua reproduzível byte a byte: sem `semantic_unit` e `quote_extent` em `[evidence]`, a
  biblioteca faz exatamente o que fazia, e a seção `pipeline` da cobertura só ganha chaves novas quando
  as opções estão ligadas.
- `verify-udvs` passa a ler `semantic_unit` e `quote_extent` da configuração gravada na cobertura e
  confere a evidência de `udv_v2` com as mesmas regras.
- Um julgamento humano de `udv_v1` só vale para `udv_v2` quando a evidência é idêntica. O plano de
  aproveitamento e o comando de pontuação estão em `challenge/PIPELINE.md`.
- O corte de premissa UDV é calibrado com positivos cuja opinião contém a própria citação, como o corte
  do cosseno. Ele mede a separação entre a janela da citação e janelas sem relação, não a precisão sobre
  UDVs semânticas sem citação.
