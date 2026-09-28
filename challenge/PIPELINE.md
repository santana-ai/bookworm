# Pipeline de UDV: o que cada experimento decidiu e como rodar `udv_v2`

Este documento junta, em um lugar, os experimentos que formaram a construção das UDV (Unidades
Deliberativas Verificáveis, descritas em [`UDV.md`](UDV.md)) e o pipeline recomendado que sai deles,
`udv_v2`. Para cada experimento: o problema, o que foi comparado, uma tabela curta e a decisão. Todo
número cita o artefato de onde vem; os caminhos são relativos a `challenge/`, salvo quando indicado. A
descrição completa de cada experimento, com todos os números, está em
[`RELATORIO_EXPERIMENTOS.md`](RELATORIO_EXPERIMENTOS.md); os comandos, em [`README.md`](README.md).

Splits: manifesto `artifacts/splits/temporal_v1.json`, 144 audiências de treino, 32 de validação e 30
de teste. Nenhuma escolha deste documento usou o teste; onde há número de teste, ele foi medido depois
da escolha e não mudou nada.

## Dois trabalhos diferentes

Uma UDV semântica precisa de duas coisas, e os experimentos mostram que o melhor método é diferente
em cada uma.

- **Trabalho 1, achar o trecho.** Entre as falas da pessoa, qual trecho corresponde à opinião da
  matéria. É um problema de ordenação: o que importa é o trecho certo vir em primeiro. Aqui o cosseno
  do Serafim-335m venceu: nenhum outro recuperador, reordenador ou combinação o superou com
  significância (E1, retrieval_v2), e o reordenador Laya ficou pior.
- **Trabalho 2, quanto confiar no trecho achado.** Dado o trecho escolhido, se ele sustenta a opinião.
  É um problema de separação entre trechos que sustentam e trechos que não sustentam. Aqui o cosseno
  separa pouco (E5, AUC 0,730 na validação do NLI) e o verificador primário do E3x separa melhor (AUC
  0,876 na validação, 0,912 no teste).

Por isso `udv_v2` usa o cosseno para escolher o trecho e o verificador só para dar a confiança, num
campo separado que não muda o nível.

## Experimentos

### Calibração do corte do cosseno (`threshold_v1`)

**Problema.** O nível `semantic_match_high` depende de um corte de cosseno. O corte herdado (0,47)
não tinha origem registrada.

**Comparado.** Quatro regras, só no treino (183 citações confiáveis de 106 audiências; 142 mascaradas
de 87): `legacy_random` (ponto médio do quantil 0,25 dos positivos e do 0,75 de negativos sorteados de
outras audiências), `hard_negative` (negativos da mesma pessoa), `masked_hard_negative` e
`masked_top1_youden`. Intervalo por bootstrap de audiência, 1.000 réplicas.

| regra | corte | IC 95% |
|---|---|---|
| `legacy_random` | 0,4543 | 0,4366 a 0,4687 |
| `hard_negative` | 0,5019 | 0,5004 a 0,5416 |
| `masked_hard_negative` | 0,3798 | 0,3623 a 0,4075 |
| `masked_top1_youden` | 0,3928 | 0,3928 a 0,7725 |

Fonte: `artifacts/calibration/threshold_v1.json`, `rules.<regra>.threshold` e
`rules.<regra>.bootstrap.threshold`.

**Decisão.** `udv_v1` usa 0,45 (`legacy_random`). A regra primária declarada, `masked_top1_youden`, é
degenerada: só 12 das 142 consultas mascaradas têm o top-1 certo e o intervalo vai de 0,39 a 0,77.

### E1: qual recuperador acha o trecho (`retrieval_v1`)

**Problema.** Escolher o encoder ou o método léxico que põe o trecho de suporte em primeiro.

**Comparado.** 15 recuperadores contra `serafim_335m`, unidade sentença, em dois benchmarks: `nli`
(620 opiniões de validação cujos 4 chunks o especialista marcou como suficientes) e `masked_quotes`
(40 citações de validação com a citação mascarada na opinião). McNemar em acc@1 e troca de sinal por
audiência em MRR, com Holm por família.

| recuperador (validação, sentença) | nli acc@1 | nli MRR | p Holm MRR | masked MRR |
|---|---|---|---|---|
| `serafim_335m` (referência) | 0,8774 | 0,9295 | | 0,2612 |
| `serafim_335m_ir` | 0,8984 | 0,9435 | 1,0 | 0,2423 |
| `bge_m3` | 0,8839 | 0,9337 | 1,0 | 0,2430 |
| `serafim_900m` | 0,8823 | 0,9335 | 1,0 | 0,2604 |
| `hybrid_rrf` | 0,8742 | 0,9294 | 1,0 | 0,2051 |
| `bm25_hearing` | 0,8371 | 0,9020 | 0,042 | 0,1688 |
| `e5_large` | 0,7452 | 0,8435 | 0,001 | 0,2067 |

Fonte: `artifacts/experiments/retrieval/retrieval_v1/retrieval_v1_report.json`,
`summaries.<bench>.validation.sentence` e
`comparisons.<bench>.validation.retrievers_within_unit.sentence`.

**Decisão.** Nenhum recuperador supera `serafim_335m` com significância em nenhum benchmark; dois são
significativamente piores no `nli`. O encoder de produção fica.

### E2: qual unidade de trecho (`retrieval_v1`)

**Problema.** A sentença isolada pode cortar a fala que a matéria resumiu.

**Comparado.** Sentença, janelas de 2 e 3 sentenças consecutivas do mesmo turno e turno inteiro, com
`serafim_335m`, pela métrica `acc_at_1_minus_random` (acerto no top-1 menos o acerto esperado ao acaso,
que cresce quando a unidade é maior).

| unidade contra sentença | masked train | masked validação | nli train | nli validação |
|---|---|---|---|---|
| `window2` | +0,097 (Holm 0,007) | +0,179 (Holm 0,017) | +0,007 (Holm 0,252) | +0,014 (Holm 0,410) |
| `window3` | +0,200 (Holm 0,0003) | +0,254 (Holm 0,017) | -0,020 (Holm 0,019) | -0,019 (Holm 0,410) |

Fonte: `artifacts/experiments/retrieval/retrieval_v1/retrieval_v1_report.json`,
`comparisons.<bench>.<split>.units_within_retriever.serafim_335m`.

**Decisão.** `udv_v2` usa `window2`, pela regra do ADR 0006
(`../bookworm/docs/adr/0006-udv-v2-windows-full-quotes-and-verifier.md`): a unidade precisa ganhar
de `sentence` com Holm < 0,05 em algum benchmark no treino e na validação e não pode perder com
significância em nenhum. `window3` perde no `nli` do treino. A regra foi escrita depois de ler o E2, e
não existe teste direto entre `window2` e `window3`.

### retrieval_v2: reordenar o top 20 com o Laya

**Problema.** Saber se um modelo de decisão (Laya) melhora a ordenação quando reordena os 20 primeiros
do cosseno.

**Comparado.** `rerank_laya_p3` e `rerank_laya_p4` (duas perguntas ao Laya), `rerank_bge` e
`rerank_mmarco`, contra `serafim_335m`, unidade sentença.

| reordenador (validação) | nli MRR | p Holm | masked MRR |
|---|---|---|---|
| `serafim_335m` (referência) | 0,9295 | | 0,2612 |
| `rerank_laya_p4` | 0,8690 | 0,0004 | 0,2160 |
| `rerank_laya_p3` | 0,8645 | 0,0006 | 0,1999 |
| `rerank_mmarco` | 0,9300 | 1,0 | 0,2240 |
| `rerank_bge` | 0,9316 | 1,0 | 0,2293 |

Fonte: `artifacts/experiments/retrieval/retrieval_v2/retrieval_v2_report.json`,
`comparisons.<bench>.validation.retrievers_within_unit.sentence.items`.

**Decisão.** Resultado negativo. O Laya piora a ordenação com significância (MRR 0,869 contra 0,930 no
`nli`), e os outros reordenadores não mudam nada. O Laya é bom para julgar suporte (E3x), não para
ordenar trechos.

### E3 e E3x: um verificador de suporte (`nli_verifier`, `nli_verifier_exploration`)

**Problema.** O especialista do dataset julgou se cada opinião é inferível de 4 chunks recuperados. A
pergunta é se um modelo aberto reproduz esse julgamento, para servir de confiança.

**Comparado.** No E3 v1, cosseno, mDeBERTa XNLI e ASSIN2 e os 12 juízes LLM guardados no dataset. No
E3x, modelos Laya em português e depois de tradução para inglês, com agregações aprendidas por
validação cruzada no treino; o primário foi fixado pela regra do treino antes de ler a validação.

| sistema | validação ROC AUC [IC 95%] | teste ROC AUC [IC 95%] | teste kappa |
|---|---|---|---|
| `cosine_serafim.max` | 0,7302 [0,660; 0,798] | | |
| `xnli_mdeberta.max.entailment` | 0,7698 [0,696; 0,833] | 0,8009 [0,720; 0,864] | |
| `laya_multi_pt:panel:max` (referência do E3x) | | 0,8571 [0,790; 0,907] | 0,4985 |
| `laya_en_en:learned_cross_model_with_pt` (primário) | 0,8758 [0,822; 0,915] | 0,9122 [0,870; 0,945] | 0,5924 |

Fontes: validação do E3 v1 em `artifacts/experiments/nli_verifier/nli_verifier_v1/evaluation_report.json`
(`systems.<sistema>.evaluation.validation.ranking`); validação do primário em
`artifacts/experiments/nli_verifier_exploration/e3x_v2/confirmation.csv`; teste em
`artifacts/experiments/nli_verifier_exploration/e3x_v2/final_test.json` (`results`, `comparisons`).
O juiz LLM de referência, escolhido no treino entre os 12 guardados no dataset
(`prompt_1_gpt-4o-mini-2024-07-18`), tem kappa de validação 0,6909 (`judges` do mesmo relatório v1); o
maior kappa de validação entre os 12 é 0,7114 (`prompt_2_deepseek-chat`,
`nli_verifier_v2/evaluation_report.json`), escolhido olhando a validação e por isso não usado como
referência.

No teste, o primário supera o painel Laya em ROC AUC por +0,055 [0,023; 0,100], p de Holm 0,024, e o
XNLI por +0,111, p de Holm 0,020 (`final_test.json`, `comparisons`). Na validação, as diferenças não
passam Holm na família grande (p de Holm 0,78, `confirmation.csv`).

**Decisão.** O primário do E3x é o verificador adotado, com o corte do treino 0,7478
(`final_test.json`, `results.<primário>.threshold`). O kappa dele no teste (0,59) fica abaixo do kappa
de validação do juiz LLM de referência (0,69), então ele não substitui um juiz; ele ordena bem.

### Tradução (`translation`)

**Problema.** O primário lê premissa e hipótese em inglês; a tradução precisa ser estável.

**Comparado.** NLLB-200 distilled 600M contra M2M100 418M nos mesmos 41.599 trechos e 3.681 opiniões.

| tradutor | trechos com saída degenerada | opiniões com saída degenerada | teste ROC AUC do Laya en_en aprendido |
|---|---|---|---|
| NLLB | 56 | 0 | 0,9074 |
| M2M100 | 103 | 1 | 0,9021 |

Fontes: `artifacts/experiments/translation/translation_v1_report/report.json` e
`artifacts/experiments/translation/translation_v1_m2m100_report/report.json` (`segments.flags` e
`opinions.flags`); `artifacts/experiments/nli_verifier_exploration/e3x_v2/final_test.json`
(`laya_en_en:learned_cross_model` e `laya_en_en_m2m100:learned_cross_model`).

**Decisão.** NLLB, por ter metade das saídas degeneradas; o efeito no AUC é pequeno e está dentro dos
intervalos. A checagem manual de tradução (`artifacts/validation/translation_spot_check_v2/`) não foi
preenchida.

### E5: sinais de confiança sobre o cosseno (`confidence_v1`)

**Problema.** Achar um sinal que diga quando o top-1 do cosseno está certo, sem modelo novo.

**Comparado.** `top1_score`, margem para o segundo, z-score, entailment XNLI e combinações logísticas,
por AURC na validação (620 consultas do `nli`, população com mais de um candidato).

| sinal (nli, validação) | ROC AUC | AURC |
|---|---|---|
| `top1_score` (referência) | 0,629 | 0,0797 |
| `margin` | 0,677 | 0,0685 |
| `logistic.scores_entailment` | 0,681 | 0,0680 |
| `entailment.xnli_mdeberta` | 0,579 | 0,0939 |

Fonte: `artifacts/experiments/confidence/confidence_v1/confidence_v1_report.json`,
`results.nli.serafim_335m.sentence.evaluate.validation.signal_metrics.multi_candidate`. A diferença de
AURC da margem é -0,011 [-0,028; 0,005], intervalo que cruza zero (`difference_vs_reference`).

**Decisão.** Nenhum sinal supera `top1_score` pela regra declarada. Sinais derivados do próprio
cosseno não resolvem o Trabalho 2.

### confidence_v2: o verificador como confiança e avaliadores da literatura

**Problema.** Confirmar que o primário do E3x é uma confiança melhor que o cosseno e compará-lo com
avaliadores publicados (MiniCheck, FactCG, AlignScore, HHEM, BGE reranker).

**Comparado.** E-A: validação do benchmark NLI, rótulo do especialista.

| base | cosseno ROC AUC | primário ROC AUC | Δ [IC 95%] | p |
|---|---|---|---|---|
| E-A, validação (698 opiniões) | 0,730 | 0,876 | +0,146 [0,085; 0,216] | Holm 0,012 |

Fontes: `artifacts/experiments/confidence_v2/ea_report.json` e [`CONFIDENCE_V2.md`](CONFIDENCE_V2.md).

**Decisão.** O primário é a confiança adotada. Na estimativa pontual, nenhum avaliador da literatura tem
ROC AUC maior que o primário, mas a comparação direta do primário com cada avaliador não foi declarada
nem testada (`CONFIDENCE_V2.md`, E-A). A parte E-B do experimento, sobre UDVs com rótulos que não eram
humanos, não está nesta versão (`CONFIDENCE_V2.md`).

### Camada do verificador sobre `udv_v1`

**Problema.** Ver como a decisão do primário se distribui nas UDVs reais, com premissa de uma sentença.

| nível de `udv_v1` | com evidência | passa no corte 0,7478 |
|---|---|---|
| `quote_found` | 277 | 28,2% |
| `semantic_match_high` | 1.785 | 23,0% |
| `semantic_match_weak` | 43 | 0% |
| todas | 2.105 | 23,2% |

Fonte: `artifacts/udv/udv_v1_verifier_report.json` e `artifacts/udv/udv_v1_verifier.jsonl`. A
correlação de Spearman entre a probabilidade e o cosseno é 0,69 no nível `semantic_match_high` e 0,71
nos dois níveis semânticos juntos (`cosine_relation.spearman` do mesmo relatório).

**Decisão.** O corte do benchmark foi ajustado com premissas de 4 chunks; com uma sentença, ele reprova
72% das citações literais, o que mostra que a premissa curta sozinha deixa o verificador conservador.
`udv_v2` alonga a premissa (citação inteira e janela) e recalibra um corte próprio para premissas de UDV.

### E6: casamento aproximado de citações e nomes (`fuzzy_v1`)

**Problema.** Citações com pequenas diferenças de grafia e nomes que a resolução exata não acha.

| medida (treino) | valor |
|---|---|
| opiniões elegíveis sem citação confiável exata | 436 |
| com casamento aproximado de caracteres ≥ 80 | 317 |
| concordância desse trecho com o top-1 do cosseno | 0,527 [0,476; 0,578] |
| taxa ao acaso (outra pessoa da mesma audiência) ≥ 80 | 0,0282 [0,0149; 0,0432] |
| participantes não resolvidos | 39 (80 opiniões) |
| resolvidos por `token_set_ratio` ≥ 80 | 17 (37 opiniões) |

Fonte: `artifacts/experiments/fuzzy/fuzzy_v1_report.json` (`quotes.train.funnel`,
`quotes.train.gains.char.80.0`, `names.train.by_rule.token_set_ratio.80.0`).

**Decisão.** Fica desligado. As planilhas de revisão
(`artifacts/experiments/fuzzy/fuzzy_v1_quotes_review.jsonl`, 458 linhas, e
`artifacts/experiments/fuzzy/fuzzy_v1_names_review.jsonl`, 70 linhas) não têm julgamento, e metade dos
trechos aproximados discorda do cosseno.

### Regras de ligação UDV-ator

**Problema.** As contagens de avaliação dos perfis de ator dependem de qual regra liga uma UDV a um ator.

| regra | UDVs ligadas em `udv_v1` | UDVs ligadas em `udv_v2` |
|---|---|---|
| turnos atribuídos | 2.104 | 2.104 |
| turno de evidência | 2.099 | 2.098 |
| divergências entre as duas | 0 | 0 |

Fontes: `../bookworm/docs/actors.md`, seção "Duas regras de ligação", e
`artifacts/udv/udv_v2_downstream_report.json` (`runs.<rodada>.link_rules`). Nas duas rodadas, 106 UDVs de
teste ficam ligadas a atores com perfil (`artifacts/actor_profiles/train_speeches_stats.json` e
`artifacts/actor_profiles/train_speeches_stats_udv_v2.json`).

**Decisão.** As duas regras nunca escolhem atores diferentes; o arquivo de ligações usa a primeira.

## Pipeline recomendado: `udv_v2`

Decisões registradas no ADR 0006 (`../bookworm/docs/adr/0006-udv-v2-windows-full-quotes-and-verifier.md`)
e configuração em `configs/udv_v2.toml`.

1. **Resolução de pessoa.** A mesma de `udv_v1` (nome contido no cabeçalho do turno).
2. **Citação.** Casamento de prefixo de 6 ou mais palavras da citação da opinião dentro dos turnos da
   pessoa, como em `udv_v1`. A evidência se estende do prefixo até o fim da citação, achado por um sufixo
   de 6, 4 ou 3 palavras a no máximo 2 vezes o tamanho da citação; sem sufixo, cobre tantas partes de
   sentença quantas a citação tem. A evidência nunca sai do turno e guarda offsets exatos.
3. **Busca semântica.** Sem citação confiável, a unidade candidata é `window2` (duas sentenças
   consecutivas do mesmo turno), codificada pelo `serafim_335m`; vence a janela de maior cosseno.
4. **Nível.** `semantic_match_high` quando o cosseno da janela é pelo menos 0,50; abaixo disso,
   `semantic_match_weak`. `no_evidence` e `person_not_resolved` como antes.
5. **Confiança.** Toda UDV com evidência recebe a probabilidade do primário do E3x sobre o texto da
   evidência de `udv_v2`, e duas decisões: no corte do benchmark (0,7478) e no corte de premissa UDV
   (0,2429). A decisão fica no arquivo lateral e não muda o nível.

### Cortes recalibrados no treino

Os dois cortes usam só as 144 audiências de treino; a checagem de vazamento das duas calibrações passou
(`leak_check.passed`). Os pares são os mesmos: 183 opiniões com citação confiável de 106 audiências, cada
uma contra a janela da própria citação (positivo), uma janela sorteada de outra audiência do treino
(`legacy_random`) e uma janela sorteada da mesma pessoa sem a citação (`hard_negative`).

| corte | regra | valor | IC 95% (bootstrap de audiência) |
|---|---|---|---|
| cosseno de janela, adotado | `legacy_random` | 0,5015, arredondado para 0,50 | 0,4858 a 0,5129 |
| cosseno de janela | `hard_negative` | 0,5720 | 0,5685 a 0,6070 |
| cosseno de janela | `masked_hard_negative` | 0,4432 | 0,4101 a 0,4671 |
| cosseno de janela | `masked_top1_youden` | 0,6878 | 0,5278 a 0,8020 |
| verificador, premissa UDV, adotado | `legacy_random` | 0,2429 | 0,2210 a 0,2782 |
| verificador, premissa UDV | `hard_negative` | 0,2722 | 0,2485 a 0,3190 |
| verificador, benchmark NLI | `max_f1_not_inferable` | 0,7478 | |

Fontes: `artifacts/calibration/threshold_v2.json` (`rules.<regra>.threshold` e
`rules.<regra>.bootstrap.threshold`) e `artifacts/calibration/udv_verifier_threshold_v1.json` (mesmas
chaves, `train_threshold`). No corte do cosseno, nenhum positivo fica abaixo e nenhum negativo aleatório
fica acima (`positives_below_threshold` e `negatives_at_or_above_threshold` de `legacy_random`). No
corte do verificador, 14 dos 183 positivos ficam abaixo e 2 negativos aleatórios ficam acima. No corte
0,7478, só 86 dos 183 positivos (47,0%) passam, contra 0 negativos aleatórios e 4 `hard_negative`
(`pairs_at_train_threshold`). Isto é, com premissa de uma citação ou janela, o corte do benchmark reprova
mais da metade dos pares em que a opinião cita literalmente a janela.

Limite dos dois cortes: os positivos são opiniões que contêm a própria citação. Os cortes medem a
separação entre a janela da citação e janelas sem relação, e não a precisão sobre UDVs semânticas de
opiniões sem citação. O bootstrap do corte do verificador reamostra audiências com os pares fixos, porque
cada negativo novo precisaria de tradução e de pontuação (`bootstrap.method`).

### Resultados de `udv_v2`

Fonte desta subseção, salvo indicação: `artifacts/udv/udv_v2_analysis.json`, gerado por
`uv run python -m utils.udv_v2_analysis analyze`.

**Níveis** (`tiers`):

| nível | `udv_v1` | `udv_v2` | `udv_v2` treino | `udv_v2` validação | `udv_v2` teste |
|---|---|---|---|---|---|
| `quote_found` | 277 | 277 | 183 | 53 | 41 |
| `semantic_match_high` | 1.785 | 1.744 | 1.217 | 233 | 294 |
| `semantic_match_weak` | 43 | 84 | 54 | 19 | 11 |
| `no_evidence` | 8 | 8 | 2 | 0 | 6 |
| `person_not_resolved` | 90 | 90 | 80 | 3 | 7 |

`udv_v2` tem 2.203 registros, 1.020 de 1.065 pessoas resolvidas e 2.105 evidências com offsets
localizados (`artifacts/udv/udv_v2_coverage.json`). `bookworm verify-udvs --config configs/udv_v2.toml
--run-name udv_v2 --baseline artifacts/udv/udv_v1.jsonl` não acusa nenhum problema
(`artifacts/udv/udv_v2_verify.json`, `problems` vazio), e reconstruir `udv_v2` com o código atual
reproduz o arquivo byte a byte (sha256 `02f42b28...`).

**Diferença por registro contra `udv_v1`** (`diff`): 234 registros têm evidência idêntica (texto,
`start_char`, `end_char` e `speaker_turn`) e 1.969 mudaram. `records_changed_any` dá 1.970 porque conta
também a UDV em que só o nível mudou (`udv-6-0-6`, cosseno 0,4838, entre o corte antigo 0,45 e o novo
0,50). `udv_v2_verify.json` (`baseline_diff.evidence_changed`) também dá 1.970, por outro motivo: ele
compara o objeto `evidence` inteiro, incluindo `score`, e conta `udv-117-0-0`, cuja sentença e offsets
são os mesmos e cujo cosseno difere em 1,2e-7 (0,69387329 contra 0,69387317). As duas rodadas usaram
versões diferentes de torch e sentence-transformers (seção 9.1 do relatório); a causa dessa diferença
de arredondamento não foi isolada.

| tipo de mudança | registros |
|---|---|
| mesma evidência, mesmo nível | 233 |
| mesma evidência, nível mudou | 1 |
| citação estendida (o texto de `udv_v1` está dentro do de `udv_v2`) | 152 |
| janela que contém a sentença de `udv_v1` | 1.343 |
| janela em outro lugar da fala | 474 |

Das 277 citações, 125 ficaram iguais, porque a citação inteira já cabia na sentença do prefixo, e 152
foram estendidas. Nas semânticas, 474 de 1.828 (25,9%) passaram a apontar para outro trecho da fala.
Entre os níveis, 43 UDVs passaram de `semantic_match_high` para `semantic_match_weak` e 2 fizeram o
caminho inverso (`tier_moves`).

**Tamanho da evidência de citação** (`quote_evidence`, 277 registros):

| medida | `udv_v1` | `udv_v2` |
|---|---|---|
| palavras, mediana (quantis 0,1 e 0,9) | 27 (12; 54) | 45 (21; 81) |
| partes de sentença, média | 1,02 | 1,92 |
| evidências que contêm as 3 últimas palavras da citação | 82 | 172 |

Em 139 das 277 opiniões, a citação tem mais de uma parte de sentença. Mesmo com a extensão, 105
evidências de `udv_v2` não contêm as 3 últimas palavras da citação: ou o fim da citação da matéria não
aparece literalmente na fala, ou nenhum sufixo foi achado e valeu a contagem de sentenças. A maior
evidência tem 198 palavras (`udv-98-1-0`), porque a última sentença da citação é longa na transcrição.
As evidências semânticas passaram de 161 para 274 caracteres na mediana (`semantic_evidence`).

**Verificador** (`verifier`; linhas por UDV em `artifacts/udv/udv_v2_verifier.jsonl`, relatório em
`artifacts/udv/udv_v2_verifier_report.json`, gerados por
`uv run python -m utils.udv_verifier --config configs/udv_v2_verifier.toml translate`, `score` e
`apply`). O reajuste do primário no treino do benchmark reproduziu C, coeficientes e corte de
`final_test.json` (`refit_check`).

Mediana da probabilidade do primário por nível e split (`distributions_by_tier_and_split`):

| nível | todas | treino | validação | teste |
|---|---|---|---|---|
| `quote_found` | 0,724 | 0,703 | 0,711 | 0,833 |
| `semantic_match_high` | 0,586 | 0,595 | 0,564 | 0,576 |
| `semantic_match_weak` | 0,126 | 0,125 | 0,133 | 0,125 |

Parcela que passa em cada corte (`passing_train_threshold`, `passing_udv_threshold` e, para `udv_v1`,
`udv_v1_passing_train_threshold`):

| nível | n (`udv_v2`) | `udv_v1` no corte 0,7478 | `udv_v2` no corte 0,7478 | `udv_v2` no corte 0,2429 |
|---|---|---|---|---|
| `quote_found` | 277 | 28,2% | 48,0% | 92,4% |
| `semantic_match_high` | 1.744 | 23,0% | 39,6% | 84,6% |
| `semantic_match_weak` | 84 | 0% | 2,4% | 21,4% |
| todas | 2.105 | 23,2% | 39,2% | 83,1% |

Por split, no corte 0,7478: treino 572 de 1.454, validação 110 de 305, teste 144 de 346; no corte
0,2429: treino 1.198, validação 257, teste 295. Nas 1.969 UDVs cuja evidência mudou, a média da
probabilidade subiu de 0,478 para 0,590 e 1.312 subiram; nas 152 citações estendidas, de 0,431 para 0,715
(`v1_vs_v2`). As 136 UDVs pontuadas com evidência igual têm a mesma probabilidade nas duas execuções
(diferença máxima 0).

**Conferência do escore** (`evidence_score_check` de `udv_v2_verifier_report.json`). A camada recalcula o
cosseno da janela que o verificador leu e o compara com `evidence.score` da UDV. Em 1.771 das 1.828 UDVs
semânticas a diferença é 0. Nas outras 57 (`encoded_text_differs`), o intervalo da janela na transcrição
inclui partes que não são candidatas (trechos de menos de 4 palavras, rubricas ou a divisão em "Sr."), que
o encoder não leu; a premissa do verificador é mais longa que o texto codificado, e a diferença chega a
0,0902.

**Nível do cosseno contra a decisão do verificador**, só UDVs semânticas (`cosine_tier_agreement`):

| corte do verificador | alta e passa | alta e não passa | fraca e passa | fraca e não passa | concordância | kappa |
|---|---|---|---|---|---|---|
| 0,7478 | 691 | 1.053 | 2 | 82 | 0,423 | 0,054 |
| 0,2429 | 1.476 | 268 | 18 | 66 | 0,844 | 0,262 |

O Spearman entre a probabilidade e o cosseno é 0,58 em `semantic_match_high` e 0,62 nos dois níveis
semânticos (`spearman_with_cosine`). Os dois sinais concordam pouco sobre quais UDVs semânticas são
confiáveis: no corte do benchmark, 60% das UDVs de nível alto não passam; no corte de premissa UDV, 268
de nível alto não passam e 18 de nível fraco passam. Qual dos dois acerta mais só a anotação humana das
UDVs de `udv_v2` pode dizer.

### Validação humana: o que vale para `udv_v2`

A amostra `artifacts/validation/human_validation_v1_udv_v1/` foi sorteada de `udv_v1` nas audiências de
teste. Um julgamento vale para `udv_v2` só quando a evidência mostrada ao anotador é idêntica em
`udv_v2`. O plano está em `artifacts/udv/udv_v2_annotation_plan.json`, gerado pelo mesmo comando
`analyze`.

| estrato de `udv_v1` | itens | evidência idêntica (herda o rótulo) | mudou (precisa de novo julgamento) |
|---|---|---|---|
| `direct_quote` | 35 | 17 | 18 |
| `semantic_match_high` | 65 | 0 | 65 |
| `semantic_with_short_quote` | 15 | 0 | 15 |
| `semantic_match_weak` | 6 | 0 | 6 |
| `speaker_check` | 6 | 6 | 0 |

Dos 121 itens `trecho_sustenta`, só 17 (todos citações) mantêm a evidência; nenhum item semântico
mantém, porque a janela sempre difere da sentença julgada. Os 17 itens herdados não são uma amostra
aleatória de `udv_v2`: são as citações que já cabiam numa sentença.

Os 104 itens que mudaram estão na planilha suplementar
`artifacts/validation/human_validation_v1_udv_v2_supplement/annotation.csv`, com a mesma opinião e a
evidência de `udv_v2`, em nova ordem e com novos identificadores (a correspondência fica em
`annotation_key.json`, na mesma pasta). Nenhuma linha dela foi julgada.

**Comando de precisão.** Com as planilhas preenchidas e as chaves:

```bash
uv run python -m utils.udv_v2_analysis score-annotation --final-test \
  --annotation artifacts/validation/human_validation_v1_udv_v1/annotation.csv \
  --supplement-dir artifacts/validation/human_validation_v1_udv_v2_supplement
```

Ele reusa as funções de `utils/precision_report.py` (precisão estrita e tolerante por estrato, intervalo
de Wilson a 95%, critérios congelados na chave) para `udv_v1`, para os itens herdados de `udv_v2` e, com
`--supplement-dir`, para `udv_v2` inteira: o item sem mudança conta com o rótulo de `udv_v1` e o item com
mudança, só com o rótulo da planilha suplementar. Com a
planilha incompleta, conta só as linhas com rótulo válido, informa "n julgados de N" no total e por
estrato, lista rótulos fora do conjunto permitido sem contá-los e marca o relatório e cada critério como
`INTERIM`. Cada execução grava um arquivo novo, `artifacts/udv/udv_v2_precision_<interim|final>_<UTC>.json`,
que nunca é sobrescrito. `--final-test` é obrigatório porque a amostra é do teste; nada foi escolhido a
partir dela.

Critérios declarados em 2026-09-23 (`criteria` da chave): limite inferior de Wilson da precisão estrita de
`direct_quote` de pelo menos 0,90 e da precisão tolerante de `semantic_match_high` de pelo menos 0,75.

**Resultado humano provisório.** Os números intermediários da planilha em preenchimento, o estado de
cada critério e o que ainda falta anotar estão no relatório, seção 7.2. O arquivo
`artifacts/udv/udv_v2_precision_interim_20260928T143948Z.json` é uma leitura anterior da mesma planilha
(31 linhas de `semantic_match_high` julgadas e um rótulo inválido) e foi substituído por esse recálculo.

## Pendências e limitações

- **Nomes aproximados.** O casamento aproximado de nomes (E6) resolveria 17 dos 39 participantes não
  resolvidos do treino (37 opiniões) com `token_set_ratio` de pelo menos 80
  (`artifacts/experiments/fuzzy/fuzzy_v1_report.json`, `names.train.by_rule.token_set_ratio.80.0`). Fica
  desligado até a planilha `artifacts/experiments/fuzzy/fuzzy_v1_names_review.jsonl` ser julgada.
- **Libras.** Os 8 registros `no_evidence` de `udv_v2` são de 3 participantes das audiências 53 e 111
  cuja única fala registrada é a marcação "(Manifestação em LIBRAS.)"; a tradução está nos turnos do
  intérprete (`artifacts/udv/udv_v2.jsonl`, nível `no_evidence`; `HEARING_ACTORS.md` conta 6
  participantes que falaram em Libras). Atribuir a fala do intérprete a eles daria evidência a no máximo
  8 das 2.203 opiniões. Não implementado.
- **Validação humana de `udv_v2`.** As 104 linhas da planilha suplementar (os itens semânticos e as 18
  citações estendidas da amostra) precisam de julgamento com a evidência de `udv_v2`; sem isso, não há
  precisão humana de `udv_v2` além das citações herdadas.
- **Domínio do verificador.** O primário foi ajustado com premissas de 4 chunks recuperados; nas UDVs a
  premissa é uma citação ou uma janela. O corte de premissa UDV corrige a escala no treino, mas o efeito
  da mudança de domínio na ordenação não foi medido com rótulo humano.
- **`window2` contra `window3`.** O E2 não compara as duas janelas diretamente; a escolha segue a regra
  do ADR 0006, escrita depois de ler o E2.
