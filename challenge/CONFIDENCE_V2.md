# confidence_v2: sinais de confiança para a evidência de uma UDV

## Pergunta

O construtor de UDVs registra, para cada evidência semântica, o cosseno do codificador de produção
(`PORTULAN/serafim-335m-portuguese-pt-sentence-encoder`) entre a proposição e a frase escolhida. O
cosseno mede proximidade de sentido; ele não diz se a frase afirma o que a proposição afirma. O
verificador primário do E3x (`laya_en_en:learned_cross_model_with_pt`, `utils/udv_verifier.py`) foi
treinado para essa segunda pergunta. Este experimento mede se o verificador é um sinal de confiança
melhor que o cosseno e como ele se compara com os avaliadores de suporte mais usados e mais recentes
na literatura de grounding e consistência factual.

A declaração completa (candidatos, revisões, premissas, métricas, famílias de Holm e regra do smoke)
está em `configs/confidence_v2.toml`, escrita antes de qualquer escore novo. O código está em
`utils/confidence_v2.py`, `utils/grounding_scorers.py` e `utils/grounding_models.py`; os testes, com
avaliadores falsos, em `tests/test_confidence_v2.py`. Os artefatos ficam em
`artifacts/experiments/confidence_v2/`.

## Levantamento da literatura

Levantamento feito em 2026-09-27 nos artigos, repositórios de código e cartões de modelo do Hugging
Face. A média no LLM-AggreFact é a acurácia balanceada média sobre os 11 conjuntos do placar público
(llm-aggrefact.github.io), recalculada a partir dos dados da página; nenhum dos conjuntos é em
português. Nenhum avaliador abaixo documenta treino em português: os que rodam em inglês leem a
tradução NLLB já usada pelo E3.

| Avaliador | Artigo, ano | Checkpoint (revisão fixada) | Licença | Idiomas | LLM-AggreFact | Rodou aqui |
|---|---|---|---|---|---|---|
| SummaC (ZS, Conv) | Laban et al., TACL 2022 | sem checkpoint próprio; usa modelos NLI (padrão `tals/albert-xlarge-vitaminc-mnli`) | código Apache-2.0; licença do NLI padrão não confirmada | en | 67,7 (ZS), 61,0 (Conv) | não |
| TRUE (T5 NLI) | Honovich et al., NAACL 2022 | `google/t5_xxl_true_nli_mixture` (T5-XXL, 11B) | Apache-2.0 | en | 61,1 | não |
| AlignScore | Zha et al., ACL 2023 | `yzha/AlignScore`, `AlignScore-large.ckpt` (`8509e78`) sobre `FacebookAI/roberta-large` (`722cf37`) | MIT | en | 70,5 | sim |
| MiniCheck | Tang, Laban e Durrett, EMNLP 2024 | `lytang/MiniCheck-Flan-T5-Large` (`96eafd0`), `lytang/MiniCheck-DeBERTa-v3-Large` (`2f2d01a`), `lytang/MiniCheck-RoBERTa-Large` | MIT | en | 75,0 (FT5), 73,1 (DeBERTa), 73,5 (RoBERTa) | FT5 e DeBERTa |
| Bespoke-MiniCheck-7B | idem, 2024 | `bespokelabs/Bespoke-MiniCheck-7B` (InternLM2.5-7B) | CC BY-NC 4.0 | en | 77,4 (1º) | não |
| HHEM-2.1-open | Vectara, 2024 | `vectara/hallucination_evaluation_model` (`8e4a2e6`), base `google/flan-t5-base` (`7bcac57`) | Apache-2.0 | en (a versão 2.3, com português, é fechada) | 71,8 | sim |
| FactCG | Lei et al., NAACL 2025 | `yaxili96/FactCG-DeBERTa-v3-Large` (`0430e35`) | MIT | en | 75,6 | sim |
| Granite Guardian (groundedness) | Padhi et al., arXiv 2412.07724, 2024 | `ibm-granite/granite-guardian-3.2-3b-a800m` (`3de033d`); 3.3-8b no placar | Apache-2.0 | en | 76,5 (3.3-8B) | não, por decisão |
| LLM como juiz (RAGAS, G-Eval, P(sim)) | Es et al., EACL 2024 demo; Liu et al., EMNLP 2023 | qualquer LLM instruído; declarado `Qwen/Qwen3-4B-Instruct-2507` (`cdbee75`), carregado só em pares de exemplo | Apache-2.0 (Qwen3) | multilíngue | 72,8 (Qwen2.5-7B) a 77,2 (Claude-3.5 Sonnet) | não, por decisão |
| bge-reranker-v2-m3 | BAAI, 2024 | `BAAI/bge-reranker-v2-m3` (`953dc6f`) | Apache-2.0 | multilíngue | não listado | sim, em português |
| LettuceDetect | Kovács e Recski, arXiv 2502.17125, 2025; v2 em arXiv 2607.00895, 2026 | `KRLabsOrg/lettucedect-v2-mmbert-base` | MIT (v1), Apache-2.0 (v2) | v2: en, de, fr, es, it, pl, zh; português não confirmado | não listado | não |
| Lynx | Ravi et al., arXiv 2407.08488, 2024 | `PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct` | CC BY-NC 4.0 | en | não listado | não |

O problema que cada um resolve, e como o escore é lido:

- **SummaC.** Modelos NLI aplicados a um documento inteiro contra um resumo inteiro funcionam mal,
  porque foram treinados com pares de frases. O SummaC aplica o NLI por frase, monta uma matriz de
  probabilidades (frases do documento por frases do resumo), toma o máximo por frase do resumo e a
  média entre elas (ZS) ou passa a matriz por uma convolução aprendida (Conv).
- **TRUE.** As métricas de consistência factual eram avaliadas de formas incompatíveis entre si. O
  TRUE padroniza 11 conjuntos como tarefas binárias medidas por ROC AUC e mostra que um NLI grande
  (T5-11B) é uma linha de base forte. Entrada `premise: ... hypothesis: ...`, saída `1` ou `0`.
- **AlignScore.** Métricas treinadas em uma tarefa estreita não generalizam. O AlignScore treina uma
  única função de alinhamento entre textos com 4,7 milhões de exemplos de 7 tarefas (NLI, QA,
  paráfrase, verificação de fatos, sumarização e outras). No modo `nli_sp`, o contexto é dividido em
  trechos de cerca de 350 palavras e a afirmação em frases; o escore é a probabilidade da classe
  "alinhado" da cabeça de 3 classes, máxima entre trechos e média entre frases.
- **MiniCheck.** Checar a saída de um LLM contra documentos com GPT-4 é caro. O MiniCheck treina
  modelos pequenos com dados sintéticos que exigem juntar várias frases do documento e chega perto do
  GPT-4 no LLM-AggreFact a um custo cerca de 400 vezes menor. FT5: entrada
  `predict: documento</s>afirmação`, um passo do decodificador, softmax entre os tokens 3 e 209,
  escore = probabilidade do 209. DeBERTa: `documento[SEP]afirmação` como um só texto, softmax da
  cabeça de 2 classes. Documentos longos são quebrados em trechos e vale o máximo.
- **HHEM-2.1-open.** Detecta quando a resposta de um sistema RAG ou um resumo afirma algo que a
  fonte não sustenta, num modelo pequeno o bastante para produção. Prompt fixo
  `Determine if the hypothesis is true given the premise?`, escore = probabilidade da classe
  "consistente".
- **FactCG.** Os dados sintéticos dos checadores anteriores raramente pediam raciocínio em mais de um
  passo. O FactCG gera afirmações a partir de caminhos num grafo extraído do documento. Entrada: o
  modelo de instrução `... based on the paragraph above can we conclude that "..."?`, escore =
  softmax da cabeça de 2 classes.
- **Granite Guardian.** Um só modelo de guarda para riscos de segurança e de RAG (relevância do
  contexto, groundedness, relevância da resposta). Na família 3.x a pergunta de groundedness é
  respondida com "Yes" (há risco) ou "No"; o suporte é 1 - P(Yes).
- **LLM como juiz.** Um LLM instruído é perguntado se o trecho sustenta a afirmação. O RAGAS
  decompõe a resposta em afirmações e conta as que o LLM julga sustentadas; o G-Eval pondera as notas
  pela probabilidade de cada token de nota; a variante comum com logprobs lê P(sim) numa pergunta
  sim/não, como fazem o Bespoke-MiniCheck-7B e o Granite Guardian. É um método comum na literatura.
  Não foi rodado aqui, por decisão: o Laya, que responde perguntas do mesmo tipo com probabilidades a
  um custo muito menor, foi usado no seu lugar com várias perguntas (seção sobre o Laya abaixo). O
  Qwen3-4B e o Granite Guardian 3.2-3B chegaram a ser carregados uma vez, em oito pares de exemplo que
  não são do conjunto de dados, para checar carregamento e tokens de resposta.
- **bge-reranker-v2-m3.** Reordena passagens por relevância para busca multilíngue. Mede relevância
  e não implicação; entra porque é o sinal de confiança que um pipeline de recuperação já tem. Escore =
  logit de relevância bruto, lendo o texto em português.
- **LettuceDetect e Lynx.** Detectores de alucinação em RAG. O LettuceDetect marca trechos não
  sustentados token a token (exigiria uma regra de agregação própria, e o português não está entre os
  idiomas confirmados); o Lynx é um LLM de 8B ou 70B com licença não comercial. Nenhum dos dois foi
  rodado.

Motivos de exclusão, além da decisão sobre LLMs: Bespoke-MiniCheck-7B e Lynx têm licença não
comercial e não cabem na memória de GPU disponível (um modelo de 7B em bfloat16 tem cerca de 15 GB de
pesos, acima do limite de `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` nesta máquina de 24 GB); TRUE (11B)
pelo mesmo motivo de memória; SummaC porque é uma forma de agregar NLI por frase, o que as premissas
curtas deste experimento já fazem com os NLIs existentes; MiniCheck-RoBERTa por ter só 512 posições e
já haver duas variantes do MiniCheck.

## O que rodou e o que foi descartado

A regra do smoke (`[smoke]` da configuração) foi aplicada a 50 opiniões de treino antes de qualquer
escore completo; a decisão está em `artifacts/experiments/confidence_v2/smoke/decision.json` e em
`[smoke_decision]`. Todos os candidatos novos carregaram na revisão fixada, passaram na sonda (o HHEM
reproduz os sete valores do cartão do modelo) e ficaram dentro do orçamento de 5 horas projetadas.

| Candidato | Leitura | Pares E-A | Tempo E-A (s) | Truncamento |
|---|---|---|---|---|
| `minicheck_ft5` | inglês (NLLB) | 18.072 | 2.950 | nenhum |
| `minicheck_deberta` | inglês (NLLB) | 18.072 | 6.747 | nenhum |
| `factcg_deberta` | inglês (NLLB) | 15.896 novos + 2.426 do cache | 7.254 | nenhum |
| `alignscore_large` | inglês (NLLB), sem premissa concatenada | 14.444 | 1.269 | 13 de 14.712 trechos acima de 512 tokens |
| `hhem_open` | inglês (NLLB) | 18.072 | 2.348 | nenhum |
| `bge_reranker` | português | 18.072 | 6.266 | 72 de 3.678 premissas concatenadas acima de 1024 tokens |
| Laya, 4 perguntas novas, `laya_multi_pt` | português | 72.288 respostas | 7.209 | não se aplica |
| Laya, 4 perguntas novas, `laya_en_en` | inglês (NLLB) | 72.288 respostas | 14.153 | não se aplica |

Os tempos saem dos relatórios `scores/<candidato>_ea_report.json`. A vazão real foi menor que a do
smoke (o `bge_reranker` projetava 0,71 h e levou 1,74 h; as perguntas novas do `laya_en_en` projetavam
2,64 h e levaram 3,93 h); a decisão de rodar já estava registrada e não depende do tempo real. Todos os
45.509 textos em inglês necessários já estavam no cache de tradução (`translate_report.json`).

Não rodaram, com o motivo:

- **Qualquer LLM como juiz** (Qwen3-4B-Instruct-2507, o reserva Qwen3-1.7B, Bespoke-MiniCheck-7B):
  decisão do usuário registrada em `not_run_amendment` antes de qualquer escore; o Laya com várias
  perguntas ocupa esse lugar.
- **Granite Guardian 3.2-3B**: é um LLM de guarda lido por P(Yes), descartado pela mesma decisão.
- **Bespoke-MiniCheck-7B, Lynx e TRUE (11B)**: licença não comercial (os dois primeiros) e memória de
  GPU insuficiente para 7B a 11B parâmetros nesta máquina.
- **SummaC**: é uma regra de agregação de NLI por frase; as premissas já são trechos curtos e os NLIs
  do E3 já estão na comparação.
- **MiniCheck-RoBERTa**: 512 posições e duas variantes do MiniCheck já presentes.
- **LettuceDetect**: marca trechos token a token, exigiria uma agregação própria, e o português não está
  entre os idiomas confirmados.

## Como ler as métricas

Todas as métricas tratam o sinal como confiança: valores altos devem ir para os itens corretos. ROC AUC
mede a ordenação sem limiar. AURC é o risco médio (fração de negativos entre os aceitos) ao longo de
todas as coberturas; quanto menor, melhor. A precisão a 80% e a 90% de cobertura é a fração de
positivos entre os 80% ou 90% itens de maior escore, com o corte feito no próprio conjunto avaliado, e
por isso descreve a ordenação, não um limiar implantável. Os intervalos são percentis de 5.000
reamostragens de audiências (semente 42). Cada comparação usa as mesmas reamostragens nos dois lados;
Δ é o candidato menos o cosseno (em AURC, Δ negativo favorece o candidato); o p é o bootstrap
bilateral e "Holm" é o p ajustado dentro da família declarada.

## E-A: benchmark NLI, rótulo de especialista

Premissa: os 4 trechos recuperados, com o escore da opinião igual ao máximo entre trechos. Positivo =
`label_inferable` verdadeiro (inferível dos 4 trechos segundo o especialista), que não é verdade global.
O conjunto de teste do benchmark não foi lido. A família de Holm principal tem 30 comparações (15
candidatos contra o cosseno em ROC AUC e AURC); as duas combinações por posto formam uma família
separada de 4 comparações. O verificador primário foi reajustado no treino exatamente como em
`utils.udv_verifier.fit_primary`, e o reajuste confere com `final_test.json` (coeficientes, médias
internas e limiar idênticos, `ea_report.json`, `primary.refit_check`).

### Validação (698 opiniões, 32 audiências, 76 negativos)

Fonte: `artifacts/experiments/confidence_v2/ea_report.json`, `results.validation`.

| Sinal | ROC AUC [IC 95%] | AURC [IC 95%] | Precisão a 80% | Precisão a 90% | Δ AUC vs cosseno | Δ AURC vs cosseno |
|---|---|---|---|---|---|---|
| `cosine_serafim` | 0,730 [0,655; 0,797] | 0,054 [0,036; 0,074] | 0,927 [0,896; 0,955] | 0,919 [0,882; 0,946] | referência |  |
| `e3x_primary` | 0,876 [0,823; 0,915] | 0,023 [0,013; 0,036] | 0,957 [0,931; 0,977] | 0,940 [0,902; 0,964] | 0,146 [0,085; 0,216], p=0,0004, Holm 0,0120 | -0,032 [-0,045; -0,018], p=0,0004, Holm 0,0120 |
| `laya_multi_pt_p4` | 0,792 [0,722; 0,846] | 0,040 [0,025; 0,058] | 0,943 [0,915; 0,964] | 0,928 [0,895; 0,953] | 0,062 [-0,013; 0,137], p=0,1100, Holm 1,0000 | -0,014 [-0,027; 0,000], p=0,0564, Holm 0,9586 |
| `xnli_mdeberta` | 0,770 [0,691; 0,835] | 0,040 [0,024; 0,062] | 0,937 [0,897; 0,963] | 0,911 [0,874; 0,946] | 0,040 [-0,043; 0,123], p=0,3615, Holm 1,0000 | -0,014 [-0,029; 0,003], p=0,1068, Holm 1,0000 |
| `laya_multi_pt_mean_q7` | 0,781 [0,710; 0,834] | 0,043 [0,027; 0,061] | 0,939 [0,913; 0,960] | 0,920 [0,885; 0,947] | 0,051 [-0,024; 0,125], p=0,1768, Holm 1,0000 | -0,012 [-0,025; 0,002], p=0,1012, Holm 1,0000 |
| `laya_multi_pt_median_q7` | 0,791 [0,721; 0,843] | 0,039 [0,025; 0,057] | 0,943 [0,916; 0,963] | 0,927 [0,896; 0,950] | 0,061 [-0,015; 0,137], p=0,1208, Holm 1,0000 | -0,015 [-0,029; 0,001], p=0,0604, Holm 0,9586 |
| `laya_multi_pt_min_q7` | 0,769 [0,692; 0,826] | 0,053 [0,030; 0,081] | 0,937 [0,912; 0,959] | 0,924 [0,896; 0,950] | 0,038 [-0,044; 0,117], p=0,3591, Holm 1,0000 | -0,002 [-0,020; 0,020], p=0,8686, Holm 1,0000 |
| `laya_en_en_mean_q7` | 0,806 [0,731; 0,859] | 0,037 [0,020; 0,063] | 0,939 [0,903; 0,964] | 0,920 [0,885; 0,946] | 0,076 [-0,003; 0,153], p=0,0612, Holm 0,9586 | -0,017 [-0,034; 0,005], p=0,1080, Holm 1,0000 |
| `laya_en_en_median_q7` | 0,807 [0,731; 0,863] | 0,036 [0,021; 0,059] | 0,936 [0,901; 0,965] | 0,924 [0,891; 0,949] | 0,077 [-0,003; 0,153], p=0,0600, Holm 0,9586 | -0,018 [-0,033; 0,001], p=0,0596, Holm 0,9586 |
| `laya_en_en_min_q7` | 0,801 [0,724; 0,857] | 0,040 [0,021; 0,070] | 0,939 [0,899; 0,966] | 0,925 [0,892; 0,950] | 0,071 [-0,010; 0,153], p=0,0892, Holm 1,0000 | -0,015 [-0,034; 0,011], p=0,2300, Holm 1,0000 |
| `minicheck_ft5` | 0,834 [0,770; 0,887] | 0,032 [0,018; 0,051] | 0,946 [0,916; 0,971] | 0,933 [0,894; 0,956] | 0,104 [0,036; 0,175], p=0,0040, Holm 0,1000 | -0,022 [-0,034; -0,009], p=0,0004, Holm 0,0120 |
| `minicheck_deberta` | 0,820 [0,764; 0,868] | 0,034 [0,021; 0,050] | 0,954 [0,933; 0,970] | 0,928 [0,892; 0,958] | 0,090 [0,023; 0,161], p=0,0112, Holm 0,2464 | -0,020 [-0,034; -0,006], p=0,0048, Holm 0,1104 |
| `factcg_deberta` | 0,820 [0,771; 0,863] | 0,033 [0,020; 0,050] | 0,943 [0,912; 0,968] | 0,928 [0,886; 0,955] | 0,090 [0,018; 0,166], p=0,0176, Holm 0,3695 | -0,022 [-0,033; -0,008], p=0,0028, Holm 0,0728 |
| `alignscore_large` | 0,805 [0,735; 0,866] | 0,038 [0,022; 0,057] | 0,946 [0,923; 0,967] | 0,922 [0,887; 0,949] | 0,075 [0,006; 0,148], p=0,0324, Holm 0,5831 | -0,017 [-0,030; -0,002], p=0,0300, Holm 0,5699 |
| `hhem_open` | 0,815 [0,739; 0,877] | 0,037 [0,021; 0,062] | 0,945 [0,911; 0,967] | 0,924 [0,892; 0,949] | 0,085 [0,011; 0,155], p=0,0240, Holm 0,4799 | -0,017 [-0,033; 0,004], p=0,1056, Holm 1,0000 |
| `bge_reranker` | 0,842 [0,771; 0,900] | 0,033 [0,018; 0,054] | 0,954 [0,929; 0,972] | 0,938 [0,900; 0,960] | 0,112 [0,064; 0,159], p=0,0004, Holm 0,0120 | -0,021 [-0,033; -0,009], p=0,0044, Holm 0,1056 |
| `rank_mean` | 0,833 [0,772; 0,883] | 0,031 [0,020; 0,045] | 0,948 [0,922; 0,970] | 0,932 [0,897; 0,956] | 0,103 [0,072; 0,138], p=0,0004, Holm 0,0016 | -0,024 [-0,033; -0,014], p=0,0004, Holm 0,0016 |
| `rank_max` | 0,798 [0,726; 0,858] | 0,040 [0,025; 0,056] | 0,946 [0,916; 0,969] | 0,927 [0,895; 0,953] | 0,068 [0,049; 0,088], p=0,0004, Holm 0,0016 | -0,015 [-0,019; -0,009], p=0,0004, Holm 0,0016 |

Na validação, o verificador primário tem a maior ROC AUC (0,876) e o menor AURC (0,023) de todos os
sinais, e é o único que supera o cosseno nas duas métricas depois de Holm. Os avaliadores da literatura
ficam entre 0,805 e 0,842 de ROC AUC; todos têm Δ AUC positivo sem ajuste, mas depois de Holm só o
`bge_reranker` (AUC) e o `minicheck_ft5` (AURC) permanecem abaixo de 0,05. Nenhuma agregação do Laya
com as sete perguntas difere do cosseno depois de Holm. O menor p possível com 5.000 reamostragens é
0,0004, que multiplicado por 30 dá o 0,0120 que aparece nas linhas mais fortes. As duas combinações por
posto superam o cosseno, mas têm ROC AUC abaixo do primário sozinho (0,833 e 0,798 contra 0,876): somar
o cosseno ao primário piora a ordenação do primário neste conjunto. O experimento não declarou a
comparação direta do primário com cada avaliador da literatura, então a distância entre eles (0,034 de
ROC AUC para o `bge_reranker`, o mais próximo) fica só como estimativa pontual.

Premissa concatenada (secundária, fora de Holm, `results.validation.concatenated`): `minicheck_ft5`
0,869 de ROC AUC e 0,026 de AURC, `factcg_deberta` 0,855 e 0,027, `hhem_open` 0,848 e 0,030,
`bge_reranker` 0,830 e 0,038, `minicheck_deberta` 0,817 e 0,033. Os três modelos em inglês treinados
para documentos (MiniCheck-FT5, FactCG, HHEM) ganham de 0,03 a 0,04 de ROC AUC quando leem os 4 trechos
juntos; o Laya perde um pouco (0,766 a 0,799).

### Treino (2.981 opiniões, 144 audiências, 339 negativos; descritivo)

Fonte: `ea_report.json`, `results.train`. O primário e as duas combinações que o usam foram ajustados
neste conjunto e estão dentro da amostra; os p ajustados aparecem só como descrição, porque a família
de decisão é a de validação.

| Sinal | ROC AUC [IC 95%] | AURC [IC 95%] | Δ AUC vs cosseno | Δ AURC vs cosseno |
|---|---|---|---|---|
| `cosine_serafim` | 0,743 [0,710; 0,774] | 0,053 [0,043; 0,063] | referência |  |
| `e3x_primary` (dentro da amostra) | 0,875 [0,851; 0,898] | 0,026 [0,021; 0,033] | 0,132 [0,105; 0,160] | -0,026 [-0,034; -0,019] |
| `laya_multi_pt_p4` | 0,797 [0,767; 0,825] | 0,038 [0,031; 0,046] | 0,054 [0,021; 0,086] | -0,014 [-0,022; -0,006] |
| `xnli_mdeberta` | 0,762 [0,730; 0,794] | 0,047 [0,037; 0,058] | 0,019 [-0,023; 0,061] | -0,006 [-0,016; 0,005] |
| `laya_multi_pt_mean_q7` | 0,788 [0,758; 0,817] | 0,040 [0,032; 0,049] | 0,045 [0,011; 0,077] | -0,012 [-0,021; -0,004] |
| `laya_multi_pt_median_q7` | 0,792 [0,762; 0,821] | 0,039 [0,031; 0,047] | 0,049 [0,016; 0,082] | -0,014 [-0,022; -0,006] |
| `laya_multi_pt_min_q7` | 0,783 [0,751; 0,813] | 0,041 [0,033; 0,050] | 0,040 [0,006; 0,073] | -0,011 [-0,019; -0,003] |
| `laya_en_en_mean_q7` | 0,814 [0,788; 0,839] | 0,038 [0,030; 0,046] | 0,071 [0,040; 0,102] | -0,015 [-0,024; -0,006] |
| `laya_en_en_median_q7` | 0,818 [0,793; 0,842] | 0,035 [0,028; 0,042] | 0,075 [0,045; 0,106] | -0,018 [-0,026; -0,010] |
| `laya_en_en_min_q7` | 0,800 [0,770; 0,827] | 0,043 [0,034; 0,054] | 0,057 [0,024; 0,088] | -0,009 [-0,018; 0,000] |
| `minicheck_ft5` | 0,837 [0,811; 0,861] | 0,032 [0,025; 0,041] | 0,094 [0,067; 0,122] | -0,020 [-0,028; -0,013] |
| `minicheck_deberta` | 0,820 [0,791; 0,847] | 0,035 [0,027; 0,044] | 0,077 [0,045; 0,109] | -0,018 [-0,026; -0,010] |
| `factcg_deberta` | 0,813 [0,790; 0,837] | 0,034 [0,028; 0,041] | 0,070 [0,039; 0,102] | -0,018 [-0,026; -0,011] |
| `alignscore_large` | 0,825 [0,797; 0,852] | 0,036 [0,029; 0,045] | 0,082 [0,049; 0,114] | -0,016 [-0,025; -0,008] |
| `hhem_open` | 0,820 [0,791; 0,849] | 0,035 [0,028; 0,043] | 0,077 [0,046; 0,108] | -0,018 [-0,026; -0,010] |
| `bge_reranker` | 0,831 [0,803; 0,855] | 0,034 [0,028; 0,041] | 0,088 [0,064; 0,112] | -0,018 [-0,025; -0,013] |
| `rank_mean` (dentro da amostra) | 0,838 [0,811; 0,863] | 0,032 [0,026; 0,039] | 0,095 [0,081; 0,109] | -0,020 [-0,026; -0,015] |
| `rank_max` (dentro da amostra) | 0,811 [0,781; 0,839] | 0,040 [0,032; 0,048] | 0,068 [0,056; 0,079] | -0,013 [-0,016; -0,010] |

A ROC AUC do primário no treino (0,875) é praticamente igual à da validação (0,876), e a ROC AUC de
cada avaliador da literatura difere em no máximo 0,020 entre treino e validação.

### Perguntas novas e dispersão do Laya

Quatro perguntas novas de suporte (`n1_entails`, `n2_states`, `n3_attributable`, `n4_reader_agrees`,
declaradas em `[laya.new_questions]` antes de serem feitas) foram somadas às sete da bateria (q11). A
regra declarada só dá valor às perguntas novas se o intervalo de q11 menos q7 excluir zero do lado
favorável depois de Holm, na validação.

| Agregado (validação) | ROC AUC q11 | Δ AUC q11 - q7 | Δ AURC q11 - q7 |
|---|---|---|---|
| `laya_multi_pt_mean_q11` | 0,788 [0,716; 0,841] | 0,006 [0,003; 0,010], p=0,0016, Holm 0,0192 | -0,001 [-0,002; -0,000], p=0,0356, Holm 0,3915 |
| `laya_multi_pt_median_q11` | 0,794 [0,724; 0,846] | 0,003 [-0,007; 0,011], Holm 1,0000 | 0,000 [-0,002; 0,003], Holm 1,0000 |
| `laya_multi_pt_min_q11` | 0,769 [0,691; 0,827] | 0,000 [-0,003; 0,004], Holm 1,0000 | -0,000 [-0,001; 0,001], Holm 1,0000 |
| `laya_en_en_mean_q11` | 0,808 [0,731; 0,863] | 0,002 [-0,003; 0,006], Holm 1,0000 | -0,000 [-0,001; 0,001], Holm 1,0000 |
| `laya_en_en_median_q11` | 0,808 [0,734; 0,862] | 0,001 [-0,005; 0,007], Holm 1,0000 | -0,001 [-0,003; 0,001], Holm 1,0000 |
| `laya_en_en_min_q11` | 0,801 [0,723; 0,857] | -0,000 [-0,003; 0,003], Holm 1,0000 | 0,000 [-0,000; 0,000], Holm 1,0000 |

Só a média do `laya_multi_pt` passa a regra, e só em ROC AUC, com ganho de 0,006; nas outras cinco
agregações e em AURC as perguntas novas não mudam nada detectável. Sozinhas (máximo entre trechos), as
perguntas novas dão de 0,790 a 0,793 de ROC AUC no `laya_multi_pt` e de 0,805 a 0,808 no `laya_en_en`,
o mesmo nível da pergunta p4 e das agregações q7. As perguntas do Laya estão muito correlacionadas entre
si, e fazer mais perguntas do mesmo tipo quase não acrescenta informação.

Dispersão entre perguntas (desvio padrão das q7 no trecho de maior média; limiar = mediana do treino,
`spread_thresholds_train`):

| Dispersão | ROC AUC de -dispersão, validação | Negativos marcados / não marcados, validação | Diferença [IC 95%] |
|---|---|---|---|
| `laya_multi_pt_spread_q7` | 0,472 [0,413; 0,538] | 0,115 / 0,103 | 0,012 [-0,034; 0,057] |
| `laya_en_en_spread_q7` | 0,761 [0,688; 0,823] | 0,175 / 0,029 | 0,146 [0,087; 0,217] |

No `laya_en_en`, quando as perguntas discordam entre si o item tem seis vezes mais negativos (17,5%
contra 2,9% na validação, 18,5% contra 4,2% no treino). No `laya_multi_pt` a dispersão não carrega
informação. Com q11 os números são quase iguais (`results.validation.spread`).

## E-B: 121 UDVs anotadas por um modelo de linguagem

Os rótulos são a maioria de três passadas cegas de um modelo de linguagem sobre a pergunta
`trecho_sustenta` da amostra `human_validation_v1_udv_v1`; não são julgamento humano, e todo número
abaixo é concordância com essa anotação, não acurácia. Os rótulos ficam fora do repositório; o relatório
guarda contagens, hashes e métricas agregadas. As 121 UDVs vêm de 30 audiências, todas do split de
teste. Contagens: 62 `correta`, 56 `parcial`, 3 `incorreta`. Premissa: a frase única de evidência da
UDV. Positivo = `correta`; negativo = `parcial` ou `incorreta`. Este conjunto é confirmatório: nada foi
escolhido nele. O cosseno lido confere com `evidence.score` das 86 UDVs de camada semântica (diferença
máxima 1,2e-7); nas 35 UDVs `quote_found` ele foi calculado no mesmo par.

Fonte: `artifacts/experiments/confidence_v2/eb_report.json`.

| Sinal | ROC AUC [IC 95%] | AURC [IC 95%] | Precisão a 80% | Precisão a 90% | Δ AUC vs cosseno | Δ AURC vs cosseno |
|---|---|---|---|---|---|---|
| `cosine_serafim` | 0,768 [0,673; 0,858] | 0,273 [0,188; 0,369] | 0,557 [0,459; 0,663] | 0,550 [0,452; 0,639] | referência |  |
| `e3x_primary` | 0,844 [0,765; 0,911] | 0,224 [0,152; 0,317] | 0,598 [0,495; 0,698] | 0,560 [0,459; 0,652] | 0,076 [0,002; 0,148], p=0,0464 | -0,048 [-0,097; -0,004], p=0,0356 |
| `laya_multi_pt_p4` | 0,683 [0,567; 0,785] | 0,306 [0,221; 0,412] | 0,526 [0,430; 0,634] | 0,523 [0,426; 0,625] | -0,085 [-0,226; 0,053], Holm 1,0000 | 0,033 [-0,048; 0,113], Holm 1,0000 |
| `xnli_mdeberta` | 0,729 [0,649; 0,804] | 0,294 [0,211; 0,397] | 0,588 [0,489; 0,677] | 0,541 [0,446; 0,640] | -0,040 [-0,135; 0,058], Holm 1,0000 | 0,022 [-0,035; 0,082], Holm 1,0000 |
| `laya_multi_pt_mean_q7` | 0,734 [0,628; 0,817] | 0,281 [0,196; 0,386] | 0,557 [0,463; 0,667] | 0,541 [0,444; 0,644] | -0,035 [-0,158; 0,086], Holm 1,0000 | 0,008 [-0,067; 0,084], Holm 1,0000 |
| `laya_multi_pt_median_q7` | 0,690 [0,581; 0,787] | 0,303 [0,220; 0,408] | 0,546 [0,441; 0,644] | 0,523 [0,426; 0,622] | -0,078 [-0,215; 0,055], Holm 1,0000 | 0,031 [-0,048; 0,109], Holm 1,0000 |
| `laya_multi_pt_min_q7` | 0,694 [0,588; 0,787] | 0,299 [0,214; 0,405] | 0,546 [0,444; 0,641] | 0,541 [0,444; 0,635] | -0,075 [-0,210; 0,060], Holm 1,0000 | 0,026 [-0,053; 0,104], Holm 1,0000 |
| `laya_en_en_mean_q7` | 0,801 [0,696; 0,884] | 0,242 [0,159; 0,348] | 0,567 [0,462; 0,679] | 0,541 [0,434; 0,642] | 0,033 [-0,118; 0,166], Holm 1,0000 | -0,031 [-0,108; 0,049], Holm 1,0000 |
| `laya_en_en_median_q7` | 0,750 [0,654; 0,835] | 0,264 [0,183; 0,364] | 0,546 [0,440; 0,659] | 0,532 [0,426; 0,630] | -0,019 [-0,144; 0,098], Holm 1,0000 | -0,008 [-0,077; 0,059], Holm 1,0000 |
| `laya_en_en_min_q7` | 0,721 [0,629; 0,803] | 0,276 [0,196; 0,371] | 0,546 [0,446; 0,643] | 0,523 [0,433; 0,621] | -0,048 [-0,169; 0,070], Holm 1,0000 | 0,004 [-0,062; 0,069], Holm 1,0000 |
| `minicheck_ft5` | 0,703 [0,611; 0,789] | 0,310 [0,208; 0,434] | 0,577 [0,479; 0,685] | 0,541 [0,449; 0,647] | -0,065 [-0,186; 0,053], Holm 1,0000 | 0,038 [-0,034; 0,115], Holm 1,0000 |
| `minicheck_deberta` | 0,740 [0,659; 0,820] | 0,286 [0,202; 0,384] | 0,567 [0,474; 0,659] | 0,550 [0,450; 0,647] | -0,029 [-0,124; 0,071], Holm 1,0000 | 0,013 [-0,051; 0,079], Holm 1,0000 |
| `factcg_deberta` | 0,713 [0,620; 0,797] | 0,297 [0,204; 0,414] | 0,546 [0,441; 0,667] | 0,532 [0,433; 0,636] | -0,056 [-0,193; 0,080], Holm 1,0000 | 0,025 [-0,058; 0,110], Holm 1,0000 |
| `alignscore_large` | 0,744 [0,649; 0,837] | 0,277 [0,186; 0,382] | 0,567 [0,464; 0,678] | 0,550 [0,454; 0,646] | -0,024 [-0,136; 0,089], Holm 1,0000 | 0,004 [-0,064; 0,072], Holm 1,0000 |
| `hhem_open` | 0,803 [0,707; 0,893] | 0,242 [0,166; 0,334] | 0,598 [0,495; 0,682] | 0,550 [0,453; 0,645] | 0,034 [-0,067; 0,129], Holm 1,0000 | -0,030 [-0,084; 0,024], Holm 1,0000 |
| `bge_reranker` | 0,746 [0,666; 0,827] | 0,287 [0,193; 0,399] | 0,588 [0,479; 0,683] | 0,550 [0,452; 0,650] | -0,022 [-0,087; 0,054], Holm 1,0000 | 0,015 [-0,051; 0,070], Holm 1,0000 |
| `rank_mean` | 0,831 [0,748; 0,902] | 0,233 [0,159; 0,322] | 0,598 [0,494; 0,695] | 0,560 [0,458; 0,648] | 0,062 [0,023; 0,102], p=0,0028, Holm 0,0840 | -0,040 [-0,069; -0,013], p=0,0008, Holm 0,0256 |
| `rank_max` | 0,847 [0,769; 0,915] | 0,231 [0,152; 0,324] | 0,598 [0,495; 0,700] | 0,560 [0,458; 0,648] | 0,079 [0,024; 0,138], p=0,0028, Holm 0,0840 | -0,042 [-0,072; -0,014], p=0,0024, Holm 0,0744 |

Contraste principal (`main_contrast`): primário menos cosseno, ROC AUC +0,076 [0,002; 0,148], p de
bootstrap 0,046; AURC -0,048 [-0,097; -0,004], p 0,036. O teste de DeLong da diferença de AUC, que
trata as UDVs como independentes, dá z = 2,04 e p = 0,041. Se os dois p do contraste principal forem
ajustados juntos por Holm, os dois ficam em 0,071; a regra declarada usa o p não ajustado da AUC.

Os demais candidatos formam a família secundária (16 candidatos, 32 comparações). Nenhum avaliador da
literatura e nenhuma agregação do Laya difere do cosseno depois de Holm; todos os Δ têm intervalos que
cruzam zero, e cinco dos seis avaliadores da literatura têm AUC pontual abaixo do cosseno. O HHEM e a média q7
do `laya_en_en` (0,803 e 0,801) são os mais próximos do primário. Só `rank_mean` em AURC passa Holm
(0,026), e a combinação não melhora o primário sozinho (0,831 e 0,847 contra 0,844 de ROC AUC).

As perguntas novas não ajudam no E-B (Δ AUC de q11 menos q7 entre -0,046 e +0,001; `eb_report.json`,
`added_questions`). A dispersão do `laya_en_en` com o limiar do treino marca 108 das 121 UDVs, porque as
premissas de uma frase produzem mais discordância entre perguntas que os trechos do benchmark; entre as
marcadas há 52,8% de negativos, entre as 13 não marcadas 15,4% (diferença 0,374 [0,064; 0,612]).

Descritivo, `correta` ou `parcial` contra `incorreta`, com 3 negativos (sem intervalo nem teste):
ROC AUC 0,972 do cosseno, 0,949 do primário, 0,969 do `bge_reranker`, e de 0,477 a 0,881 nos
demais. Com três negativos esses números mudam muito ao trocar um único item.

## Resposta

No benchmark (E-A, validação, rótulo de especialista, premissa de 4 trechos), o verificador primário do
E3x é um sinal de confiança melhor que o cosseno do Serafim: +0,146 de ROC AUC e -0,032 de AURC, os dois
com intervalo longe de zero e p de Holm 0,012 numa família de 30 comparações. Ele também tem a maior
ROC AUC entre todos os sinais medidos, incluindo os seis avaliadores da literatura, que ficam entre
0,805 e 0,842.

Nas UDVs (E-B, rótulo de um modelo de linguagem, premissa de uma frase), pela regra declarada o
verificador é **melhor** que o cosseno: +0,076 de ROC AUC [0,002; 0,148], p = 0,046. A margem é
pequena: o limite inferior do intervalo está em 0,002 e o resultado deixa de passar em 0,05 se os dois p
do contraste principal forem ajustados juntos (0,071). Em resumo, o primário ordena melhor
que o cosseno nas duas bases, com evidência forte no benchmark e fraca, no limite do teste, nas 121
UDVs. Nenhum avaliador da literatura supera o cosseno no E-B, e nenhum supera o primário em qualquer das
duas bases.

## Ressalvas

- **Mudança de domínio.** No E-A a premissa são 4 trechos de transcrição de cerca de 630 caracteres
  cada; no E-B é uma frase. O primário e os avaliadores foram medidos nas duas condições, mas um
  resultado de uma não se transfere para a outra sem essa ressalva; a queda dos avaliadores da
  literatura de E-A para E-B (de 0,805 a 0,842 para 0,703 a 0,803) é compatível com isso.
- **O rótulo do E-B não é humano.** É a maioria de três passadas cegas de um modelo de linguagem. A
  concordância com essa anotação não é acurácia, e um viés comum do anotador e de algum sinal (por
  exemplo, ambos favorecerem frases com as mesmas palavras da proposição) infla a AUC desse sinal.
- **Tamanho do E-B.** 121 UDVs de 30 audiências, 59 negativos, quase todos `parcial`. O contraste
  principal passou por pouco; com essa amostra, diferenças de 0,05 de AUC não são distinguíveis.
- **A AUC do primário no E-B já era conhecida** antes desta declaração (0,8442, relatório do
  `udv_v1_verifier`); a do cosseno não era. A regra de decisão foi escrita antes de a do cosseno ser
  calculada.
- **Primário dentro da amostra no treino do E-A.** Os números de treino do primário e das combinações
  por posto são descritivos.
- **Tradução.** Os avaliadores em inglês e o `laya_en_en` leem a tradução NLLB; erros de tradução entram
  no escore. Nenhum avaliador da literatura foi treinado em português.
- **Precisão a 80% e 90%.** O corte é feito no próprio conjunto avaliado, então descreve a ordenação e
  não é um limiar implantável; no E-A a taxa de positivos é alta (89%) e essas precisões variam pouco
  entre sinais.
- **Truncamento.** O AlignScore truncou 13 dos 14.712 trechos (limite de 512 tokens); o `bge_reranker`
  truncou 72 das 3.678 premissas concatenadas (1024 tokens), que só entram na análise secundária.
- **Sem juiz LLM.** Nenhum LLM de instrução foi rodado como juiz, por decisão; a comparação com a
  prática mais comum da literatura fica restrita ao Laya com várias perguntas.

## Reprodução

```
uv run python -m utils.confidence_v2 smoke
uv run python -m utils.confidence_v2 laya-smoke
uv run python -m utils.confidence_v2 decide
uv run python -m utils.confidence_v2 score
uv run python -m utils.confidence_v2 laya-score
uv run python -m utils.confidence_v2 evaluate-ea
uv run python -m utils.confidence_v2 evaluate-eb --annotation <annotation.csv> --annotator "<descrição>"
```

Os comandos com modelo rodam em `mps`, um processo por vez, com
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4`. Os escores por par ficam em
`artifacts/cache/confidence_v2/` (ignorado no Git), e uma nova execução só calcula o que falta.
