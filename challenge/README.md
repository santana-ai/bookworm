# challenge

Projeto `uv` onde rodam os experimentos, notebooks e análises em cima do dataset **PublicHearingBR**:
206 audiências públicas da Câmara dos Deputados, cada uma pareada com uma transcrição longa, uma
matéria jornalística e um resumo estruturado dos participantes e suas opiniões. O dataset é descrito no
artigo *PublicHearingBR: A Brazilian Portuguese Dataset of Public Hearing Transcripts for Summarization
of Long Documents* (Fernandes et al., 2024) e publicado em
[unicamp-dl/PublicHearingBR](https://huggingface.co/datasets/unicamp-dl/PublicHearingBR).

## Objetivo e hipótese

Uma audiência pública gera uma transcrição de dezenas de páginas. A imprensa publica uma matéria que
seleciona e resume parte do que foi dito. Hoje não existe nenhum vínculo rastreável entre uma opinião
publicada e a fala original que a sustenta: não dá para responder, de forma verificável, quem disse o
quê, onde na transcrição, e com que confiança essa ligação pode ser feita.

A hipótese deste projeto é que dá para construir essa ligação: uma representação que conecte cada
opinião publicada a um trecho verificável da transcrição de origem, com um nível de confiança explícito
quando a ligação for incerta ou não puder ser feita. Com esse vínculo, passa a ser possível medir, de
forma reprodutível, o quanto e como a cobertura editorial se afasta do que foi efetivamente dito
(omissão, seleção, reformulação), sem inferir a intenção de quem escreveu a matéria, só descrevendo o
que é observável.

É isso que motiva o desafio: usar esse dataset para uma ferramenta de auditoria documental, não mais um
sumarizador genérico. Os notebooks deste diretório documentam o processo de chegar lá, incluindo o que
não funcionou nas primeiras tentativas.

## Reproduzindo o download

```bash
uv run python -m utils.download_dataset
```

Isso baixa os arquivos para `dataset/` (nunca versionado no Git): `PublicHearingBR_LDS.jsonl`,
`PublicHearingBR_NLI.jsonl` e os READMEs originais do dataset.

## LDS e NLI: dois arquivos, duas fontes diferentes

O dataset vem em dois arquivos `.jsonl`, com 206 registros cada, alinhados 1:1 pelo campo `id`. É fácil
assumir que um é uma versão "enriquecida" do outro, mas não é bem assim: eles vêm de dois pipelines de
extração distintos, sobre fontes de texto diferentes.

### `PublicHearingBR_LDS.jsonl` (Long Document Summarization)

Cada registro tem `id`, `materia` (o texto da matéria jornalística), `transcricao` (a transcrição
completa da audiência) e `metadados` (`assunto` e `envolvidos`, cada um com `nome`, `cargo` e uma lista
de `opinioes`). Essas opiniões foram extraídas **da matéria**, com um LLM guiado por um prompt
específico para esse fim, e depois corrigidas manualmente (fusão de opiniões duplicadas, correção de
atribuição a pessoa errada). Este arquivo é tratado como o *ground truth* do dataset: o que a matéria
efetivamente diz que cada pessoa defendeu.

### `PublicHearingBR_NLI.jsonl` (Natural Language Inference)

Cada registro tem `id` e `metadados_extraidos` (`assunto`, `tl_dr` e `envolvidos`), mas **não** guarda o
texto da matéria nem da transcrição. Aqui as opiniões vêm de um experimento diferente: um baseline de
ChatGPT lendo a **transcrição diretamente** (com um prompt distinto do usado no LDS) para tentar extrair
sozinho as opiniões de cada participante. Cada opinião gerada nesse experimento vem acompanhada de até
quatro `chunks_proximos` (trechos da fala do participante, recuperados por similaridade) e de uma
`verificacao_alucinacao`: um rótulo manual (`verificacao_manual`, indicando se um especialista considerou
a opinião inferível a partir desses trechos) mais doze julgamentos automáticos equivalentes, feitos por
quatro modelos diferentes com três variações de prompt cada. Esse par opinião/trechos/rótulo é o que
permite usar o arquivo como um dataset de inferência textual (premissa: os trechos; hipótese: a opinião;
rótulo: se é inferível ou não).

### Por que isso importa

As opiniões do LDS e do NLI não coincidem entre si, mesmo para a mesma audiência: o campo `assunto`
nunca é idêntico nos dois arquivos para o mesmo `id`, porque cada um passou por uma extração
independente sobre uma fonte diferente (matéria vs. transcrição). Já os `chunks_proximos` do NLI são,
na prática, excertos reais retirados da `transcricao` do LDS (dá para confirmar isso comparando os dois
arquivos pelo `id`): é uma evidência textual pronta, associada a cada opinião, mesmo sem vir com offsets
explícitos.

Toda essa investigação, com os números que a sustentam, está em `eda.ipynb` (o notebook de análise
exploratória; o arquivo em disco pode ter um sufixo de versão temporário, tipo `eda_v01.ipynb`, com o
histórico completo guardado em `../backup/`).

## UDV: o problema que ela resolve

Uma opinião do LDS é só uma string dentro de uma lista. Não há nenhum jeito de perguntar, a partir dela,
quem disse aquilo, em que trecho da transcrição, ou com que grau de confiança. Isso importa porque a
matéria jornalística é uma seleção editorial do que foi dito na audiência, não um espelho completo dela:
para auditar essa seleção (o que foi incluído, omitido ou reformulado) é preciso conseguir voltar da
opinião publicada até a fala original que a sustenta.

A Unidade Deliberativa Verificável (UDV) é a tentativa de dar a cada opinião esse vínculo rastreável:
quem disse (o ator), o que foi dito (a proposição), e que evidência na transcrição sustenta isso, com um
nível de confiança explícito em vez de tratar toda opinião como igualmente verificada.

`udv.ipynb` (o arquivo em disco também pode ter um sufixo de versão temporário, com o histórico completo
em `../backup/`) é o protótipo dessa ideia, e os resultados são reportados como são, sem suavizar o que
não funcionou:

- **Segmentar a transcrição por participante exigiu duas tentativas.** A transcrição segue o formato
  oficial da Câmara (`O SR./A SRA. NOME(PARTIDO) - fala`), mas convidados sem filiação partidária
  (especialistas, testemunhas estrangeiras) têm um cabeçalho sem parênteses. Um regex que exige
  parênteses parece funcionar (encontra turnos em toda audiência), mas exclui esses participantes
  silenciosamente; corrigir isso foi necessário antes de qualquer coisa.
- **Resolver o nome de cada participante contra os turnos de fala funcionou bem numa amostra de 20
  audiências:** 102 de 108 participantes (~94%) tiveram pelo menos um turno de fala localizado, usando
  uma comparação de nomes tolerante a nome do meio e acentuação faltando.
- **Ligar cada opinião a um trecho exato da fala funcionou mal (primeira tentativa).** Na amostra de 20
  audiências, das 255 opiniões de participantes resolvidos (as outras 13 são de pessoas não
  resolvidas): 152 nem têm uma citação direta entre aspas (são paráfrase da matéria), 65 têm citação
  mas ela não foi localizada na fala da pessoa, e 38 tiveram a citação efetivamente encontrada (pela
  política de casamento descrita mais abaixo). Buscar evidência por correspondência de texto cobre uma
  fração pequena do problema.
- **Similaridade de texto (TF-IDF) cobre bem mais, com um corte de confiança escolhido por inspeção
  da distribuição de pares corretos e aleatórios.** Comparando a similaridade obtida em pares já confirmados por citação
  contra pares aleatórios (35 pares de cada lado), um corte de 0,25 separa bem as duas distribuições
  (mediana dos pares corretos em 0,53; dos aleatórios, abaixo de 0,05). Com esse corte, das mesmas 268
  opiniões: 38 por citação, 153 por similaridade acima do corte, 64 por similaridade abaixo do corte
  (uma tentativa de baixa confiança, não descartada), e 13 sem pessoa resolvida. Nenhuma opinião de
  pessoa resolvida fica sem tentativa de evidência.
- **A checagem manual confirmou que o corte separa bem confiável de não confiável.** Numa amostra de 20
  pares por nível de confiança, lidos e julgados um a um num arquivo revisável (não por um especialista
  humano independente, uma limitação do método, só um indício direcional forte): a evidência de alta
  confiança estava correta em 17 de 20 casos (85%) e parcialmente correta nos outros 3. A de baixa
  confiança acertou em só 3 de 20 (15%), com 8 parcialmente corretas e 9 incorretas. Ou seja, a camada de
  alta confiança se sustenta; a de baixa confiança não deveria ser tratada como evidência de verdade, só
  como pista fraca.

TF-IDF compara vocabulário compartilhado, não significado, o que explica a taxa de erro alta na camada
fraca: uma opinião que parafraseia a fala trocando as palavras por sinônimos, mantendo o mesmo sentido,
tende a ter similaridade baixa mesmo estando correta. O próximo passo é testar uma comparação por
significado (embeddings densos) na mesma amostra de validação manual.

### Escolha do modelo de embeddings

Um modelo multilíngue genérico não é a melhor opção disponível para português: o benchmark MTEB-PT
("Beyond Multilingual Averages: MTEB-PT, a Benchmark for Portuguese Sentence Encoders") mostra que
rankings multilíngues não predizem bem o desempenho específico em português, e no benchmark ASSIN2
(similaridade textual em português) o BERTimbau-Large chega a 0,852 de correlação de Pearson contra
0,809 de um BERT multilíngue genérico.

O artigo que descreve a família Serafim PT* ("Open Sentence Embeddings for Portuguese with the Serafim
PT* encoders family", PORTULAN) distingue as duas variantes do idioma e reporta que o Serafim 335,
baseado no BERTimbau (português brasileiro), lidera nos conjuntos de avaliação brasileiros, enquanto o
Serafim 900 (baseado no Albertina, português europeu) é melhor nos conjuntos europeus. Como as
transcrições do PublicHearingBR são em português do Brasil, o modelo usado é
`PORTULAN/serafim-335m-portuguese-pt-sentence-encoder`, carregado diretamente com
`sentence-transformers` (sem necessidade de mean pooling manual).

Na mesma amostra de 20 audiências (268 opiniões), a calibração por embeddings (pares corretos por
citação vs. pares aleatórios) deu média 0,731 contra 0,211 (35 pares cada), com corte escolhido em 0,47
(3/35 pares corretos abaixo do corte, 0/35 aleatórios acima). As versões anteriores do notebook davam
0,46 (31 pares) e 0,44 (33 pares) pelo mesmo método; o valor muda junto com o conjunto de pares
confirmados, que depende da política de casamento de citação, e com o re-sorteio dos pares aleatórios,
porque a semente é fixada uma vez antes das duas calibrações. Com cerca de 35 pares de cada lado, o
corte oscila na ordem de 0,02, uma limitação da calibração. A cobertura resultante: 38 por citação, 211
por similaridade acima do corte, 6 abaixo do corte, 13 pessoa não resolvida, contra 153/64 do TF-IDF. Comparando com a validação manual já feita, embeddings escolhe a mesma sentença que TF-IDF em
15/40 pares e uma sentença diferente em 25/40; desses 25, julgamento manual (`udv_manual_review.json`,
tier `embedding_diff`) classificou 9 corretas, 10 parciais e 6 incorretas: a escolha alternativa de
embeddings acerta ou confirma parcialmente a opinião em 19/25 (76%) e erra por completo em 6/25 (24%).

### Resolução de nomes e casamento de citação

A rodada completa do runner nas 206 audiências expôs quatro formas de cabeçalho de turno que o
resolvedor de nomes não tratava, três delas ausentes da amostra de 20 (a quarta, o nome social,
aparece nela uma vez, na audiência 12): a abreviatura com ponto dentro do
parêntese do presidente da sessão (`O SR. PRESIDENTE(Dr. Zacharias Calil. UNIÃO - GO)`, que a regra
antiga cortava em "Dr"); o nome social entre parênteses sem ponto
(`A SRA. PATRÍCIA RODRIGUES DA SILVA(PAGU RODRIGUES)`, quando `envolvidos` traz "Pagu Rodrigues"); o
marcador `(Manifestação em língua estrangeira. Tradução simultânea.)` lido como se fosse um nome; e
nomes de uma só palavra em `envolvidos` ("Gaguim") contra o nome completo no turno. O sintoma era o
`tier` `person_not_resolved`, que afirma que a pessoa não foi encontrada na transcrição, atribuído a
quem presidiu a sessão e falou entre 9 e 28 vezes.

Cada turno passa a ter uma lista de candidatos de nome: o nome do cabeçalho; o conteúdo do parêntese
antes do último ". " quando o que vem depois é filiação partidária (com "/" ou " - "); e o parêntese
inteiro quando ele não é filiação partidária (nome social, alias). A pessoa casa com qualquer candidato. Um nome de uma só
palavra só casa quando exatamente um orador da audiência tem essa palavra no nome, para não ligar um
sobrenome comum a várias pessoas. Nas 206 audiências isso resolve 17 participantes a mais (1.020 de
1.065) e amplia os turnos de 2 participantes já resolvidos (audiências 172 e 195, com 4 e 3 opiniões),
o que pode mudar a sentença de evidência sem mudar o nível, sem que nenhum turno fique atribuído a dois
participantes da mesma audiência; a
seção final de `udv.ipynb` mede o efeito de cada regra.

No casamento de citação, três mudanças: todas as citações entre aspas de uma opinião são tentadas, não
só a primeira; o prefixo precisa casar em fronteira de palavra ("Nós temos o" deixa de casar "Nós temos
os"), com a fronteira esquerda relaxada quando o trecho casado começa com maiúscula, porque a
transcrição às vezes cola o artigo à palavra seguinte ("noHousing First"); e a evidência gravada é a
sentença que contém o prefixo, ou o trecho de sentenças que ele atravessa, nunca o prefixo sozinho.
Essas correções explicam a diferença entre os contadores da amostra de 20 desta versão do notebook e os
das anteriores (v03: 101 de 108 participantes, 32 citações, corte de embeddings 0,46 com 31 pares; v04:
102 de 108, 34 citações, corte 0,44 com 33 pares).

Marcações de palco entre parênteses, como "(Manifestação em LIBRAS.)", deixam de contar como sentença.
Três participantes que só têm essas marcações como fala (8 opiniões, audiências 53 e 111) passam ao
`tier` `no_evidence`, em vez de receberem a marcação como evidência. A similaridade dessas 8 opiniões
com a marcação vai de 0,35 a 0,46: na rodada de 18/09, com corte 0,46, as 8 estavam em
`semantic_match_weak`; com o corte de 0,44 que valeu em seguida, 3 delas ficariam acima do corte. Nas
206 audiências, essas são as únicas partes descartadas por essa regra.

### Medindo o casamento de citação sem distinção de maiúsculas

```bash
uv run python -m utils.measure_case_insensitive_quotes --config configs/udv.toml --run-name udv_v0
```

O script compara, nas 206 audiências, o casamento por texto com dois relaxamentos: ignorar só a
primeira letra do prefixo e ignorar a caixa no prefixo inteiro. Ele não altera a pipeline nem regrava as
UDVs. Os números abaixo foram medidos contra a rodada de 19/09/2026, quando o casamento ainda era
sensível à caixa e aceitava qualquer tamanho de prefixo; o código dessa rodada está em `../backup/`
(`udv_pipeline_2026-09-19.py`, `build_udvs_2026-09-19.py`), e a política que saiu dessa medição está na
subseção seguinte.

Ignorando a caixa, 245 opiniões que naquela rodada estavam em `semantic_match_high` passam a ter citação
localizada (em 220 delas a diferença de caixa está só na primeira letra do prefixo; relaxar só a
primeira letra, sem mexer no resto, recuperaria 228 opiniões) e nenhum casamento exato se perde. Outros 27 registros
`quote_found` casam em outro ponto da fala, porque o prefixo relaxado permite descer menos na escada de
10, 6, 4 e 3 palavras; em 21 deles a sentença de evidência muda.

O tamanho do prefixo que casou separa dois regimes. Dos 245 casamentos novos, 117 vêm de prefixos de 6
palavras ou mais e 128 de prefixos de 3 ou 4 palavras, que são frases de ligação ("Eu acho que", "Do
ponto de vista") e casam em quase qualquer fala. O artefato daquela rodada tinha o mesmo problema: 128
dos 259 `quote_found` foram localizados por prefixos de menos de 6 palavras, e em só 46 deles a sentença
coincide com a que o encoder escolheria para a mesma opinião.

Por isso o script projeta três políticas de aceitação sobre as 2.203 UDVs, em
`artifacts/udv/case_insensitive_quotes_summary.json` (campo `projected_tiers`):

| política | `quote_found` |
| --- | --- |
| casamento exato (rodada de 19/09) | 259 |
| ignorar a caixa, aceitar todo casamento | 504 |
| ignorar a caixa, exigir prefixo de 6+ palavras | 260 |
| ignorar a caixa, exigir prefixo de 6+ palavras ou sentença igual à escolha do encoder | 360 |

A calibração do corte de embeddings também muda: aceitando todo casamento relaxado, os pares confirmados
por citação passam de 33 para 62, e o corte calculado pela mesma regra (ponto médio entre q25 dos pares
corretos e q75 dos aleatórios) fica em 0,45 nos três modos, inclusive no exato, cujo corte registrado
era 0,44. A diferença vem do sorteio dos pares negativos, o que dá a escala da incerteza desse corte (a
calibração da regra adotada, com 35 pares, está na subseção seguinte).

Qual política adotar depende também da precisão dos casamentos, que o script não mede: ele escreve
`artifacts/udv/case_insensitive_review_template.json`, uma amostra determinística de 40 casamentos novos
(semente 42) com os campos `judgment` e `note` vazios, para julgamento manual, e
`artifacts/udv/case_insensitive_quotes.jsonl` com todos os candidatos, cada um com a citação, o prefixo,
a sentença encontrada e a evidência da rodada comparada.

### Política de casamento de citação adotada

A regra de aceitação passou a ser: **casar o prefixo ignorando maiúsculas e minúsculas e aceitar como
citação só o casamento cujo prefixo tenha 6 palavras ou mais**. A escada de prefixos continua a mesma
(10, 6, 4 e 3 palavras), mas os degraus de 3 e 4 palavras deixam de produzir o nível `quote_found`,
porque são frases de ligação ("Eu acho que", "Do ponto de vista") que casam em quase qualquer fala. O
que sustenta a regra é o que a medição acima mostra: no artefato anterior, 128 dos 259 `quote_found`
vinham de prefixos curtos, e só 46 deles apontavam a mesma sentença que o encoder escolheria para a
mesma opinião.

Um prefixo curto que casa não é descartado: quando a sentença em que ele cai é a mesma que o encoder
escolheu (ou uma das duas contém a outra), a evidência continua sendo a sentença do encoder, o nível
continua saindo do score, e a evidência registra os dois sinais juntos, com `support_type`
`semantic_with_short_quote`, o `quote_prefix` e o `score` preenchidos. São 100 registros nas 206
audiências, todos em `semantic_match_high` (score mínimo 0,494; a distribuição por tamanho de prefixo
está no notebook): 48 deles eram `quote_found` por prefixo curto e 52 já eram similaridade, agora com a
citação registrada ao lado. O
corte de embeddings foi recalibrado com os pares que a regra nova confirma (35 pares, contra 33 do
casamento exato), e passou de 0,44 para 0,47.

Efeito medido nas 206 audiências, contra a rodada de 19/09/2026: 117 opiniões ganham `quote_found` (61
por prefixo de 6 palavras, 54 de 10, 2 de 7), 116 perdem (58 vinham de prefixo de 4 palavras, 45 de 3,
10 de 2, 2 de 1, 1 de 5) e 143 continuam `quote_found`. Dessas 143, 15 casam um prefixo mais longo ao
ignorar a caixa: 12 vinham de prefixo curto (3 para 10 palavras em 5 casos, 3 para 6 em 5, 4 para 6 em
2), que é o que a regra nova resgata, e 3 vão de 6 para 10 palavras; 9 delas mudam o texto da evidência.
O total fica em 260 `quote_found`, todos com prefixo de 6 palavras ou mais (145 de 6, 112 de 10, 3 de
7), contra 259 antes.

A comparação registro a registro depende da rodada anterior, que foi sobrescrita: para reproduzi-la,
gere de novo o `udv_v0` de 19/09 com o código de `../backup/` (`udv_pipeline_2026-09-19.py`,
`build_udvs_2026-09-19.py`) e o corte 0,44, e rode `utils.verify_udvs --baseline` contra ele. As
contagens de 143/116/117 não dependem disso: o notebook as recalcula cruzando o casamento sensível à
caixa com o atual.

Três limitações conhecidas da regra. A primeira citação da opinião que casa é a que decide, mesmo que
uma citação posterior casasse com prefixo mais longo: são 3 opiniões nas 206 audiências (o notebook
lista as três), duas terminando com corroboração de prefixo curto e uma só com similaridade, porque o
prefixo curto caiu em outra sentença. O prefixo curto que corrobora não precisa ser distintivo: 99 dos
100 têm 4 palavras ou menos e, embora 63 deles só apareçam na fala do próprio participante, 13 aparecem
em 10 ou mais das 1.020 falas resolvidas ("Eu acho que" aparece em 417), caso em que a coincidência
entre prefixo e sentença do encoder diz pouco. E `quote_found` afirma que o prefixo de 6 ou 10 palavras foi localizado, não que a citação
inteira está na transcrição: quando um prefixo de 6 palavras é aceito, o degrau de 10 já falhou, ou
seja, a citação diverge da fala em algum ponto depois da sexta palavra. Some-se a isso que a evidência
é sempre a primeira ocorrência do prefixo na fala: ignorar a caixa faz 13 prefixos gravados ocorrerem
mais de uma vez (eram 4 no casamento sensível à caixa) e, em 1 deles, outra ocorrência tem sobreposição
maior com a opinião: é um `direct_quote` (`udv-65-1-0`), ou seja, os offsets apontam a passagem errada. O critério de corroboração herda essa dependência da primeira ocorrência: testar
diretamente se o prefixo cai dentro da sentença gravada acrescentaria 7 registros, e essa troca fica em
aberto porque a projeção de 20/09 que sustenta a política foi calculada com o critério atual. A precisão desses casamentos
ainda depende do julgamento manual dos 40 pares de
`artifacts/udv/case_insensitive_review_template.json`, e essa amostra foi sorteada antes da regra: pelos
`support_type` de hoje, 21 dos 40 pares estão no estrato aceito como `quote_found`, 5 viraram
corroboração de prefixo curto e 14 são similaridade pura.

### Gerando e revisando a amostra de checagem manual

```bash
uv run python -m utils.generate_udv_manual_review
```

Isso escreve `udv_manual_review_template.json`: uma amostra determinística de 20 pares opinião-evidência
por nível de confiança (semente 42), com os campos `judgment` e `note` vazios. Esse script não
julga nada sozinho, ele só prepara o material; classificar cada par como `correta`, `parcial` ou
`incorreta` é trabalho manual de quem for revisar, editando o JSON diretamente. `udv.ipynb` lê o
arquivo já revisado (`udv_manual_review.json`), nunca o template.

A amostra em `udv_manual_review.json` não é reproduzida por esse script: ela foi sorteada em duas
etapas (10 + 10 pares por nível), antes das correções de resolução de nomes e de casamento de citação,
e o sorteio em uma etapa sobre os pools atuais escolhe pares diferentes. O notebook verifica o que a
política de citação nova faz com ela: para os 65 pares já julgados, a evidência de similaridade
escolhida é a mesma que foi julgada, então nenhum julgamento fica inválido, e 4 deles (3 de
`semantic_match_high`, 1 de `embedding_diff`, os quatro com julgamento `correta` no arquivo) passam a ter citação
localizada e deixam de ser produzidos pela camada de similaridade. O arquivo fica congelado como
conjunto de validação dessa amostra.

### Gerando as UDVs em lote

O notebook testa a ideia numa amostra e só conta a cobertura; ele não grava as UDVs em nenhum lugar.
Para ter as UDVs como artefato consultável (uma por opinião, com evidência, offsets e confiança),
existe um runner separado, configurado por arquivo:

```bash
uv run python -m utils.build_udvs --config configs/udv.toml --limit 20 --run-name dev20
uv run python -m utils.build_udvs --config configs/udv.toml
```

O primeiro comando processa só os 20 primeiros registros do LDS (a mesma amostra do notebook, útil como
regressão); o segundo processa as 206 audiências. `--ids 4 17 90` restringe a audiências específicas.

`configs/udv.toml` fixa tudo o que muda o resultado: o hash SHA-256 esperado do LDS (a execução aborta
se o arquivo for outro), o nome e a **revisão** (commit) do encoder no Hugging Face, o corte de
similaridade (0,47, com a origem da calibração registrada ao lado), a semente e os diretórios de saída.
O dispositivo é escolhido automaticamente (`mps` em Apple Silicon, `cuda`, senão `cpu`) e fica gravado no
relatório da rodada.

Saídas, em `artifacts/udv/`:

- `<run>.jsonl`: uma UDV por opinião do LDS. Campos: `id`, `hearing_id`, `actor` (`name`, `role`, vindos de
  `envolvidos`), `proposition` (o texto da opinião, intacto), `evidence` (`text`, `support_type`, `score`,
  `quote_prefix`, `start_char`, `end_char`, `speaker_turn`), `tier`, `provenance` e `method`.
- `<run>_coverage.json`: contagem por `tier`, contagem por `support_type` (`evidence_support_types`),
  pessoas resolvidas, offsets localizados, dispositivo, tempo por audiência, versões das bibliotecas e
  uma cópia da configuração usada.

`support_type` tem três valores: `direct_quote` (citação localizada por prefixo de 6+ palavras, `score`
nulo), `semantic_with_short_quote` (sentença escolhida pelo encoder que também contém o prefixo curto de
uma citação, com `score` e `quote_prefix` preenchidos) e `semantic_similarity` (só o encoder,
`quote_prefix` nulo).

Para conferir uma rodada contra o LDS (ids completos e únicos, proposição e ator intactos, `tier`
consistente com evidência, score, prefixo de citação, `support_type` e proveniência, offsets que
recuperam o texto e caem num turno do ator, contadores do relatório recomputados), com diff opcional
contra uma rodada anterior:

```bash
uv run python -m utils.verify_udvs --run-name udv_v0 --baseline caminho/para/udv_v0_anterior.jsonl
```

O comando imprime um relatório JSON e termina com erro se encontrar qualquer inconsistência.

Os `tier` são os mesmos do notebook: `quote_found`, `semantic_match_high`, `semantic_match_weak`,
`no_evidence` (pessoa resolvida, mas sem sentença com 4 palavras ou mais) e `person_not_resolved`.
`provenance` segue o vocabulário do projeto: `weak` quando a evidência veio de citação direta localizada
por casamento de texto, `model` quando veio de similaridade de embeddings (inclusive quando um prefixo
curto de citação corrobora a sentença, porque quem escolheu a sentença foi o encoder), `null` quando não
há evidência. Nenhuma UDV gerada aqui é anotação humana.

`start_char`/`end_char` são offsets absolutos em `transcricao`, e `speaker_turn` é o índice do turno de
fala onde a sentença foi encontrada. A localização é tolerante a diferenças de espaçamento, mas falha
quando a sentença atravessa a fronteira entre dois turnos da mesma pessoa; nesses casos os três campos
ficam `null` e o relatório de cobertura conta quantos foram.

Os embeddings ficam em cache em `artifacts/cache/embeddings/` (não versionado), indexados por revisão do
encoder, dispositivo e conteúdo dos textos, então rodar de novo com a mesma configuração não recomputa
nada. Uma limitação herdada do encoder: `max_seq_length` é 128 tokens, sentenças mais longas são truncadas
antes de virar vetor.

## Splits com prevenção de vazamento

Escolher qualquer hiperparâmetro olhando o conjunto de teste invalida a avaliação, então os recortes
precisam existir antes do primeiro modelo. Sortear opiniões ao acaso vazaria de duas formas: as
opiniões de uma mesma audiência compartilham transcrição, participantes e assunto, e apareceriam dos
dois lados; e pautas duram semanas no Congresso, então um sorteio deixaria o modelo aprender o
vocabulário de um debate ainda em curso no período de teste. A unidade de divisão é a audiência, e o
primeiro recorte é o temporal: treino no passado, teste no futuro.

### A data, que o dataset não tem

O schema real do LDS é `id`, `materia`, `transcricao` e `metadados`: não há campo de data, comissão nem
URL. O que existe é o carimbo de publicação da matéria da Agência Câmara, em texto corrido logo abaixo
da linha fina, no formato `DD/MM/AAAA - HH:MM`, capturado por

```
(\d{2})/(\d{2})/(\d{4})\s*-\s*(\d{2}):(\d{2})
```

Treze matérias foram atualizadas depois de publicadas e trazem um segundo carimbo precedido de
`Atualizado em`; a extração descarta qualquer carimbo que caia dentro desse trecho e devolve o primeiro
que sobra. As 206 matérias têm carimbo.

Esse carimbo é a hora da publicação, não um campo da audiência, então usá-lo para ordenar só se
sustenta se a publicação acompanhar o evento. A checagem usa o próprio texto: as matérias nomeiam o dia
do debate na forma `nesta quarta-feira (17)`. O dia do mês citado é reconstruído como a data mais
recente até o carimbo que tenha aquele dia, e o dia da semana citado é comparado com o dessa data.
Resultado nas 206: 156 matérias nomeiam o dia, 155 confirmam a data extraída, e a defasagem entre
publicação e evento é de zero dia em 142 menções e de um dia em 15, nunca maior. A única audiência sem
confirmação é a 111, cuja matéria cita `nesta terça-feira (23)` sendo que 23/03/2024 caiu num sábado e
terça-feira é o dia do próprio carimbo, um erro da matéria. A data extraída é, portanto, a data da
audiência com erro de no máximo um dia, sempre para frente.

### Como os cortes são escolhidos

São 128 datas distintas entre 18/11/2021 e 09/05/2024, com até 5 audiências no mesmo dia. O corte cai
entre datas, nunca dentro de uma, e só são considerados cortes cujo intervalo até a próxima data com
audiência seja de ao menos `min_boundary_gap_days` (2), para que o erro de um dia da extração não possa
trocar uma audiência de lado. Entre os 65 cortes elegíveis, escolhe-se o que deixa a fração acumulada
mais perto do alvo (0,70 para o fim do treino, 0,85 para o fim da validação), desempatando pelo maior
intervalo. Os dois cortes escolhidos caem em recessos do calendário: 13/11/2023, com 8 dias até a
próxima audiência, e 20/12/2023, com 76 dias, o recesso parlamentar.

| split | audiências | fração | período | UDVs |
| --- | ---: | ---: | --- | ---: |
| `train` | 144 | 69,9% | 18/11/2021 a 13/11/2023 | 1.536 |
| `validation` | 32 | 15,5% | 21/11/2023 a 20/12/2023 | 308 |
| `test` | 30 | 14,6% | 05/03/2024 a 09/05/2024 | 359 |

Os níveis de confiança das UDVs não se distribuem igualmente: as opiniões de pessoa não resolvida se
concentram no treino (80 das 90, ou 5,2% do treino contra 1,0% da validação e 1,9% do teste), porque os
formatos de cabeçalho que o resolvedor erra são os das audiências mais antigas, e 6 das 8 opiniões sem
evidência caem no teste. Parte de qualquer diferença de métrica entre conjuntos virá dessa composição.

### O que o split temporal não isola

Separar por tempo não separa atores: 55 dos 879 nomes distintos aparecem em mais de um conjunto e 24
aparecem em treino e teste, o caso típico sendo um ministro que depõe várias vezes no período. Isso é
esperado e não é corrigível sem abrir mão da cronologia; medir generalização para atores inéditos exige
um split por ator, que é outro recorte. Quanto a near-duplicates, a maior similaridade TF-IDF entre
matérias de conjuntos diferentes é 0,459 e nenhum par cruzando fronteira passa do corte de 0,5
registrado em `configs/splits.toml`, então não há audiência quase duplicada separada pelo split. O par
mais próximo (68 no treino, 105 no teste) trata do mesmo tema clínico em momentos diferentes, que é o
que um split temporal deve permitir.

### Gerando e verificando

```bash
uv run python -m utils.build_splits
uv run python -m utils.verify_splits
```

Saídas, em `artifacts/splits/`:

- `temporal_v1.json`: o manifesto, com `split_version`, `grouping_method`, `seed`, as três listas de
  `id`, as fronteiras e a data extraída de cada audiência, para que a atribuição possa ser refeita sem
  executar nada.
- `temporal_v1_report.json`: a validação da extração de data, a distribuição por ano e por data, os
  cortes considerados, os contadores por split (incluindo UDVs por `tier`) e as duas medidas de
  vazamento acima.

O verificador refaz a extração de data, a escolha dos cortes e a atribuição a partir do LDS, e confere
a partição (cada audiência exatamente uma vez), a cronologia, que nenhuma data caia em dois conjuntos,
que o intervalo de cada fronteira seja maior que o erro da data extraída, e recomputa todos os
contadores do relatório. Termina com erro se encontrar qualquer inconsistência.

O split temporal é determinístico e não consome a semente; ela é gravada no manifesto porque os splits
por comissão e por similaridade, ainda não construídos, vão precisar dela. A comissão continua sem
extração validada: o nome aparece em texto corrido, em 188 matérias e em 202 transcrições (nas 206 em
ao menos uma das duas), mas extrair dali um rótulo canônico, com as audiências conjuntas de duas ou
mais comissões tratadas corretamente, ainda não foi feito nem medido.

## Lint

```bash
uv run ruff check utils/
uv run ruff format --check utils/
```
