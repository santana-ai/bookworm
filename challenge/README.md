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
  audiências:** 101 de 108 participantes (~93%) tiveram pelo menos um turno de fala localizado, usando
  uma comparação de nomes tolerante a nome do meio e acentuação faltando.
- **Ligar cada opinião a um trecho exato da fala funcionou mal (primeira tentativa).** Na amostra de 20
  audiências, das 268 opiniões processadas: 151 nem têm uma citação direta entre aspas (são paráfrase da
  matéria), 71 têm citação mas ela não foi localizada na fala da pessoa, e só 32 tiveram a citação
  efetivamente encontrada. Buscar evidência por correspondência exata de texto cobre uma fração pequena
  do problema.
- **Similaridade de texto (TF-IDF) cobre bem mais, com um corte de confiança calibrado a partir de
  dados, não escolhido a dedo.** Comparando a similaridade obtida em pares já confirmados por citação
  contra pares aleatórios, um corte de 0,25 separa bem as duas distribuições (mediana dos pares corretos
  em torno de 0,57; dos aleatórios, abaixo de 0,06). Com esse corte, das mesmas 268 opiniões: 32 por
  citação, 158 por similaridade acima do corte, 64 por similaridade abaixo do corte (uma tentativa de
  baixa confiança, não descartada), e 14 sem pessoa resolvida. Nenhuma opinião fica totalmente sem
  tentativa de evidência.
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
citação vs. pares aleatórios) deu média 0,732 contra 0,225 (31 pares cada), com corte escolhido em 0,46
(2/31 pares corretos abaixo do corte, 0/31 aleatórios acima). A cobertura resultante: 32 por citação,
217 por similaridade acima do corte, 5 abaixo do corte, 14 pessoa não resolvida, contra 158/64 do
TF-IDF. Comparando com a validação manual já feita, embeddings escolhe a mesma sentença que TF-IDF em
15/40 pares e uma sentença diferente em 25/40; desses 25, julgamento manual (`udv_manual_review.json`,
tier `embedding_diff`) classificou 9 corretas, 10 parciais e 6 incorretas: a escolha alternativa de
embeddings acerta ou confirma parcialmente a opinião em 19/25 (76%) e erra por completo em 6/25 (24%).

### Gerando e revisando a amostra de checagem manual

```bash
uv run python -m utils.generate_udv_manual_review
```

Isso escreve `udv_manual_review_template.json`: a mesma amostragem determinística de pares
opinião-evidência usada no notebook, mas com os campos `judgment` e `note` vazios. Esse script não
julga nada sozinho, ele só prepara o material; classificar cada par como `correta`, `parcial` ou
`incorreta` é trabalho manual de quem for revisar, editando o JSON diretamente. `udv.ipynb` lê o
arquivo já revisado (`udv_manual_review.json`), nunca o template.
