# Roadmap

Este documento é vivo: é atualizado a cada milestone concluído ou decisão de rota. Diferente de
`CONSTITUTION.md` (a tese e a especificação, que não mudam com o progresso do dia a dia) e de
`CLAUDE.md` (as regras operacionais de como trabalhar), o `ROADMAP.md` é o único lugar que responde
"onde estamos" e "para onde vamos agora".

## Norte

Ver [`CONSTITUTION.md`](CONSTITUTION.md) para a tese científica completa. Em uma frase: aprender uma
representação verificável da deliberação pública (quem disse o quê, com que evidência, com que
confiança) que sirva para comparar, buscar, clusterizar, avaliar resumos e auditar cobertura editorial,
sem nunca substituir a transcrição original.

## Hipótese

É possível ligar cada opinião publicada numa matéria a um trecho verificável da transcrição de origem,
com um nível de confiança explícito, e usar esse vínculo para medir (nunca acusar) o desvio entre a
cobertura editorial e o que foi efetivamente dito numa audiência pública.

## Prazos-chave

Inscrição até 30/07/2026, submissão até 30/09/2026 (ver `CLAUDE.md` §1 para os fatos completos do
desafio).

## Checklist de entregáveis do desafio (tracking real, não só código)

- [ ] Inscrição da equipe (prazo 30/07/2026 — confirmar se já foi feita).
- [ ] Artigo científico ≤10 páginas no template do desafio: problema, metodologia, arquitetura, resultados.
- [ ] Repositório de código documentado (`bookworm/`) com README cobrindo motivação, instalação, dados,
      execução, arquitetura, ética, citação.
- [ ] Vídeo ≤5 minutos, roteirizado a partir do "experimento decisivo" (`CONSTITUTION.md`).
- [ ] Dashboard/demo interativa em Streamlit (P1, recomendado).
- [ ] Envio para rodrigo.barros@kunumi.com e gianlucca@kunumi.com até 30/09/2026.

## Escopo priorizado (P0 / P1 / P2)

O `CONSTITUTION.md` descreve um programa de pesquisa amplo, pensado para uma equipe full-time ao longo
de meses ou anos, não para uma equipe pequena em poucas semanas entregando um artigo curto. Sem
prioridade explícita, o risco é não terminar nada com profundidade suficiente para nenhum critério de
avaliação do desafio. Estas tags orientam o que fazer a seguir:

- **P0, necessário para o artigo/vídeo de 30/09:** higiene de projeto (lint/tipos/testes); ingestão e
  perfil do dataset (`inspect-schema`, validação, parquet), sabendo que o schema real (validado na EDA)
  não tem `date`/`committee`/`source_url` estruturados nem `hearing_id` string (é `id` inteiro, ver
  `CONSTITUTION.md` §25.4); splits com prevenção de vazamento (ao menos temporal e por comissão),
  dependendo de extrair e validar data/comissão do texto livre da `materia`; baseline TF-IDF + SVD; UDV
  v0/v1 (fonte: opiniões estruturadas + evidência recuperada, sem inventar campos); EDI v1 (omissão,
  super-representação, distorção de posição); comparação explicável entre audiências; busca simples
  (lexical ou híbrida); relatório de avaliação estatística com intervalo/teste, não só ponto estimado; o
  "experimento decisivo" (audiência inédita, UDVs, assinatura latente, comparação, 2 resumos avaliados)
  como roteiro do vídeo de 5 minutos; a biblioteca `bookworm` importável (`from bookworm import ...`) e
  sua CLI (`uv run bookworm <comando>`), que é como o baseline é exercitado.
- **P1, valioso se sobrar tempo:** NMF, matriz estruturada com pesos explícitos, clustering com
  avaliação de estabilidade, testes adversariais determinísticos (negação, troca de ator, omissão);
  `bookworm init` (scaffold de projeto dependente); API HTTP do `bookworm`; dashboard Streamlit
  consumindo a API/lib, construído depois do núcleo científico funcionar, atacando o critério de
  Impacto/Aplicabilidade e o "dashboard/demo recomendado" do desafio.
- **P2, visão futura, fora do escopo desta submissão:** autoencoder, VAE, contrastive learning, graph
  encoder, fatoração tensorial (CP/Tucker), avaliação causal completa, agenda de 4 papers. Não iniciar
  nenhum item de P2 antes de P0 estar completo e documentado.

Ao decidir o que fazer a seguir, pergunte primeiro se o item é P0. Itens P1/P2 só entram na sprint atual
se todo P0 relevante já tiver: implementação, testes, tipagem, config, métrica e limitação registrada.

## Notebooks de conceito planejados (nenhum escrito ainda)

`challenge/concepts/` vai conter um notebook Jupyter por conceito técnico usado no projeto, explicando a
intuição em células markdown (pt-BR) e demonstrando o funcionamento com um exemplo sintético pequeno.
Lista planejada, cobrindo tudo que aparece em `CONSTITUTION.md`:

`tfidf`, `svd`, `nmf`, `sparse_pca`, `autoencoder`, `vae`, `contrastive_learning`, `graph_encoder`,
`tensor_factorization`, `embeddings`, `nli`, `clustering`, `umap`.

## Onde estamos agora

- Fundação escrita: `CLAUDE.md`, `CONSTITUTION.md`, dataset baixado e reproduzível
  (`challenge/utils/download_dataset.py`), `challenge/README.md`.
- `eda.ipynb` (v01): validou que os números agregados do dataset batem com o artigo original, mapeou o
  schema real dos dois arquivos (LDS e NLI), confirmou que a transcrição tem turnos de fala detectáveis
  por regex, e que LDS e NLI vêm de pipelines de extração diferentes.
- `udv.ipynb` (v00 a v05), protótipo de UDV, numa amostra de 20 audiências (268 opiniões); os números
  abaixo são da execução do v05, depois da correção do resolvedor e da política de casamento de citação
  descritas mais adiante:
  - resolução de pessoa: **102/108** participantes (~94%) tiveram pelo menos um turno de fala
    localizado, depois de corrigir o regex de turno para aceitar cabeçalhos sem parênteses (convidados
    sem filiação partidária ficavam de fora);
  - v0, evidência só por citação direta: **152 opiniões sem citação** (paráfrase), **65 com citação não
    localizada**, **38 com citação efetivamente encontrada** na fala da própria pessoa (217 das 255
    opiniões de pessoas resolvidas sem nenhuma evidência; 230/268 contando as 13 de pessoa não
    resolvida);
  - v1, evidência por citação direta + similaridade TF-IDF (corte de 0,25 escolhido por inspeção da
    distribuição de pares corretos vs. aleatórios): **38 por citação**, **153
    por similaridade acima do corte**, **64 por similaridade abaixo do corte** (melhor tentativa, baixa
    confiança), **13 pessoa não resolvida**. 191/268 terminam com evidência de confiança razoável;
  - v2, checagem manual de uma amostra (20 pares por nível, classificados manualmente num arquivo
    revisável, `challenge/udv_manual_review.json`, não por um especialista humano independente, uma
    limitação real do método): `semantic_match_high` teve **17/20 corretas, 3 parciais, 0 incorretas**
    (85%); `semantic_match_weak` teve **3/20 corretas, 8 parciais, 9 incorretas** (15%). O corte de 0,25
    está separando bem evidência confiável de evidência não confiável;
  - conclusão: a camada de alta confiança se sustenta; a de baixa confiança não deve ser tratada como
    evidência de verdade, só como pista fraca. TF-IDF compara vocabulário compartilhado, não
    significado, o que explica a taxa de erro alta na camada fraca. Isso motiva testar embeddings
    densos (comparação por significado) na mesma amostra de validação manual.
  - v3, comparação por significado com embeddings densos
    (`PORTULAN/serafim-335m-portuguese-pt-sentence-encoder`, a variante do Serafim PT* treinada para
    português brasileiro): calibração com pares corretos por citação vs. pares aleatórios deu média
    0,731 vs. 0,211 (35 pares cada), corte escolhido em 0,47 (3/35 pares corretos abaixo do corte, 0/35
    aleatórios acima; o v03, com 31 pares, dava 0,46 e o v04, com 33, dava 0,44 pelo mesmo método: o
    conjunto de pares confirmados depende da política de casamento de citação e os aleatórios são
    re-sorteados a cada versão, então com cerca de 35 pares de cada lado o corte oscila na ordem de
    0,02, uma limitação da calibração). Cobertura nas mesmas 268
    opiniões: 38 por citação, 211 por similaridade acima do corte, 6 abaixo do corte, 13 pessoa não
    resolvida (contra 153/64 do TF-IDF). Comparado à validação
    manual já feita, embeddings escolhe a mesma sentença que TF-IDF em 15/40 pares e uma sentença
    diferente em 25/40; desses 25, julgamento manual (mesmo arquivo
    `challenge/udv_manual_review.json`, tier `embedding_diff`) classificou 9 corretas, 10 parciais e 6
    incorretas. A cobertura de alta confiança cresce de 153 para 211 opiniões e a camada de baixa
    confiança cai de 64 para 6; nos 25 casos em que os dois métodos discordam, a escolha de embeddings
    foi julgada correta ou parcial em 19 (76%) e incorreta em 6 (24%).
- Runner de UDV em lote (`challenge/utils/build_udvs.py`, configurado por `challenge/configs/udv.toml`):
  grava uma UDV por opinião em JSONL, com evidência, offsets absolutos na transcrição
  (`start_char`/`end_char`/`speaker_turn`), `tier` e proveniência (`weak` para citação direta, `model`
  para similaridade), mais um relatório de cobertura por rodada. O encoder fica fixado numa revisão
  (commit `a018870` do Serafim 335m), o corte de similaridade sai do notebook para o arquivo de
  configuração, que também fixa o hash SHA-256 do LDS. Na amostra de 20 audiências o runner reproduz
  exatamente os contadores do notebook (hoje 38/211/6/13 com o v05 e o corte de 0,47; eram 34/217/4/13
  com o v04 e o corte de 0,44) e localiza offsets em 254 das 255 evidências (a que falta atravessa a
  fronteira entre dois turnos). Custo de uma rodada sem cache: 176 s em Apple M4 via MPS (média 8,8 s por audiência, máximo
  45 s, medidos na primeira rodada, 18/09/2026); com os embeddings em cache a mesma amostra roda em
  menos de 1 s.
- Primeira rodada completa do runner nas 206 audiências (18/09/2026, antes da correção do resolvedor):
  2.203 UDVs, 1.003/1.065 pessoas resolvidas, 249 `quote_found`, 1.790 `semantic_match_high`, 51
  `semantic_match_weak`, 113 `person_not_resolved` (62 pessoas), offsets em 2.081 de 2.090 evidências,
  1.419 s em MPS. Uma revisão do resolvedor contra as 206 audiências encontrou quatro formas de
  cabeçalho quase ausentes da amostra de 20 (só o nome social aparece nela, uma vez) e que deixavam 17 dessas 62 pessoas sem turno embora
  tivessem falado (7 presidentes de sessão com abreviatura no parêntese, 7 nomes sociais entre
  parênteses, 1 marcador de tradução simultânea lido como nome, 2 nomes de um só token), mais três
  problemas no casamento de citação (só a primeira citação era tentada; prefixo casando dentro de
  palavra maior; texto da evidência igual ao prefixo quando ele atravessava fronteira de sentença). O código da
  pipeline usado nessa rodada e os relatórios de cobertura dela estão em `backup/`
  (`udv_pipeline_2026-09-18.py`, `build_udvs_2026-09-18.py`, `udv_v0_coverage_2026-09-18.json`,
  `dev20_coverage_2026-09-18.json`), para que esses números continuem reproduzíveis.
- Correção do resolvedor de nomes e do casamento de citação (`challenge/utils/udv_pipeline.py`,
  `udv.ipynb` v04, 19/09/2026): cada turno passa a ter candidatos de nome (cabeçalho; conteúdo do
  parêntese antes do último ". " quando o que segue é filiação partidária; parêntese inteiro quando
  não é filiação partidária) e nomes de um só
  token casam só quando há exatamente um orador com aquele token na audiência. Efeito medido nas 206
  audiências (seção final do notebook): 1.020/1.065 pessoas resolvidas (17 a mais, 23 opiniões), 1.003 alcançadas pelas regras antigas
  (mesmo total da rodada de 18/09), 2 pessoas já resolvidas ganham turnos (audiências 172 e 195),
  mudando a evidência de 4 opiniões sem mudar o nível, nenhum turno atribuído a dois participantes da
  mesma audiência; 45 pessoas seguem sem turno (instituições, grafias diferentes, quem não falou). No
  casamento de citação: 11 opiniões ganham citação localizada (8 pela segunda citação, 3 de pessoas
  recém-resolvidas), 1 falso positivo cai ("Nós temos o" casando "Nós temos os"), 3 prefixos encurtam,
  e 2 citações coladas ao artigo por erro de transcrição ("noHousing First") são mantidas por relaxar a
  fronteira esquerda quando a citação começa com maiúscula. Dos 8 registros em que o texto da evidência
  era igual ao prefixo, só 2 estavam truncados (prefixo atravessando fronteira de sentença) e passam a
  gravar o trecho completo; os outros 6 já eram a sentença inteira. Marcações de palco ("(Manifestação em LIBRAS.)") deixam de
  contar como sentença: 3 participantes que só têm essas marcações como fala (8 opiniões) passam a
  `no_evidence` em vez de receber a marcação como evidência (3 dessas opiniões estavam em alta
  confiança). O casamento de citação segue sensível a maiúsculas; o notebook mede quantas opiniões
  ganhariam citação ignorando a caixa, e isso fica como limitação conhecida até validação manual da
  precisão. A calibração de embeddings, refeita com 33 pares confirmados em vez de 31 (3 entram, 1 sai)
  e pares aleatórios re-sorteados, move o corte de 0,46 para 0,44 (`configs/udv.toml` atualizado).
  Para os 65 pares de `udv_manual_review.json`, o notebook recomputa a evidência: a pipeline corrigida
  escolhe a mesma em todos e nenhum ganhou citação, então os julgamentos continuam válidos e a amostra
  fica congelada (o gerador de template não reproduz esse sorteio, feito em duas etapas de 10 pares).
- Rodada completa refeita com a correção (`udv_v0`, 19/09/2026; os caminhos
  `challenge/artifacts/udv/udv_v0.jsonl` e `udv_v0_coverage.json` guardam hoje a rodada seguinte, e os
  contadores desta ficaram em `backup/udv_v0_coverage_2026-09-19.json`): 2.203 UDVs; 1.020/1.065 pessoas (95,8%); 259 `quote_found`, 1.811
  `semantic_match_high`, 35 `semantic_match_weak`, 8 `no_evidence`, 90 `person_not_resolved` (45
  pessoas); offsets localizados em 2.104 de 2.105 evidências (a que falta atravessa a fronteira entre
  dois turnos); menos de 2 min em MPS com embeddings em cache. Em
  relação à rodada de 18/09: 23 opiniões saem de `person_not_resolved` (18 para alta confiança, 3 para
  citação, 2 para baixa), 8 vão de similaridade para citação, 1 de citação para similaridade, 10 vão de
  baixa para alta confiança só pela mudança do corte (scores entre 0,44 e 0,46), 8 vão de baixa para
  `no_evidence` (marcações de palco), 2 citações têm o texto estendido, 3 prefixos encurtam e 4
  evidências de similaridade trocam de sentença por turnos novos de pessoas já resolvidas: 49
  registros com evidência diferente. Verificação com `challenge/utils/verify_udvs.py` (recontagem a
  partir do LDS com as mesmas funções, texto recuperado pelos offsets, turno da evidência pertencente
  ao ator, consistência entre `tier`, score, prefixo de citação e proveniência, contadores do
  relatório): zero inconsistências, em `dev20` e em `udv_v0`.
- Experimento do casamento de citação sem distinção de maiúsculas (20/09/2026,
  `challenge/utils/measure_case_insensitive_quotes.py`, saídas em `challenge/artifacts/udv/`:
  `case_insensitive_quotes.jsonl`, `case_insensitive_existing_quotes.jsonl`,
  `case_insensitive_quotes_summary.json`, `case_insensitive_review_template.json`). Medidas nas 206
  audiências, sem alterar a pipeline e contra a rodada de 19/09: ignorando a caixa, 245 opiniões que
  estavam em `semantic_match_high` ganham citação localizada (220 diferindo da transcrição só na
  primeira letra; relaxar só a primeira letra recuperaria 228), nenhum casamento exato se perde e 27
  registros `quote_found` passam a casar em outro ponto da fala (21 com sentença diferente). O efeito
  se separa pelo tamanho do prefixo que casou: 117 dos casamentos novos vêm de prefixos de 6+ palavras
  e 128 de prefixos de 3 ou 4 palavras. O artefato daquela rodada tinha a mesma exposição: 128 dos 259
  `quote_found` foram localizados por prefixos de menos de 6 palavras e só 46 desses coincidiam com a
  sentença que o encoder escolheria. Projeções de `quote_found` sobre as 2.203 UDVs: 259 naquela
  rodada, 504 aceitando todo casamento relaxado, 260 exigindo prefixo de 6+ palavras e 360 exigindo
  prefixo de 6+ palavras ou sentença igual à escolha do encoder. Recalibrando o corte de embeddings com
  os pares confirmados por citação desse modo (62 em vez de 33), a mesma regra dá 0,45, valor que o
  modo exato também dá com os negativos re-sorteados, o que mede a incerteza do 0,44 então registrado.
- Julgamento dos casamentos do experimento acima (`claude_case_insensitive_quote_review.py`, na raiz):
  410 pares (245 novos, 21 pares exato/relaxado dos que mudam de lugar, 103 `quote_found` atuais de
  prefixo curto, 20 de controle com prefixo longo), rotulados por dois leitores independentes com
  adjudicação das 8 divergências. São julgamentos propostos por agente, não a validação humana exigida
  aqui: o template para essa validação é `case_insensitive_review_template.json` (40 pares, semente 42).
  Resultado: casamentos novos acertam 223 de 245 (91%), com 54/54 nos prefixos de 10 palavras, 60/61
  nos de 6, 70/78 nos de 4 e 37/50 nos de 3; o controle de prefixo longo acerta 20/20; os
  `quote_found` atuais de prefixo curto acertam 89 de 103, e dos 21 que mudam de lugar o texto atual
  está certo em 4 e o relaxado em 17. A estimativa estratificada de precisão do `quote_found` atual é
  0,875, ou cerca de 32 registros errados em 259, todos no estrato de prefixo curto. Exigir prefixo de
  6+ palavras leva a precisão dos julgados a 0,993 (1 erro em 147), e aceitar também os prefixos curtos
  cuja sentença coincide com a escolha do encoder leva a 0,996 (1 erro em 245); nenhum dos 29 casamentos
  curtos julgados errados coincide com o encoder. O ganho, portanto, está na regra de tamanho de
  prefixo, que vale para o artefato atual mesmo sem relaxar a caixa; ignorar a caixa é o que mantém a
  cobertura de citação depois de descartar os prefixos curtos.

- Política de casamento de citação implementada (20/09/2026, documentação fechada em 21/09,
  `challenge/utils/udv_pipeline.py`,
  `challenge/utils/build_udvs.py`, `udv.ipynb` v05): o prefixo passa a casar ignorando maiúsculas e
  minúsculas e só vira `quote_found` com 6 palavras ou mais; o prefixo curto que cai na mesma sentença
  escolhida pelo encoder (ou numa que a contenha) deixa de virar citação e passa a ser registrado junto
  com o score, no `support_type` novo `semantic_with_short_quote`, mantendo o nível pelo score. O corte
  de embeddings foi recalibrado com os pares confirmados pela regra nova (35 pares no notebook, contra
  33 do modo exato e 62 do modo que aceitava qualquer prefixo): 0,47, em `configs/udv.toml`. Rodada
  completa refeita (`udv_v0`, 20/09/2026): 2.203 UDVs, 1.020/1.065 pessoas, 260 `quote_found`, 1.793
  `semantic_match_high`, 52 `semantic_match_weak`, 8 `no_evidence`, 90 `person_not_resolved`; por
  suporte, 260 `direct_quote`, 100 `semantic_with_short_quote` e 1.745 `semantic_similarity`; offsets em
  2.104 de 2.105 evidências. Contra a rodada de 19/09: 117 opiniões entram em `quote_found`, 116 saem
  (todas de prefixo curto, 48 delas viram `semantic_with_short_quote`), 15 dos 143 `quote_found`
  mantidos passam a casar um prefixo mais longo (12 vinham de prefixo curto, 3 vão de 6 para 10
  palavras), 17 vão de alta para baixa confiança só pela mudança do corte (0,44 para 0,47) e 300
  registros têm evidência diferente; esses números por registro exigem regerar a rodada de 19/09 com o
  código de `backup/` e o corte 0,44, porque o JSONL dela foi sobrescrito. `challenge/utils/verify_udvs.py`,
  estendido para o `support_type` novo, não acusa inconsistência em `dev20` nem em `udv_v0`. O código e
  os relatórios da rodada de 19/09 ficam em `backup/` (`udv_pipeline_2026-09-19.py`,
  `build_udvs_2026-09-19.py`, `udv_v0_coverage_2026-09-19.json`, `dev20_coverage_2026-09-19.json`).
  Na amostra congelada de validação manual (`challenge/udv_manual_review.json`), a evidência de
  similaridade escolhida continua a mesma nos 65 pares, e 4 deles (3 de `semantic_match_high`, 1 de
  `embedding_diff`, os quatro com julgamento `correta` no arquivo) passam a ter citação localizada,
  saindo da camada de similaridade. Uma rodada completa com o cache de embeddings quente leva 3,1 s.
  Limitações registradas: a precisão dos casamentos aceitos ainda depende do julgamento manual dos 40
  pares de `case_insensitive_review_template.json` (os julgamentos de 20/09 são propostos por agente), e
  essa amostra foi sorteada antes da regra, então só 21 dos 40 pares caem no estrato que a política
  aceita como `quote_found` (5 viraram corroboração de prefixo curto e 14, similaridade pura); em 3
  opiniões a primeira citação que casa tem prefixo curto embora uma citação posterior casasse com
  prefixo de 6+ palavras; o prefixo curto que corrobora não precisa ser distintivo (99 dos 100 têm 4
  palavras ou menos, alguns são frases de ligação frequentes); e `quote_found` garante o prefixo de 6 ou
  10 palavras, não a citação inteira, porque aceitar um prefixo de 6 significa que o degrau de 10
  falhou. A evidência também é sempre a primeira ocorrência do prefixo na fala, e ignorar a caixa faz
  13 prefixos gravados ocorrerem mais de uma vez (eram 4 antes): em 1 deles outra ocorrência tem
  sobreposição maior com a opinião, um `direct_quote` (`udv-65-1-0`) cujos offsets apontam a passagem
  errada; trocar o critério de corroboração pelo teste direto (o prefixo cai dentro da sentença
  gravada) acrescentaria 7 registros e fica em aberto, porque a projeção de 20/09 usou o critério
  atual. Uma revisão adversarial da mudança (5 revisores, refutação em seguida) também expôs um defeito
  anterior a ela, ainda aberto: em `udv-1-1-2` a evidência é a colagem do fim de um turno com o começo
  de outro, porque `resolve_person_speech` junta os turnos com espaço e o separador de sentenças não
  quebra quando o turno termina em rubrica, e é justamente o único registro sem offsets; `verify_udvs`
  não pega esse caso, porque não checa se o texto da evidência existe na transcrição quando os offsets
  são nulos.

- Split temporal construído e verificado (21/09/2026, `challenge/utils/hearing_dates.py`,
  `challenge/utils/build_splits.py`, `challenge/utils/verify_splits.py`,
  `challenge/configs/splits.toml`, notebook `challenge/splits_v00.ipynb`). O LDS não tem campo de data,
  então a ordenação vem do carimbo de publicação da matéria (`DD/MM/AAAA - HH:MM`, com o segundo
  carimbo de `Atualizado em` descartado), presente nas 206. A extração foi validada contra o próprio
  texto antes de virar base de split: 156 matérias nomeiam o dia do debate na forma `nesta
  quarta-feira (17)`, 155 confirmam a data extraída, e a defasagem entre publicação e evento é de zero
  dia em 142 menções e de um dia em 15, nunca maior (a única sem confirmação é a audiência 111, cuja
  matéria cita um dia que caiu num sábado). Os cortes só podem cair entre datas separadas por ao menos
  2 dias, para que esse erro de um dia não troque nenhuma audiência de lado, e entre os 65 elegíveis
  escolhe-se o mais próximo de 0,70 e 0,85 da fração acumulada: 13/11/2023 (8 dias até a próxima
  audiência) e 20/12/2023 (76 dias, o recesso). Resultado: 144 audiências e 1.536 UDVs em `train`
  (18/11/2021 a 13/11/2023), 32 e 308 em `validation` (21/11 a 20/12/2023), 30 e 359 em `test`
  (05/03 a 09/05/2024). O manifesto (`artifacts/splits/temporal_v1.json`) segue o formato com
  `split_version`, `grouping_method`, `seed` e as três listas de `id`, e guarda a data extraída de cada
  audiência. `verify_splits` refaz extração, cortes e atribuição a partir do LDS e confere partição,
  cronologia, datas que não podem cair em dois conjuntos, intervalo de fronteira maior que o erro da
  data e todos os contadores do relatório: zero inconsistências, e ele acusa corretamente os cinco
  defeitos injetados num teste negativo (audiência trocada de conjunto, audiência removida, data
  alterada, fronteira alterada, contador do relatório alterado). Limitações medidas e registradas: os
  níveis de confiança não se distribuem igualmente (80 das 90 opiniões de pessoa não resolvida estão no
  treino, 6 das 8 sem evidência estão no teste); 55 dos 879 atores aparecem em mais de um conjunto e 24
  em treino e teste, o que um split temporal não isola; a maior similaridade TF-IDF entre matérias de
  conjuntos diferentes é 0,459, abaixo do corte de 0,5, então não há audiência quase duplicada
  atravessando a fronteira. A extração de comissão continua pendente: o nome aparece em texto corrido
  em 188 matérias e 202 transcrições, mas não foi transformado em rótulo canônico nem medido.

## Próximos passos

1. Split por comissão, com a extração do nome da comissão validada antes de virar base de split (o
   split temporal já está construído e verificado, ver acima). Depois dele, os splits por similaridade
   e o `GroupKFold` sobre os mesmos grupos.
2. Validação humana da política de casamento de citação já implementada: julgar à mão os 40 pares de
   `case_insensitive_review_template.json` e comparar com a precisão estimada em 20/09, lembrando que
   só 21 deles estão no estrato aceito hoje (vale considerar sortear uma amostra nova sobre os 260
   `quote_found` atuais); se a checagem humana não sustentar a regra, rever o corte de 6 palavras ou o
   tratamento dos prefixos curtos corroborados.
3. Baseline TF-IDF + SVD.
4. EDI v1 (omissão, super-representação, distorção de posição).
5. Comparação explicável entre audiências.
6. Busca simples (lexical ou híbrida).
7. Relatório de avaliação estatística (intervalo/teste, não só ponto estimado).
8. "Experimento decisivo" como roteiro do vídeo de 5 minutos.
9. Artigo científico.

## Sinais de que estamos saindo da rota

- Começar um item de P1 ou P2 antes de todo o P0 relevante estar completo e documentado.
- Um notebook com conclusão em markdown sem a célula de código que a sustente.
- Chegar perto do prazo de submissão sem o "experimento decisivo" rodável de ponta a ponta.
