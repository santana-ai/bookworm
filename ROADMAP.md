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
- `udv.ipynb` (v00 a v03), protótipo de UDV, numa amostra de 20 audiências (268 opiniões):
  - resolução de pessoa: **101/108** participantes (~93%) tiveram pelo menos um turno de fala
    localizado, depois de corrigir o regex de turno para aceitar cabeçalhos sem parênteses (convidados
    sem filiação partidária ficavam de fora);
  - v0, evidência só por citação direta: **151 opiniões sem citação** (paráfrase), **71 com citação não
    localizada**, só **32 com citação efetivamente encontrada** na fala da própria pessoa (222/268 sem
    nenhuma evidência);
  - v1, evidência por citação direta + similaridade TF-IDF (corte calibrado em 0,25, a partir da
    distribuição real de pares corretos vs. aleatórios, não escolhido a dedo): **32 por citação**, **158
    por similaridade acima do corte**, **64 por similaridade abaixo do corte** (melhor tentativa, baixa
    confiança), **14 pessoa não resolvida**. 190/268 terminam com evidência de confiança razoável;
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
    0,732 vs. 0,225 (31 pares cada), corte escolhido em 0,46 (2/31 pares corretos abaixo do corte, 0/31
    aleatórios acima). Cobertura nas mesmas 268 opiniões: 32 por citação, 217 por similaridade acima do
    corte, 5 abaixo do corte, 14 pessoa não resolvida (contra 158/64 do TF-IDF). Comparado à validação
    manual já feita, embeddings escolhe a mesma sentença que TF-IDF em 15/40 pares e uma sentença
    diferente em 25/40; desses 25, julgamento manual (mesmo arquivo
    `challenge/udv_manual_review.json`, tier `embedding_diff`) classificou 9 corretas, 10 parciais e 6
    incorretas. A cobertura de alta confiança cresce de 158 para 217 opiniões e a camada de baixa
    confiança cai de 64 para 5; nos 25 casos em que os dois métodos discordam, a escolha de embeddings
    foi julgada correta ou parcial em 19 (76%) e incorreta em 6 (24%).

## Próximos passos

1. Rodar a pipeline de UDV numa amostra maior, idealmente as 206 audiências, com métricas de cobertura.
2. Splits (temporal, por comissão), com a extração de data/comissão do texto livre validada antes de
   virar base de split.
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
