# Roadmap

Estado do projeto em 28/09/2026, versão 1.0 do repositório. [`CONSTITUTION.md`](CONSTITUTION.md) guarda a
tese e a especificação de longo prazo; este documento diz o que já foi feito, o que falta e como estão
as entregas do desafio. Os números de cada item estão em
[`challenge/RELATORIO_EXPERIMENTOS.md`](challenge/RELATORIO_EXPERIMENTOS.md), com o artefato de origem.

## Norte

Aprender uma representação verificável da deliberação pública (quem disse o quê, com que evidência, com
que confiança) que sirva para comparar, buscar, clusterizar, avaliar resumos e auditar cobertura
editorial, sem nunca substituir a transcrição original.

## Hipótese

É possível ligar cada opinião publicada numa matéria a um trecho verificável da transcrição de origem,
com um nível de confiança explícito, e usar esse vínculo para medir (nunca acusar) o desvio entre a
cobertura editorial e o que foi efetivamente dito numa audiência pública.

## Prazos

Inscrição até 30/07/2026; submissão até 30/09/2026; avaliação de 01 a 15/10/2026; resultado em
16/10/2026 ([`challenge/desafio_ideias_em_rede.md`](challenge/desafio_ideias_em_rede.md)).

## Checklist de entregáveis do desafio

- [ ] Inscrição da equipe (prazo 30/07/2026): confirmar o registro.
- [ ] Artigo científico de até 10 páginas no template do desafio, no repositório
      [github.com/JoaoVitorBoer/bookworm](https://github.com/JoaoVitorBoer/bookworm).
- [x] Código-fonte documentado em repositório acessível: biblioteca `bookworm/` 1.0.0 e experimentos em
      `challenge/`, com README, guia de reprodução, relatório, ADRs, licença e citação.
- [ ] Vídeo de até 5 minutos.
- [x] Demonstração interativa (recomendada): front estático em `bookworm/web/`, alimentado por
      `bookworm export-site`, com a verificação de atribuição por audiência. A página de perfis de ator
      entra num PR seguinte.
- [ ] Anexar o pacote de artefatos pesados (`challenge/artifacts/MANIFEST_heavy.tsv`) a um GitHub Release.
- [ ] Envio para rodrigo.barros@kunumi.com e gianlucca@kunumi.com até 30/09/2026.

## Escopo priorizado

O `CONSTITUTION.md` descreve um programa de pesquisa amplo. Para a submissão de 2026, os itens foram
priorizados assim:

- **P0, necessário para o artigo e o vídeo:** qualidade de projeto (lint, tipos, testes); leitura e
  perfil do dataset; splits sem vazamento; UDVs com evidência verificável; confiança da evidência com
  avaliação estatística (intervalo e teste, não só estimativa pontual); validação humana; biblioteca
  importável com CLI.
- **P1, se houver tempo:** comparação explicável entre audiências, busca sobre as UDVs, perfis de ator,
  índice de desvio editorial (EDI) v1, baseline TF-IDF + SVD, clusterização com estabilidade, testes
  adversariais determinísticos.
- **P2, fora desta submissão:** espaço latente multifacetado (autoencoder, VAE, aprendizado contrastivo,
  codificador de grafo, fatoração tensorial), avaliação causal, as quatro hipóteses de publicação.

## Concluído

Cada item tem código, configuração, testes e artefato versionado.

- **Dataset.** Download reprodutível numa revisão fixada, com sha256 conferido em toda leitura; EDA que
  mapeia o schema real do LDS e do NLI e mostra que os dois arquivos vêm de extrações diferentes
  (`challenge/eda_v01.ipynb`).
- **Splits temporais** (`temporal_v1`): 144 audiências de treino, 32 de validação e 30 de teste, pela
  data da matéria validada contra o próprio texto, com verificação independente
  (`bookworm build-splits`, `verify-splits`).
- **Benchmarks.** B1, citações mascaradas; B2, o arquivo NLI como banco de teste de recuperação e
  verificação, com o rótulo tratado como `retrieved_context_entailment`.
- **UDVs.** `udv_v0` e `udv_v1` (sentenças, corte de cosseno calibrado no treino, `threshold_v1`) e o
  pipeline recomendado `udv_v2` (janelas de duas sentenças, citação inteira, cortes recalibrados), cada
  uma reconstruível byte a byte e conferida por `verify-udvs` ([ADR 0006](bookworm/docs/adr/0006-udv-v2-windows-full-quotes-and-verifier.md)).
- **Trabalho 1, achar a evidência.** E1 e E2 (`retrieval_v1`: 16 recuperadores, sentença contra
  janelas) e `retrieval_v2` (Laya como reranqueador, resultado negativo).
- **Trabalho 2, confiança.** E3 v1 e v2 (NLI multilíngue, Laya, tradução NLLB e M2M100), E3x
  (verificador aprendido, avaliado no teste depois da escolha na validação), E5 (sinais derivados do cosseno), confidence_v2 E-A
  (verificador contra seis avaliadores da literatura) e a camada do verificador sobre `udv_v1` e `udv_v2`.
- **E6.** Casamento aproximado de citações e nomes, medido e desligado até a revisão das planilhas.
- **Atores.** Falas por ator entre audiências, duas regras de ligação UDV-ator, perfis gerados só com
  as audiências de treino e simulação avaliada no teste, numa rodada completa com
  `mlx-community/Qwen3.8-27B-8bit`.
- **Validação humana.** Amostra estratificada de 127 linhas de `udv_v1` nas audiências de teste,
  cegamento registrado, guia do anotador e scripts de pontuação; leitura intermediária de 65 linhas
  (relatório, seção 7.2).
- **Biblioteca `bookworm` 1.0.0.** CLI e API para UDVs, splits, atores, perfis e exportação, com paridade
  testada contra os scripts de `challenge/`, tipagem estrita e testes sem rede.
- **Demo web** em `bookworm/web/`.

## Pendente

1. **Validação humana de `udv_v1`.** Completar as 127 linhas (hoje 65, numa planilha ainda não
   versionada; a planilha do repositório está vazia) e a reanotação de 20 linhas para a concordância
   intra-anotador; só então os critérios declarados podem ser decididos. O critério de `quote_found` já
   não pode ser atingido, porque exige zero erros e um erro foi anotado.
2. **Validação humana de `udv_v2`.** Os itens semânticos e as citações estendidas precisam de julgamento
   com a evidência de `udv_v2` (`challenge/artifacts/udv/udv_v2_annotation_plan.json`).
3. **Domínio do verificador.** Medir com rótulo humano o efeito de aplicar a UDVs (premissa de uma
   citação ou janela) um verificador ajustado com quatro trechos recuperados.
4. **Planilhas pendentes.** Revisão de citações e nomes do E6 e checagem manual da tradução.
5. **Perfis.** Rodada versionada de `bookworm validate-profiles`; página de perfis na demo web.
6. **Artefatos pesados.** Publicar o pacote no GitHub Release e fixar revisões dos modelos MLX.
7. **Split por comissão**, com a extração do nome da comissão validada antes de virar base de split.
8. **Itens P1 e P2 não iniciados:** comparação entre audiências, busca sobre as UDVs, EDI, baseline
   TF-IDF + SVD, clusterização, testes adversariais, espaço latente e os notebooks de conceito
   (`challenge/concepts/`).

## Sinais de desvio de rota

- Começar um item de P1 ou P2 com um item de P0 incompleto ou sem documentação.
- Um documento ou notebook com uma conclusão sem o artefato ou a célula que a sustenta.
- Chegar perto do prazo sem o experimento central rodável de ponta a ponta.
