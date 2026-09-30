# Roadmap

Estado do projeto em 28/09/2026, versão 1.0 do repositório. [`docs/vision.md`](vision.md) guarda a
tese e a especificação de longo prazo; este documento diz o que já foi feito, o que falta e como estão
as entregas do desafio. Os números de cada item estão em
[`docs/report.md`](report.md), com o artefato de origem.

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
16/10/2026 ([`docs/challenge/brief.md`](challenge/brief.md)).

## Checklist de entregáveis do desafio

- [X] Inscrição da equipe (prazo 30/07/2026): confirmar o registro.
- [x] Artigo científico de até 15 páginas no template do desafio.
- [x] Código-fonte documentado em repositório acessível: biblioteca `bookworm/` 1.0.0 e experimentos em
      `experiments/`, com README, guia de reprodução, relatório, ADRs, licença e citação.
- [ ] Vídeo de até 5 minutos.
- [x] Demonstração interativa (recomendada): front estático em `bookworm/web/`, alimentado por
      `bookworm export-site` sobre `udv_v2`, com a verificação de atribuição por audiência e a página
      de perfis de ator (`experiments/artifacts/web/export_site_udv_v2.json`).
- [ ] Anexar o pacote de artefatos pesados (`experiments/artifacts/MANIFEST_heavy.tsv`) a um GitHub Release.
- [ ] Envio para rodrigo.barros@kunumi.com e gianlucca@kunumi.com até 30/09/2026.

## Escopo priorizado

O [`vision.md`](vision.md) descreve um programa de pesquisa amplo. Para a submissão de 2026, os itens foram
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
  (`experiments/notebooks/eda.ipynb`).
- **Splits temporais** (`temporal_v1`): 144 audiências de treino, 32 de validação e 30 de teste, pela
  data da matéria validada contra o próprio texto, com verificação independente
  (`bookworm build-splits`, `verify-splits`).
- **Benchmarks.** B1, citações mascaradas; B2, o arquivo NLI como banco de teste de recuperação e
  verificação, com o rótulo tratado como `retrieved_context_entailment`.
- **UDVs.** `udv_v0` e `udv_v1` (sentenças, corte de cosseno calibrado no treino, `threshold_v1`) e o
  pipeline recomendado `udv_v2` (janelas de duas sentenças, citação inteira, cortes recalibrados), cada
  uma reconstruível byte a byte e conferida por `verify-udvs` ([ADR 0006](../bookworm/docs/adr/0006-udv-v2-windows-full-quotes-and-verifier.md)).
  `udv_v2` é a rodada de referência; toda etapa seguinte foi refeita ou conferida sobre ela, ao lado do
  valor de `udv_v1` (`experiments/artifacts/udv/udv_v2_downstream_report.json`).
- **Trabalho 1, achar a evidência.** E1 e E2 (`retrieval_v1`: 16 recuperadores, sentença contra
  janelas) e `retrieval_v2` (Laya como reranqueador, resultado negativo).
- **Trabalho 2, confiança.** E3 v1 e v2 (NLI multilíngue, Laya, tradução NLLB e M2M100), E3x
  (verificador aprendido, avaliado no teste depois da escolha na validação), E5 (sinais derivados do cosseno), confidence_v2 E-A
  (verificador contra seis avaliadores da literatura) e a camada do verificador sobre `udv_v1` e `udv_v2`.
- **E6.** Casamento aproximado de citações e nomes, medido e desligado até a revisão das planilhas.
- **Atores.** Falas por ator entre audiências, duas regras de ligação UDV-ator, perfis gerados só com
  as audiências de treino e simulação avaliada no teste, numa rodada completa com
  `mlx-community/Qwen3.8-27B-8bit` sobre `udv_v1`; sobre `udv_v2`, as perguntas e os pedidos da
  simulação foram reconstruídos sem modelo (98 de 101 perguntas de teste idênticas). Conferência dos
  perfis contra as UDVs (`bookworm validate-profiles`) versionada para as duas rodadas
  (`experiments/artifacts/profile_validation/`).
- **Validação humana.** Amostra estratificada de 127 linhas de `udv_v1` nas audiências de teste,
  cegamento registrado, guia do anotador e scripts de pontuação. As 127 linhas e as 26 da planilha
  suplementar de `udv_v2` estão julgadas; resultado final em
  `experiments/artifacts/udv/udv_v2_precision_final.json` (relatório, seções 7.2 a 7.4). Os dois critérios
  declarados falham: `quote_found` com precisão estrita 0,7143 [0,5495; 0,8367] (25 de 35; 31 de 35
  `correta` ou `parcial`), igual nas duas rodadas; `semantic_match_high` com precisão tolerante 0,7385
  [0,6205; 0,8298] em `udv_v1` e 0,8254 [0,7138; 0,8996] em `udv_v2`, abaixo do limite inferior de 0,75.
  Os números de `udv_v2` supõem que os 78 itens `superset` mantêm o rótulo de `udv_v1`.
- **Biblioteca `bookworm` 1.0.0.** CLI e API para UDVs, splits, atores, perfis e exportação, com paridade
  testada contra os scripts de `experiments/`, tipagem estrita e testes sem rede.
- **Demo web** em `bookworm/web/`.

## Pendente

1. **Validação humana depois dos critérios que falharam.** A reanotação de 20 linhas (concordância
   intra-anotador) e o campo `existe_trecho_melhor` (revocação da busca) não foram feitos. A premissa de
   herança dos 78 itens `superset` não foi medida. Com os critérios de `quote_found` e de
   `semantic_match_high` reprovados, a evidência das UDVs não pode ser apresentada como validada nesses
   níveis; uma nova amostra, maior no `semantic_match_high`, seria necessária para decidir se a melhora de
   `udv_v2` alcança o limite de 0,75.
2. **Anotador independente.** Toda a validação tem um único anotador.
3. **Domínio do verificador.** Medir com rótulo humano o efeito de aplicar a UDVs (premissa de uma
   citação ou janela) um verificador ajustado com quatro trechos recuperados.
4. **Planilhas pendentes.** Revisão de citações e nomes do E6 e checagem manual da tradução.
5. **Perfis.** Rodada do modelo de simulação sobre `udv_v2` (comando em
   [`methodology/mlx_backend.md`](methodology/mlx_backend.md)) e revisão manual dos pares da conferência de perfis.
6. **Artefatos pesados.** Publicar o pacote no GitHub Release e fixar revisões dos modelos MLX.
7. **Split por comissão**, com a extração do nome da comissão validada antes de virar base de split.
8. **Itens P1 e P2 não iniciados:** comparação entre audiências, busca sobre as UDVs, EDI, baseline
   TF-IDF + SVD, clusterização, testes adversariais, espaço latente e os notebooks de conceito
   (`experiments/concepts/`).

## Sinais de desvio de rota

- Começar um item de P1 ou P2 com um item de P0 incompleto ou sem documentação.
- Um documento ou notebook com uma conclusão sem o artefato ou a célula que a sustenta.
- Chegar perto do prazo sem o experimento central rodável de ponta a ponta.
