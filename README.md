# bookworm

Uma audiência pública da Câmara dos Deputados gera uma transcrição de, em média, 18 mil palavras, e a
matéria jornalística que a cobre tem cerca de 630. A matéria atribui opiniões aos participantes, mas não
diz em que ponto da fala cada opinião foi dita; quem quer conferir uma atribuição precisa ler a
transcrição inteira. O `bookworm` liga cada uma das 2.203 opiniões estruturadas do dataset
[PublicHearingBR](https://huggingface.co/datasets/unicamp-dl/PublicHearingBR) a um trecho da fala da
própria pessoa, com a posição exata na transcrição, o critério que escolheu o trecho e um sinal de
confiança separado. Cada ligação é uma UDV (Unidade Deliberativa Verificável). O projeto foi
desenvolvido para o desafio Ideias em Rede, 1ª edição (Instituto Kunumi).

Uma UDV é uma ligação automática, não uma anotação humana, e não substitui a leitura da transcrição.

## Três verdades parciais

O dataset mistura três leituras de uma mesma audiência. Tratá-las como uma só leva a conclusões
erradas, por exemplo chamar de alucinação uma opinião que a matéria simplesmente não selecionou.

- **Documental:** o que a transcrição sustenta. É o que a UDV tenta ancorar.
- **Editorial:** o que a matéria selecionou e redigiu. As 2.203 opiniões do arquivo LDS são essa
  leitura, com a redação do jornalista.
- **Anotada sob recuperação:** o que um especialista validou lendo só quatro trechos recuperados da
  fala. É o rótulo do arquivo NLI (`retrieved_context_entailment`): diz se a opinião é inferível daqueles
  quatro trechos, não se ela é verdadeira nem se está na transcrição inteira.

A tese completa, com a arquitetura de longo prazo, está em [`CONSTITUTION.md`](CONSTITUTION.md).

## Dois trabalhos

Ligar uma opinião à fala exige responder duas perguntas diferentes, e os experimentos mostram que o
melhor método não é o mesmo para as duas.

1. **Achar a evidência.** Entre as falas da pessoa, qual trecho corresponde à opinião. É um problema de
   ordenação: o trecho certo precisa vir em primeiro. Quando a opinião traz uma citação literal, o trecho
   é achado por casamento de texto; sem citação, pela similaridade de sentido entre a opinião e trechos
   de duas sentenças.
2. **Dizer quanto confiar.** Dado o trecho escolhido, qual a chance de ele de fato sustentar a opinião.
   É um problema de separação: a similaridade diz que o trecho fala do mesmo assunto, mas não que afirma
   o mesmo. Um verificador treinado para essa pergunta dá a confiança, num campo separado que não muda o
   nível da UDV.

## Pipeline final e resultados

O pipeline recomendado é `udv_v2` (decisões no
[ADR 0006](bookworm/docs/adr/0006-udv-v2-windows-full-quotes-and-verifier.md), descrição em
[`challenge/PIPELINE.md`](challenge/PIPELINE.md)):

1. resolve a pessoa pelo nome no cabeçalho dos turnos de fala;
2. procura a citação da opinião (prefixo de 6 ou mais palavras) nos turnos da pessoa e estende a
   evidência até o fim da citação;
3. sem citação, escolhe a janela de duas sentenças consecutivas do mesmo turno com maior cosseno no
   encoder `PORTULAN/serafim-335m-portuguese-pt-sentence-encoder`; o nível é `semantic_match_high` a
   partir de 0,50 e `semantic_match_weak` abaixo;
4. aplica o verificador aprendido (E3x) ao texto da evidência e grava a probabilidade de suporte e as
   decisões em dois cortes num arquivo lateral.

Resultados principais, do resumo executivo de
[`challenge/RELATORIO_EXPERIMENTOS.md`](challenge/RELATORIO_EXPERIMENTOS.md), onde cada número tem o
arquivo de origem:

| achado | número |
|---|---|
| Opiniões com evidência localizada (`udv_v1` e `udv_v2`) | 2.105 de 2.203 (90 de pessoas não resolvidas, 8 sem unidade candidata) |
| Citações literais localizadas (`quote_found`) | 277 |
| Níveis semânticos de `udv_v2` (corte de cosseno 0,50) | 1.744 `semantic_match_high`, 84 `semantic_match_weak` |
| Verificador sobre as 2.105 evidências de `udv_v2` | passam 1.750 (0,8314) no corte de premissa UDV 0,2429 e 826 (0,3924) no corte do benchmark 0,7478 |
| Recuperação no benchmark NLI (validação, sentença, `serafim_335m`) | acc@1 0,8774, MRR 0,9295 |
| 15 recuperadores alternativos contra o `serafim_335m` | nenhum o supera na sentença depois de Holm |
| Janelas de 2 sentenças contra sentença, mesmo recuperador | lift +0,0143 [-0,0091; 0,0409], Holm 0,4104 |
| Laya como reranqueador do top 20 (resultado negativo) | MRR 0,8690 contra 0,9295, Holm 0,0004 |
| Sinais derivados do cosseno como confiança (E5) | nenhum IC de diferença de AURC exclui zero |
| Verificador primário (E3x), teste final, 559 opiniões | ROC AUC 0,9122 [0,8698; 0,9454], kappa 0,5924 |
| Verificador contra o cosseno na validação (confidence_v2 E-A) | ROC AUC 0,8758 contra 0,7302, Δ +0,1457 [0,0852; 0,2159], Holm 0,012 |
| Validação humana parcial de `udv_v1`, `quote_found` (precisão estrita) | 17 de 18, 0,9444 [0,7424; 0,9901] |
| Validação humana parcial de `udv_v1`, `semantic_match_high` (precisão tolerante) | 25 de 32, 0,7812 [0,6125; 0,8898] |

Os intervalos das linhas de recuperação e do verificador são de 95% por bootstrap de audiência; os da
validação humana são de Wilson a 95%. As duas linhas de validação humana vêm de uma leitura
intermediária de 65 das 127 linhas da amostra de `udv_v1`, feita numa planilha que ainda está sendo
preenchida e será versionada quando completa; a planilha deste repositório
(`challenge/artifacts/validation/human_validation_v1_udv_v1/annotation.csv`) ainda está vazia. Nenhum
critério declarado da validação pode ser decidido com a leitura parcial (relatório, seção 7). A precisão
humana de `udv_v2` não foi medida. 101 itens da amostra herdam o rótulo de `udv_v1`: 23 têm a mesma
evidência nas duas versões e 78 têm em `udv_v2` uma evidência que contém todo o trecho de `udv_v1` no
mesmo turno, regra declarada em `challenge/configs/validation_sample.toml` que torna a precisão
conservadora quanto aos rótulos `parcial` e `incorreta`. As 26 linhas da planilha suplementar de
`udv_v2`, com os itens cuja evidência é texto novo, ainda não foram julgadas (relatório, seção 7.3). O verificador foi ajustado com premissas de quatro trechos e é aplicado a evidências de uma
citação ou janela, uma mudança de domínio que não foi medida com rótulo humano.

`udv_v0` e `udv_v1` (evidência de uma sentença, corte 0,45) ficam como histórico e base de comparação.
Toda etapa depois da UDV (verificador, ligação a atores, perfis, perguntas da simulação, exportação da
demo) foi refeita ou conferida sobre `udv_v2`, ao lado do valor de `udv_v1`, em
[`challenge/artifacts/udv/udv_v2_downstream_report.json`](challenge/artifacts/udv/udv_v2_downstream_report.json).

## Demo web

`bookworm/web/` é um front estático que mostra, para cada audiência, a opinião da matéria, o trecho da
transcrição escolhido, as unidades candidatas e as duas medidas (cosseno e verificador), além de uma página
por ator com o perfil gerado. Os dados saem de `bookworm export-site` sobre `udv_v2`; a exportação confere
que a rodada foi construída com a configuração passada e recusa uma rodada de outra configuração. Como
exportar e servir: [`bookworm/web/README.md`](bookworm/web/README.md).

## Organização do repositório

| caminho | o que é |
|---|---|
| [`bookworm/`](bookworm/README.md) | a biblioteca Python: construção e verificação de UDVs, splits temporais, falas e perfis de atores, exportação para a demo web; CLI `bookworm` |
| [`bookworm/docs/adr/`](bookworm/docs/adr/README.md) | decisões de arquitetura registradas |
| [`challenge/`](challenge/README.md) | os experimentos sobre o PublicHearingBR: scripts, configurações, artefatos versionados e notebooks; o README é o guia de reprodução |
| [`challenge/PIPELINE.md`](challenge/PIPELINE.md) | o que cada experimento decidiu e o pipeline `udv_v2` |
| [`challenge/RELATORIO_EXPERIMENTOS.md`](challenge/RELATORIO_EXPERIMENTOS.md) | relatório completo, com a origem de cada número, limitações e referências |
| [`CONSTITUTION.md`](CONSTITUTION.md) | tese científica e especificação de longo prazo |
| [`ROADMAP.md`](ROADMAP.md) | estado do projeto e checklist de entregas do desafio |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | ambiente de desenvolvimento, verificações e convenções |

O artigo do desafio fica num repositório próprio:
[github.com/JoaoVitorBoer/bookworm](https://github.com/JoaoVitorBoer/bookworm).

## Início rápido

Requer Python 3.12 e [uv](https://docs.astral.sh/uv/). O caminho abaixo roda em CPU, sem modelo, e
confere a execução publicada `udv_v2` recalculando cada UDV a partir do dataset.

```bash
git clone https://github.com/santana-ai/bookworm.git
cd bookworm/challenge
uv sync
uv run python -m utils.download_dataset

cd ../bookworm
uv sync
uv run pytest

cd ../challenge
uv run --project ../bookworm bookworm verify-udvs --config configs/udv_v2.toml --run-name udv_v2
```

O último comando termina com `"problems": {}`. Um exemplo completo com o encoder TF-IDF, que constrói
UDVs em segundos sem GPU, está no [README da biblioteca](bookworm/README.md#início-rápido-em-cpu). A
sequência de todas as etapas, com os comandos e o tempo de cada uma, está no
[README de `challenge/`](challenge/README.md).

## Reprodutibilidade

- **Dados fixados.** O download usa uma revisão fixada do dataset no Hugging Face, e toda configuração
  que lê o LDS guarda o sha256 do arquivo e para se ele for outro.
- **Modelos fixados.** Todo modelo do Hugging Face é carregado offline numa revisão (commit) registrada
  na configuração do experimento; a lista, com licenças, está na seção 11.2 do relatório. Os modelos MLX
  da rodada de perfis e simulação são a exceção: `challenge/mlx_alternative/config.yaml` não fixa
  revisão.
- **Sementes.** Cada experimento declara sua semente e o número de réplicas do bootstrap (relatório,
  seção 9.2); os splits, as calibrações e a amostra de validação usam semente 42.
- **Proveniência.** Cada relatório JSON grava o sha256 da configuração, do código e das entradas que o
  produziram, e o ambiente (versões de Python e bibliotecas).
- **Artefatos pesados.** As saídas por item das rodadas (escores, consultas, features, predições; 369
  arquivos) não estão no Git. [`challenge/artifacts/MANIFEST_heavy.tsv`](challenge/artifacts/MANIFEST_heavy.tsv)
  lista cada uma com tamanho, sha256 e o comando que a regenera. Um pacote com esses arquivos será
  anexado a um GitHub Release deste repositório (pendente).
- **Histórico.** O histórico completo de pesquisa, incluindo versões antigas de notebooks e scripts e as
  saídas pesadas, fica na tag `research-2026-09-28`. `challenge/README.md` mostra como restaurar os
  arquivos pesados a partir dela.

## Integridade científica

As regras abaixo valem para todo experimento do repositório:

- **O conjunto de teste não escolhe nada.** Splits temporais (144 audiências de treino, 32 de validação,
  30 de teste) separam as audiências pela data; cortes, métodos e hiperparâmetros são escolhidos no
  treino e na validação, e todo comando que lê o teste exige a opção `--final-test`.
- **Um rótulo NLI não é verdade global.** Ele diz se a opinião é inferível de quatro trechos
  recuperados, e só isso.
- **Ausência na matéria não é alucinação.** Uma opinião sem evidência é uma opinião para a qual o
  método não achou trecho; nada é classificado como erro da matéria por isso.
- **Nenhuma inferência de intenção política.** O projeto descreve o que é observável nos textos; não
  infere intenção de participantes, jornalistas ou veículos.
- **Resultados negativos e parciais são relatados como são**, e as decisões que mudaram depois de ver
  resultados estão registradas nas configurações e no relatório.

## Licença

O código é distribuído sob a licença MIT ([`LICENSE`](LICENSE)).

### Dados e modelos

- Os dados do PublicHearingBR, e os trechos de transcrição e de matéria reproduzidos nos artefatos deste
  repositório, seguem os termos do próprio dataset, definidos pelos seus autores na página do
  Hugging Face. A licença MIT não se aplica a eles.
- Os modelos de terceiros usados nos experimentos têm licenças próprias, listadas na seção 11.2 de
  [`challenge/RELATORIO_EXPERIMENTOS.md`](challenge/RELATORIO_EXPERIMENTOS.md). A maioria é MIT ou
  Apache-2.0; o tradutor `facebook/nllb-200-distilled-600M` é CC-BY-NC-4.0 (uso não comercial), e as
  traduções derivadas dele ficam sujeitas a essa restrição. O `ruanchaves/mdeberta-v3-base-assin2-entailment`
  não declara licença no Hugging Face.

## Citação

Os metadados de citação estão em [`CITATION.cff`](CITATION.cff). Ao usar os dados, cite também o artigo
do dataset:

```bibtex
@misc{fernandes2024publichearingbr,
  title         = {PublicHearingBR: A Brazilian Portuguese Dataset of Public Hearing Transcripts for Summarization of Long Documents},
  author        = {Leandro Carísio Fernandes and Guilherme Zeferino Rodrigues Dobins and Roberto Lotufo and Jayr Alencar Pereira},
  year          = {2024},
  eprint        = {2410.07495},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2410.07495}
}
```

## Equipe

- Arthur Germano
- João Vitor Boer Abitante
- Henrique Santana
