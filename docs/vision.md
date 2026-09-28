# Visão do bookworm

## Memória verificável e espaço latente deliberativo sobre audiências públicas

Este documento consolida a tese científica e a especificação de construção do `bookworm`, e é a fonte
da visão de longo prazo do projeto. Ele descreve mais do que a versão 1.0 implementa: o que já existe, e
com que resultado, está em [`report.md`](report.md). Prioridades concretas para a submissão do
desafio de 2026 (o que é P0/P1/P2) e o estado de cada entrega ficam em [`docs/roadmap.md`](roadmap.md);
este documento não expira quando o desafio terminar.

---

## 1. Resumo executivo

`bookworm` é uma infraestrutura científica para representar audiências públicas em dois níveis
complementares:

1. uma **memória simbólica verificável**, formada por participantes, proposições, evidências,
   posições, relações e temporalidade;
2. um **espaço semântico latente multifacetado**, no qual transcrições, opiniões, resumos e matérias
   jornalísticas possam ser codificados, comparados, buscados, agrupados e parcialmente reconstruídos
   semanticamente.

A contribuição central do `bookworm` é aprender uma representação matemática compacta da deliberação pública que
preserve, separadamente:

- tema; atores; posições; argumentos; relações; temporalidade; relevância; evidência; incerteza;
  seleção editorial.

Essa representação deve permitir: comparar duas transcrições e explicar em quais dimensões elas se
parecem ou diferem; realizar busca semântica; clusterizar audiências; recuperar argumentos
equivalentes escritos com vocabulário diferente; avaliar a adequação de um resumo; comparar uma
matéria jornalística à audiência de origem; detectar omissões, super-representações, distorções de
posição e assimetrias editoriais; acompanhar a evolução temporal de temas e padrões argumentativos.

A meta científica é medir e explicar **desvios editoriais observáveis**, sem declarar que um
jornalista ou veículo tem "viés político", distinguindo factualidade, cobertura,
seleção, enquadramento, equilíbrio, relevância e possível orientação ideológica.

## 2. O que o projeto precisa demonstrar

- **Formulação científica nova.** O problema é formulado como `transcrição → memória deliberativa verificável → representação latente multifacetada → produtos e
  análises auditáveis`.
- **Hipóteses falsificáveis.** Cada componente responde a uma hipótese mensurável.
- **Contribuição metodológica.** Um espaço latente deliberativo que preserva propriedades não
  capturadas por embeddings textuais genéricos.
- **Evidência rigorosa.** Baselines, ablações, testes adversariais, avaliação humana, generalização
  temporal e temática, análise estatística, documentação de falhas.
- **Produto utilizável.** Busca, comparação, auditoria e geração de artefatos.
- **Legado aberto.** Código, benchmark, representações pré-computadas, documentação, modelos,
  protocolos, publicações, interface, API.

## 3. Releitura do PublicHearingBR e implicações

### 3.1 Composição

O PublicHearingBR contém 206 amostras. Cada uma combina transcrição de audiência pública, matéria da
Agência Câmara, e metadados estruturados com assunto, participantes, cargos e opiniões. Os documentos
cobrem audiências realizadas entre novembro de 2021 e maio de 2024.

### 3.2 Opiniões estruturadas

A extração inicial das matérias produziu 3.054 opiniões. Após revisão manual, correção de atribuições
e fusão de opiniões duplicadas ou artificialmente separadas, restaram **2.203 opiniões**. O conjunto
estruturado **não é** uma decomposição completa da transcrição; representa o conteúdo selecionado nas
matérias.

### 3.3 Escala

Transcrições: em média ~18.102 palavras, máximo próximo de 147.728. Matérias: em média 627 palavras. A
matéria média representa ~4,2% da transcrição (~96% de compressão textual). Cada amostra contém, em
média, 5,2 indivíduos e 10,7 opiniões.

### 3.4 Conjunto NLI

O experimento baseline extraiu 4.238 opiniões. Para avaliar alucinação, o artigo do dataset: (1) separou
falas por participante; (2) criou chunks de cinco sentenças com uma sentença de sobreposição; (3)
construiu base vetorial por participante; (4) recuperou os quatro chunks mais similares para cada
opinião; (5) pediu a um especialista que marcasse se a opinião poderia ser inferida dos quatro chunks.
Resultado manual: 3.734 opiniões válidas, 504 possíveis alucinações, taxa de 11,89%. **O próprio artigo
ressalta que esse valor é um limite superior**: uma opinião classificada como possível alucinação pode
ter suporte em trechos que não apareceram entre os quatro recuperados.

### 3.5 Baseline do artigo

Recall médio ~44,9%, precisão média ~24,8%. Baixa precisão contra a matéria não implica necessariamente
alucinação: o sistema pode recuperar uma opinião verdadeira que o jornalista não selecionou.

### 3.6 As três verdades: implicação central para o bookworm

O dataset contém três verdades parciais que **nunca devem ser confundidas**:

- **Verdade documental**: o que está sustentado pela transcrição.
- **Verdade editorial**: o que foi selecionado e redigido na matéria.
- **Verdade anotada sob recuperação**: o que o especialista conseguiu validar a partir dos quatro
  chunks recuperados (não é a mesma coisa que "verdade global").

`bookworm` deve modelar essas três camadas separadamente, em toda análise, relatório ou métrica que produzir.

## 4. Tese científica

**Tese principal:** é possível aprender um espaço latente compacto, multifacetado e parcialmente
interpretável que preserve propriedades deliberativas de documentos extensos melhor do que
representações textuais genéricas.

**Critério de superioridade:** a representação é superior quando, com menor dimensão e armazenamento,
melhora recuperação, comparação, clusterização, avaliação de resumos, detecção de posição, detecção de
relações, auditoria editorial e explicabilidade.

**Limite fundamental:** a representação é uma compressão semântica com perdas. Não deve ser apresentada
como substituta da transcrição original; deve preservar o necessário para tarefas deliberativas e
manter ligações formais com o texto fonte.

## 5. Arquitetura em quatro pilares

1. **Memória verificável**: Unidades Deliberativas Verificáveis (UDV), participantes, proposições,
   evidências, offsets, confiança, proveniência.
2. **Estrutura deliberativa**: grafo, relações, consenso, conflito, resposta, mudança de posição,
   temporalidade.
3. **Espaço Latente Deliberativo**: vetores por UDV, por participante, por audiência, fatores
   interpretáveis, decoder semântico, índices.
4. **Geração e auditoria editorial**: avaliação de resumo, avaliação de matéria, geração restrita,
   análise de cobertura, análise de desvio editorial.

## 6. Unidade Deliberativa Verificável (UDV)

Cada UDV contém:

```json
{
  "id": "udv-...",
  "hearing_id": "...",
  "actor": { "name": "...", "role": "...", "organization": "..." },
  "proposition": "...",
  "speech_act": "...",
  "target": "...",
  "stance": "...",
  "arguments": ["..."],
  "evidence": [
    {
      "text": "...",
      "start_char": 0,
      "end_char": 0,
      "speaker_turn": 0,
      "support_type": "direct"
    }
  ],
  "relations": [],
  "time_position": 0.0,
  "relevance": {},
  "uncertainty": {},
  "status": "verified"
}
```

A UDV é a ponte entre texto, estrutura e espaço latente. Proveniência sempre explícita:
`provenance = weak | model | human`. UDV extraída automaticamente **nunca** deve ser apresentada como
equivalente a anotação humana.

## 7. Representação matemática

### 7.1 Espaço multifacetado

Para uma audiência \\(H_i\\):

\\[
z_i = [z_i^{topic}, z_i^{actor}, z_i^{stance}, z_i^{argument}, z_i^{relation}, z_i^{time}, z_i^{relevance}, z_i^{evidence}, z_i^{uncertainty}]
\\]

Cada bloco pode ter dimensão distinta (ex.: \\(z_i^{topic}\\in\\mathbb{R}^{64}\\),
\\(z_i^{stance}\\in\\mathbb{R}^{32}\\)). A assinatura concatenada pode ter 256 ou 512 dimensões.

### 7.2 Hierarquia

- Vetor de evidência: \\(e_{ijk}=E_{evidence}(x_{ijk})\\)
- Vetor de UDV: \\(u_{ij}=E_{udv}(p_{ij}, e_{ij}, a_{ij}, s_{ij}, r_{ij})\\)
- Vetor de participante: \\(v_{ip}=A_{actor}(\\{u_{ij}: actor(j)=p\\})\\)
- Vetor de audiência: \\(z_i=A_{hearing}(\\{v_{ip}\\},G_i,T_i,C_i)\\), onde \\(G_i\\) é o grafo,
  \\(T_i\\) a temporalidade, \\(C_i\\) confiança e relevância.

## 8. Baselines matemáticos

- **8.1 TF-IDF + Truncated SVD**: \\(X_{tfidf}\\approx U_k\\Sigma_kV_k^\\top\\); baseline de Latent
  Semantic Analysis. Ver `experiments/concepts/tfidf.ipynb` e `experiments/concepts/svd.ipynb`.
- **8.2 Matriz UDV–características + SVD**: colunas: tema, posições, atos discursivos, participantes,
  instituições, tipos de argumento, relações, confiança, posição temporal.
- **8.3 NMF**: \\(X\\approx WH,\\quad W,H\\geq 0\\); relevante para fatores aditivos e interpretáveis.
  Ver `experiments/concepts/nmf.ipynb`.
- **8.4 Fatoração tensorial**: tensor
  \\(\\mathcal{X}\\in\\mathbb{R}^{hearing\\times actor\\times topic\\times stance\\times time}\\); métodos
  CP, Tucker, non-negative Tucker. Ver `experiments/concepts/tensor_factorization.ipynb`.
- **8.5 Embeddings genéricos**: representações de sentença/documento como baseline sem supervisão
  deliberativa. Ver `experiments/concepts/embeddings.ipynb`.
- **8.6 Autoencoder**: \\(z=E_\\theta(x), \\hat{x}=D_\\phi(z)\\). Ver `experiments/concepts/autoencoder.ipynb`.
- **8.7 Variational Autoencoder**: \\(q_\\theta(z|x),\\ p_\\phi(x|z)\\). Ver `experiments/concepts/vae.ipynb`.
- **8.8 Graph Autoencoder**: codifica estrutura de nós e relações. Ver
  `experiments/concepts/graph_encoder.ipynb`.
- **8.9 Modelo contrastivo multifacetado**: aproxima pares semanticamente equivalentes e separa
  negativos difíceis. Ver `experiments/concepts/contrastive_learning.ipynb`.
- **SparsePCA**: alternativa opcional aos baselines de fatoração. Ver
  `experiments/concepts/sparse_pca.ipynb`.

## 9. Objetivos de treinamento

\\[
\\mathcal{L} = \\lambda_{rec}\\mathcal{L}_{rec} + \\lambda_{contrast}\\mathcal{L}_{contrast} +
\\lambda_{stance}\\mathcal{L}_{stance} + \\lambda_{actor}\\mathcal{L}_{actor} +
\\lambda_{relation}\\mathcal{L}_{relation} + \\lambda_{summary}\\mathcal{L}_{summary} +
\\lambda_{evidence}\\mathcal{L}_{evidence} + \\lambda_{cluster}\\mathcal{L}_{cluster} +
\\lambda_{orth}\\mathcal{L}_{orth}
\\]

- **Reconstrução semântica**: o decoder reconstrói distribuição temática, posições, atores,
  argumentos, relações, perfil temporal e relevância.
- **Contraste**: positivos: transcrição–matéria, transcrição–resumo correto, UDV–evidência, opiniões
  equivalentes, audiências deliberativamente semelhantes. Negativos difíceis: mesmo tema com posição
  oposta, mesmas palavras com negação, mesma proposição atribuída a outra pessoa, resumo plausível sem
  suporte, matéria com distorção controlada.
- **Separação de fatores**: evitar que "tema" domine todas as dimensões, via heads separados, losses
  supervisionadas, penalidade de correlação, adversarial disentanglement, orthogonality regularization.

## 10. Comparação entre transcrições

\\[
S(A,B)=\\sum_m \\alpha_m S_m(A,B),\\quad m\\in\\{topic,\\ actor,\\ stance,\\ argument,\\ relation,\\ time\\}
\\]

Saída esperada:

```json
{
  "overall": 0.74,
  "topic": 0.91,
  "stance": 0.42,
  "argument": 0.80,
  "actors": 0.51,
  "relations": 0.68,
  "temporal": 0.63,
  "explanation": []
}
```

Toda comparação deve apontar UDVs e evidências responsáveis pela similaridade, com pesos configuráveis
e resultado reproduzível.

## 11. Busca semântica

Modalidades: por audiência (consulta textual ou audiência exemplo), por proposição (argumentos
equivalentes), multifacetada (ex.: "audiências sobre fiscalização, com apoio condicionado e forte
polarização"). Função de score:

\\[
score = \\alpha s_{latent} + \\beta s_{lexical} + \\gamma s_{graph} + \\delta s_{evidence} + \\epsilon s_{confidence}
\\]

Filtros: pessoa, comissão, data, stance, tema, nível de confiança. Métricas: Recall@k, nDCG, MRR.

## 12. Clusterização

Métodos: k-means, spherical k-means, agglomerative clustering, HDBSCAN, spectral clustering,
clustering sobre grafo, mixtures. Avaliação: silhouette, Davies–Bouldin, estabilidade (bootstrap +
adjusted Rand index), pureza quando houver rótulo, coerência humana, interpretabilidade, utilidade. Ver
`experiments/concepts/clustering.ipynb` e `experiments/concepts/umap.ipynb` (redução de dimensionalidade
para visualização).

## 13. Avaliação de resumos

\\[
Q(R,H)=w_1C_{topic}+w_2C_{stance}+w_3C_{actor}+w_4F_{relation}+w_5F_{evidence}-w_6D_{weight}-w_7H_{unsupported}
\\]

Dimensões: factualidade, cobertura temática, cobertura de posições, cobertura de atores, fidelidade
relacional, preservação de condições, distorção de peso, redundância, informação não sustentada. O
sistema deve produzir justificativas e trechos, não só um score.

## 14. Auditoria editorial

**Princípio:** o sistema não infere intenção do jornalista. Ele mede propriedades observáveis.

**Editorial Deviation Index (EDI):**

\\[
EDI = \\lambda_1 Omission + \\lambda_2 OverRepresentation + \\lambda_3 UnderRepresentation +
\\lambda_4 StanceDistortion + \\lambda_5 RelationDistortion + \\lambda_6 EvidenceAsymmetry + \\lambda_7 LexicalFraming
\\]

- **Omissão relevante:** \\(O(p)=R_H(p)(1-C_A(p))\\)
- **Super-representação:** \\(SR(p)=W_A(p)-W_H(p)\\)
- **Distorção de posição**: exemplos: apoio condicionado → apoio; dúvida → rejeição; previsão → fato;
  citação → posição própria.
- **Assimetria de evidência**: quanto contexto e justificativa cada lado recebe.
- **Enquadramento lexical**: verbos, qualificadores e rótulos, sempre com contexto.
- **Orientação ideológica**: linha futura separada; exige taxonomia explícita, anotadores
  especialistas, corpus externo, validação, linguagem probabilística. **Fora do MVP.**

O EDI nunca é retornado como um número isolado: sempre com evidências, limitações, confiança e aviso
de que não mede intenção.

## 15. Splits e prevenção de vazamento

Com 206 audiências, divisão aleatória isolada é insuficiente. Splits obrigatórios: temporal (treino em
períodos anteriores, teste em posteriores); temático (clusters temáticos não atravessam livremente
treino/teste); por comissão (generalização institucional); por similaridade (near-duplicates no mesmo
grupo); GroupKFold ou variantes; conjunto adversarial oculto (construído depois do congelamento do
modelo). Use embeddings apenas para agrupamento do split, sem usar rótulos do teste no treino. Armazene
manifesto: `{"split_version": "v1", "train": [], "validation": [], "test": [], "grouping_method": "...", "seed": 42}`.

## 16. Avaliação do espaço latente

- **Retrieval**: Recall@k, MRR, nDCG, MAP.
- **Similaridade humana**: Spearman, Kendall, correlação por dimensão.
- **Reconstrução**: erro por faceta, macro-F1, KL divergence, graph reconstruction, temporal
  reconstruction.
- **Clusterização**: métricas internas, estabilidade, coerência de especialistas.
- **Avaliação de resumo**: classificação de resumos corretos/incompletos/distorcidos/alucinados,
  correlação com especialistas.
- **Auditoria editorial**: precisão de detecção de omissão/distorção, false positive rate,
  calibração, concordância humana.
- **Compressão**: \\(CompressionRatio = \\frac{bytes(z)+bytes(metadata)}{bytes(original)}\\), reportada
  mas nunca tratada como objetivo único.

## 17. Testes adversariais

Geradores determinísticos para: negação, troca de participante, citação, mudança de posição, omissão,
repetição, super-representação, enquadramento lexical, informação plausível ausente, evidência fora de
contexto, ruído de transcrição, ordem alterada, mesma temática com posição oposta. Use Hypothesis
quando adequado. Propriedades esperadas: negação altera stance; troca de ator altera actor; omissão
reduz coverage; repetição aumenta overrepresentation; fato ausente reduz factuality.

## 18. Avaliação causal

Intervenções: remover evidência, trocar ator, inverter negação, alterar posição, remover voz
minoritária, mudar ordem, substituir verbo neutro por valorativo, aumentar artificialmente a frequência
de um ator. O espaço deve reagir na faceta correta (ex.: inversão de negação muda `stance`, não
necessariamente `topic`; troca de participante muda `actor` sem destruir `topic`; alteração lexical
afeta `framing` mais que `factuality`).

## 19. Entregável central: bookworm Latent Space

Componentes: encoder de evidência; encoder de UDV; encoder de participante; encoder de audiência;
representação multifacetada; decoder semântico; baseline SVD; baseline NMF; autoencoder; modelo
contrastivo; índice vetorial; comparação; busca; clusterização; avaliação de resumo; auditoria
editorial; explicabilidade; benchmark; API; documentação; artigo.

## 20. Entregáveis completos do projeto

- **Dados e benchmark**: perfil estatístico, splits, esquema UDV, benchmark próprio, conjunto
  adversarial, conjunto de resumos perturbados, conjunto de matérias perturbadas.
- **Modelos**: baselines lineares, embeddings, modelo latente, verificador, calibrador, decoder.
- **Produto**: CLI, API, interface, busca, comparação, clusters, auditoria (ver seção 24).
- **Ciência**: artigo principal, relatório de ablações, relatório de ética, cartão do modelo,
  documentação de reprodução.

## 21. Hipóteses de publicação (visão de longo prazo)

- **Paper 1**: A Verifiable Multifaceted Latent Space for Long-Form Public Deliberations.
- **Paper 2**: Beyond Reference Summaries: Separating Documentary Truth from Editorial Selection.
- **Paper 3**: Auditing Editorial Coverage of Public Hearings with Evidence-Grounded Representations.
- **Paper 4**: PublicHearingBR as an NLI and Deliberative Representation Benchmark.

Para a submissão de 2026, ver o escopo P0 em [`docs/roadmap.md`](roadmap.md). O artigo do desafio (≤10
páginas) deriva de um recorte de uma dessas hipóteses, não das quatro simultaneamente.

## 22. Riscos científicos e mitigação

- **Dataset pequeno**: modelos compactos, baselines lineares, pré-treinamento, regularização,
  cross-validation, weak supervision, anotação ativa.
- **Confusão entre tema e posição**: negativos difíceis, heads separados, losses supervisionadas,
  testes causais.
- **Gold editorial incompleto**: separar factualidade de seleção, usar NLI, múltiplas referências,
  avaliação humana.
- **Auditoria editorial acusatória**: medir desvio observável, evitar inferência de intenção,
  linguagem calibrada, especialistas.
- **Baixa interpretabilidade**: fatores explícitos, NMF, probes, nearest UDVs, explanations.

## 23. Por que o projeto tem potencial científico amplo

A proposta conecta long-document NLP, representation learning, matrix/tensor factorization, graph
representation learning, contrastive learning, NLI, summarization evaluation, computational
journalism, political communication, digital democracy, interpretable AI. O problema nasce em
português e no contexto brasileiro, mas é universal.

## 24. Cláusula de integridade (não negociável)

O projeto não deve:

- declarar intenção política sem evidência;
- confundir omissão com falsidade;
- confundir matéria com inventário completo da transcrição;
- confundir quatro chunks recuperados com toda a transcrição;
- apresentar vetor latente como substituto do original;
- usar confiança não calibrada;
- escolher métricas depois de ver o teste;
- omitir resultados negativos;
- ajustar o EDI depois de ver avaliação final;
- mudar splits silenciosamente;
- reportar apenas a melhor seed;
- chamar ausência na matéria de "alucinação";
- chamar label NLI de verdade global.

---

## 25. Especificação de construção

### 25.1 Restrições

- Python `>=3.12,<3.13`; `uv` para ambiente, lock e execução; tipagem estrita; caminhos com `pathlib`;
  configuração por arquivos; sementes determinísticas.
- Sem dados brutos versionados no Git; sem credenciais; sem notebooks como única implementação de
  código de produção (notebooks são para exploração e para os conceitos em `experiments/concepts/`); sem
  chamadas remotas nos testes; componentes de LLM sempre atrás de interface; execução em CPU possível
  para baselines; GPU opcional.
- Não implemente tudo de uma vez; trabalhe por milestones, cada um com código executável, testes,
  documentação, métricas, artefatos versionados e decisões registradas. Antes de alterar arquitetura,
  registre um ADR em `bookworm/docs/adr/`. Não introduza dependências sem justificar.

### 25.2 Superfície pública do bookworm

`bookworm` é uma biblioteca Python, e não um script ou um notebook. Ela é usada de quatro formas:

1. **Biblioteca importável**: `from bookworm import ...` dentro de código Python (ex.: notebooks e
   scripts de `experiments/`).
2. **CLI**: via Typer, invocada como `uv run bookworm <comando>` (ex.: `bookworm compare`,
   `bookworm search`, `bookworm cluster`).
3. **`bookworm init`**: comando que faz o scaffold de um projeto novo que depende de `bookworm` (ex.:
   usado para inicializar a estrutura de `experiments/` como projeto `uv` consumidor da lib).
4. **API**: serviço HTTP (`api/service.py`, `api/models.py`) expondo as mesmas operações da CLI, para
   ser consumido por um dashboard (ex.: Streamlit) ou outra aplicação.

Nenhum destes quatro pontos de entrada duplica lógica: todos chamam as mesmas funções públicas da
biblioteca (seção 29).

### 25.3 Estrutura do repositório

`bookworm/` é o pacote/biblioteca (entregável como "código-fonte documentado em repositório
acessível"). `experiments/` é onde os experimentos de fato rodam: é seu próprio projeto `uv`, com
`bookworm` como dependência. A árvore abaixo é a da versão 1.0; os módulos previstos nas seções 7 a 14
que ainda não existem (espaço latente, clusterização, avaliação de resumos, auditoria editorial) estão
em [`docs/roadmap.md`](roadmap.md).

```text
LICENSE, CITATION.cff, README.md, CONTRIBUTING.md

docs/
├── report.md, pipeline.md   # relatório dos experimentos e pipeline recomendado
├── methodology/             # udv, confidence, hearing_actors, actor_profiles, actor_simulation, mlx_backend
├── validation/              # guia do anotador
├── challenge/brief.md       # brief do desafio; o artigo do dataset é arXiv 2410.07495
├── vision.md, roadmap.md
└── path_map.md              # caminhos da versão de pesquisa e os desta versão

bookworm/
├── pyproject.toml, uv.lock, .python-version
├── README.md                # uso, instalação, CLI e API da biblioteca
├── docs/
│   ├── cli.md, configuration.md, architecture.md, data_model.md, testing.md, limitations.md
│   ├── actors.md, profiles.md, profile_validation.md
│   └── adr/                 # decisões de arquitetura (README.md é o índice)
├── src/bookworm/
│   ├── cli/, config.py, pipeline.py, models.py, errors.py
│   ├── data/                # schemas, leitura do LDS, datas, splits e sua verificação
│   ├── transcript/          # falantes, turnos, sentenças, offsets
│   ├── features/            # TF-IDF e codificadores de sentenças
│   ├── udv/                 # construção, citações, evidência, janelas, sinais, verificação, exportação
│   ├── actors/              # falas por ator
│   └── profiles/            # perfis de atores gerados por LLM, revisão e validação
├── tests/
│   ├── unit/
│   └── integration/         # inclui os testes marcados `dataset`, que leem o LDS local
└── web/                     # front estático da demo, alimentado pelo `bookworm export-site`

experiments/
├── pyproject.toml, uv.lock  # projeto uv próprio; depende de bookworm
├── README.md                # guia de reprodução
├── src/experiments/         # data, udv, retrieval, verifier, validation, actors, common, mlx
├── configs/                 # uma configuração TOML por experimento
├── prompts/                 # prompts versionados dos componentes de LLM
├── notebooks/               # eda, splits, udv, actor_profiles
├── scripts/                 # execução encadeada dos perfis e da simulação
├── tests/
├── dataset/                 # dados brutos do Hugging Face, nunca versionados
└── artifacts/               # saídas versionadas; MANIFEST_heavy.tsv lista as que ficam fora do Git
```

### 25.4 Modelo de dados

Validado empiricamente em `experiments/notebooks/eda.ipynb` contra os arquivos reais baixados de
`unicamp-dl/PublicHearingBR` no Hugging Face (`experiments/dataset/PublicHearingBR_LDS.jsonl` e
`PublicHearingBR_NLI.jsonl`): **nunca presuma silenciosamente nomes de campos**, sempre confira contra
o JSON real. Os modelos abaixo refletem o schema real, não o schema originalmente assumido pelos
documentos de design.

`PublicHearingBR_LDS.jsonl`: 206 registros, um por audiência, alinhados 1:1 por posição/`id` com o
arquivo NLI:

```python
from pydantic import BaseModel

class Envolvido(BaseModel):
    nome: str
    cargo: str
    opinioes: list[str]

class Metadados(BaseModel):
    assunto: str
    envolvidos: list[Envolvido]

class HearingRecord(BaseModel):
    id: int
    materia: str
    transcricao: str
    metadados: Metadados
```

`PublicHearingBR_NLI.jsonl`: mesmos 206 `id`s, mas **sem** `materia`/`transcricao` (só metadata
extraída); cada opinião aqui é bem mais rica que no LDS, com evidência recuperada e verificação de
alucinação:

```python
class VerificacaoAlucinacao(BaseModel):
    verificacao_manual: bool
    # + 12 chaves "prompt_{1,2,3}_{modelo}" (gpt-4o-mini-2024-07-18, gpt-4o-2024-08-06,
    # deepseek-chat, sabia-3.1-2025-05-08), cada uma com {"alucinacao": bool, "explicacao": str,
    # "trechos_para_basear_analise": list[str] | None}

class OpiniaoNLI(BaseModel):
    opiniao: str
    chunks_proximos: list[str]
    verificacao_alucinacao: VerificacaoAlucinacao

class EnvolvidoNLI(BaseModel):
    nome: str
    cargo: str
    opinioes: list[OpiniaoNLI]

class MetadadosExtraidos(BaseModel):
    assunto: str
    envolvidos: list[EnvolvidoNLI]
    tl_dr: str

class NLIRecord(BaseModel):
    id: int
    metadados_extraidos: MetadadosExtraidos
```

Divergências confirmadas frente ao que os documentos de design originais assumiam (mantidas aqui como
alerta, não como algo a "consertar" silenciosamente):

- **Não existem campos estruturados `date`, `committee` ou `source_url`.** A data da audiência
  (formato `dd/mm/aaaa`, presente em 206/206 matérias) e o nome da comissão (`"Comissão de ..."`)
  aparecem só como **texto livre** dentro de `materia`: extraíveis por regex/heurística, mas isso
  precisa de validação antes de virar base confiável para os splits temporal e por comissão (seção 15).
- O identificador é `id` **inteiro**, não uma `hearing_id: str`.
- `assunto` **não é um metadado canônico único**: o mesmo `id` tem um `assunto` diferente em
  `metadados` (LDS) e `metadados_extraidos` (NLI): 0/206 idênticos na amostra verificada. Cada arquivo
  passou por uma extração/LLM independente sobre a mesma transcrição.
- O arquivo NLI carrega um **sinal multi-LLM não documentado em nenhuma versão anterior deste
  documento**: além da `verificacao_manual` (a mesma anotação humana que originou os "504/4238 =
  11,89%" da seção 3.4), há 12 julgamentos automáticos de alucinação (4 modelos × 3 prompts,
  reproduzindo o experimento da seção 4.2.2 do paper do dataset). É um recurso pronto, não uma
  obrigação, que pode servir de baseline/comparação extra para a verificação de UDV e para o EDI.

### 25.5 CLI mínima

```bash
uv run bookworm inspect-schema dataset/PublicHearingBR_LDS.jsonl
uv run bookworm profile-data
uv run bookworm build-splits
uv run bookworm build-tfidf
uv run bookworm train-svd
uv run bookworm train-nmf
uv run bookworm encode-hearings
uv run bookworm compare HEARING_A HEARING_B
uv run bookworm search "consulta"
uv run bookworm cluster
uv run bookworm evaluate-summary HEARING_ID summary.txt
uv run bookworm evaluate-article HEARING_ID article.txt
uv run bookworm run-adversarial
uv run bookworm reproduce-baseline
```

### 25.6 Milestones

0. **Higiene do projeto**: inicializar, configurar lint/tipos/testes/CI local/documentação. Critério
   de aceite: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src`, `uv run pytest`
   todos passam.
1. **Ingestão e perfil do dataset**: baixar/instruir download, ler JSONL, validar, gerar parquet,
   produzir perfil, salvar hashes, detectar duplicatas, preservar texto intacto. Métricas: amostras,
   palavras por transcrição/matéria, participantes, opiniões, nulos, tamanho, datas, comissões.
   Critério de aceite: o perfil deve reproduzir aproximadamente os números da seção 3 (diferenças devem
   ser explicadas).
2. **Splits**: temporal, temático, por comissão, GroupKFold, adversarial holdout (ver seção 15).
3. **Baseline TF-IDF + SVD**: normalizar sem remover negações; TF-IDF; TruncatedSVD; normalizar
   vetores; indexar; comparar; clusterizar. Experimentos: palavra vs. char n-grams, 64/128/256
   dimensões, texto integral vs. matéria vs. opiniões concatenadas. Métricas: retrieval
   transcrição↔matéria, vizinhos, clustering, compressão. Artefatos: vocabulary, IDF, componentes,
   vetores, relatório.
4. **Matriz estruturada + SVD/NMF**: linhas: UDV provisória ou opinião estruturada; colunas: tema,
   pessoa, cargo, stance, ato, argumento, posição temporal, confiança. Use apenas campos disponíveis e
   weak labels explícitos; **não invente rótulos**. Modelos: TruncatedSVD, NMF, SparsePCA opcional.
   Liste top features por fator.
5. **Baseline NLI e evidência**: dataset de 4.238 pares. Cuidado: o label significa "inferível a
   partir dos quatro chunks apresentados", não "verdadeiro na transcrição inteira"; nomeie a tarefa
   `retrieved_context_entailment`. Modelos: TF-IDF+logistic regression, sentence embeddings+logistic
   regression, cross-encoder, transformer fine-tuned. Métricas: precision/recall por classe, ROC-AUC,
   PR-AUC, calibration, confusion matrix.
6. **UDV v0**: fontes: opiniões estruturadas, participante, matéria, evidências recuperadas. Etapas:
   atomicidade, resolução de pessoa, retrieval, suporte, offsets, confiança. UDV v0 deve ser distinguida
   de anotação humana (`provenance = weak | model | human`).
7. **Espaço multifacetado v1**: facetas mínimas: topic, actor, stance, evidence, editorial. Cada
   faceta implementa:

   ```python
   from typing import Protocol
   import numpy as np

   class FacetEncoder(Protocol):
       def fit(self, records: list[object]) -> "FacetEncoder": ...
       def transform(self, records: list[object]) -> np.ndarray: ...
       def save(self, path: Path) -> None: ...
   ```

   Comparar agregação por mean, relevance-weighted mean, attention pooling, DeepSets.
8. **Autoencoder**: entrada: vetor estruturado ou concatenação de facetas. Encoder MLP, bottleneck,
   heads de reconstrução, early stopping. Perdas: MSE (contínuas), BCE (multilabel), cross-entropy
   (categóricas), KL (distribuições). Reprodutibilidade: seeds, deterministic flags, checkpoints,
   config serializada.
9. **Contraste**: ver seção 9. Registrar como cada negativo foi criado.
10. **Grafo**: NetworkX primeiro. Nós: hearing, actor, UDV, topic. Arestas: said, supports, opposes,
    related_to, evidence_for. Avaliar PyTorch Geometric só se necessário; não adicione complexidade
    prematuramente.
11. **Comparação explicável**: ver seção 10.
12. **Busca**: ver seção 11. Índices lexical, dense, multifacetado.
13. **Clusterização**: ver seção 12.
14. **Avaliação de resumo**: ver seção 13. Pipeline: segmentar afirmações, codificar, recuperar UDVs,
    verificar, medir cobertura, medir peso, produzir relatório.
15. **Auditoria editorial**: ver seção 14. Escopo v1: omission, overrepresentation,
    underrepresentation, stance distortion, evidence asymmetry. Orientação ideológica explícita fica
    fora do MVP.
16. **Testes adversariais**: ver seção 17.
17. **Avaliação estatística**: relatórios com média, mediana, intervalo de confiança, bootstrap,
    testes pareados, tamanho de efeito. Comparações entre SVD, NMF, embeddings, autoencoder,
    contrastivo, graph. Não declarar superioridade sem intervalo e teste apropriado.

### 25.7 Armazenamento de artefatos

```text
experiments/data/artifacts/
└── experiment_id/
    ├── config.json
    ├── metrics.json
    ├── environment.txt
    ├── git_commit.txt
    ├── vectors.parquet
    ├── model/
    ├── plots/
    └── report.md
```

Use IDs estáveis e hashes.

### 25.8 Experimentos

Cada experimento recebe: config, split, seed, model, feature set. Registro em JSON/Parquet primeiro;
MLflow é opcional, não obrigatório.

### 25.9 Tipagem e qualidade

- **Ruff**: E, F, I, UP, B, SIM, PERF, RUF.
- **Mypy**: modo estrito no core.
- **Pytest**: unit, integration, property, regression. Cobertura mínima inicial: 80%.

### 25.10 Desempenho

Streaming para JSONL; parquet; batches; memmap quando necessário; não carregar transcrições
duplicadas; cache com hash; embeddings em float32; avaliar float16 apenas para armazenamento.

### 25.11 Documentação

- **README de `bookworm/`**: motivação, instalação, dados, execução, arquitetura, ética, citação (essa
  documentação vive em `bookworm/README.md`, não neste `docs/vision.md`).
- **Data card**: origem, licença, período, composição, limitações, riscos.
- **Model card**: tarefas, métricas, falhas, grupos, uso proibido.

### 25.12 API pública desejada

```python
encode_hearing(hearing_id: str) -> HearingEmbedding
compare_hearings(a: str, b: str) -> HearingComparison
search_hearings(query: SearchQuery) -> list[SearchResult]
cluster_hearings(config: ClusterConfig) -> ClusterReport
evaluate_summary(hearing_id: str, summary: str) -> SummaryEvaluation
evaluate_article(hearing_id: str, article: str) -> EditorialReport
explain_similarity(a: str, b: str) -> SimilarityExplanation
decode_profile(vector: list[float]) -> DeliberativeProfile
```

```python
class HearingComparison(BaseModel):
    overall: float
    topic: float
    actor: float
    stance: float
    argument: float
    relation: float
    temporal: float
    supporting_udvs: list[str]
    explanation: list[str]

class SummaryEvaluation(BaseModel):
    factuality: float
    topic_coverage: float
    stance_coverage: float
    actor_coverage: float
    relation_fidelity: float
    weight_distortion: float
    unsupported_claims: list[str]
    omissions: list[str]
```

## 26. Definition of Done

Uma tarefa só está pronta quando possui: implementação, testes, typing, docs, config, métricas,
artefato, limitação registrada.

## 27. Fluxo de trabalho

Cada mudança parte do milestone atual e de uma inspeção do repositório e dos dados reais: o schema do
dataset é lido dos arquivos, nunca presumido. A mudança é a menor que fica completa (código, testes,
documentação e artefato), e só entra depois de lint, tipos e testes passarem. Um milestone só termina
quando seus critérios de aceite são cumpridos. O conjunto de teste não entra em nenhuma decisão de
treinamento ou seleção, e a distinção entre verdade documental, seleção editorial e inferência limitada
aos chunks recuperados é mantida em todo artefato.

## 28. Resultado esperado

Ao fim, o repositório deve ser capaz de:

```bash
uv run bookworm encode-hearings
uv run bookworm compare H001 H002
uv run bookworm search "apoio condicionado à fiscalização independente"
uv run bookworm cluster --facet stance
uv run bookworm evaluate-summary H001 resumo.txt
uv run bookworm evaluate-article H001 materia.txt
```

Cada comando deve produzir: JSON, relatório legível, evidências, versões, confiança, limitações.

## 29. Definição final

`bookworm` deve produzir uma memória verificável e um espaço latente multifacetado da deliberação
pública, capaz de codificar, comparar, buscar, clusterizar, reconstruir semanticamente e auditar
transcrições, resumos e matérias, mantendo a distinção entre verdade documental, estrutura deliberativa
e seleção editorial.
