# bookworm — memória verificável e espaço latente deliberativo

`bookworm` (nome provisório, pode mudar) é o projeto que este repositório constrói para o desafio
**Ideias em Rede, 1ª Edição** (Instituto Kunumi). Este `CLAUDE.md`, na raiz, é o único arquivo de
instruções e vale para o repositório inteiro: **não coloque arquivos meta de agente** (`CLAUDE.md`,
`AGENTS.md`) dentro de `bookworm/` ou `challenge/`.

## Onde as coisas ficam

- **[`bookworm/`](bookworm/)** — o **core**: a biblioteca `bookworm`, empacotada como projeto `uv`
  normal (`pyproject.toml`, `src/bookworm/`, `tests/`, `docs/`). Nenhum experimento roda aqui dentro.
  `bookworm` evolui a partir do que se aprende em `challenge/`: o que funciona lá é promovido para cá.
  O front web da demo (visualização da verificação de atribuição e perguntas extrativas) mora em
  `bookworm/web/` e consome o que a biblioteca exporta.
- **[`challenge/`](challenge/)** — onde tudo acontece de fato: comandos são rodados, notebooks são
  executados, dados são analisados, arquivos são salvos. É seu **próprio projeto `uv`**, tendo
  `bookworm` como uma de suas dependências. Contém `challenge/README.md` (o que é o dataset, incluindo a
  distinção entre os arquivos LDS e NLI), `challenge/desafio_ideias_em_rede.md` (brief oficial do
  desafio), `challenge/2410.07495v2.pdf` (paper do dataset), `challenge/dataset/` (dados brutos baixados
  do Hugging Face via `challenge/utils/download_dataset.py`, nunca versionados), `challenge/utils/`
  (scripts reprodutíveis: download, construção e verificação de splits e de UDVs, geração de material
  para revisão manual etc.), `challenge/configs/` (configuração por arquivo, ex.: `splits.toml`,
  `udv.toml`), `challenge/artifacts/` (saídas versionadas desses scripts; a exceção é
  `artifacts/cache/`, ignorada no Git por conter caches derivados regeneráveis) e o notebook mais recente
  de cada frente de investigação (`challenge/eda_vXX.ipynb`, `challenge/splits_vXX.ipynb`,
  `challenge/udv_vXX.ipynb`, ...).
- **[`backup/`](backup/)** — guarda o histórico completo de versões de notebooks e outros arquivos
  sensíveis a perda de trabalho, independente do histórico do Git. Cada versão nova de um notebook ganha
  um sufixo `vXX` (`eda_v00.ipynb`, `eda_v01.ipynb`, ...); `backup/` acumula todas as versões, enquanto a
  pasta de trabalho (`challenge/`) mantém só a mais recente.
- **[`CONSTITUTION.md`](CONSTITUTION.md)** — a tese científica e a especificação de construção
  completas do `bookworm` (visão de longo prazo, arquitetura em 4 pilares, milestones, cláusula de
  integridade). Não expira quando o desafio terminar.
- **[`ROADMAP.md`](ROADMAP.md)** — controle de progresso vivo: escopo priorizado (P0/P1/P2), onde
  estamos agora e os próximos passos. Ao contrário de `CONSTITUTION.md`, é atualizado a cada milestone
  concluído ou decisão de rota.
- **`bookworm/README.md`** (quando o projeto for scaffolded) — documentação de uso/instalação/API da
  biblioteca. Fica lá, não em `CONSTITUTION.md` nem aqui.
- **Arquivos `claude_*.py` na raiz**: ferramentas de trabalho mantidas por mim (Claude) entre uma
  sessão e outra, para não ter que recriar do zero uma análise já feita (ex.: `claude_generate_udv_manual_review.py`
  guarda os julgamentos que propus para uma amostra de revisão manual). A raiz é território de
  agente/usuário (só `CLAUDE.md`, `AGENTS.md`, `CONSTITUTION.md`, `ROADMAP.md` e esses `claude_*.py`
  vivem aqui), então esses arquivos podem referenciar quem os produziu sem problema. Isso é diferente de
  `bookworm/`/`challenge/`, que precisam ficar limpos, reproduzíveis por qualquer humano, sem nenhuma
  marca de autoria de agente (ver §4).

## 1. O desafio, em fatos

- **Dataset:** PublicHearingBR — 206 audiências públicas da Câmara dos Deputados, cada uma com
  transcrição, matéria jornalística e opiniões estruturadas por participante. ~18 mil palavras/transcrição,
  ~627 palavras/matéria (~96% de compressão), 2.203 opiniões estruturadas, 4.238 pares NLI
  (`retrieved_context_entailment`, não "verdade global").
- **Prazos:** inscrição até **30/07/2026**, submissão até **30/09/2026** (avaliação 01–15/10, resultado 16/10/2026).
- **Entregáveis:** artigo científico (≤10 páginas, template do desafio) + código-fonte documentado em
  repositório acessível + vídeo (≤5 min). Dashboard/demo interativa é recomendado, não obrigatório.
- **Critérios de avaliação:** Inovação e Originalidade 40% · Rigor Metodológico 30% · Impacto e
  Aplicabilidade 10% · Qualidade da Apresentação e Código 20%.
- **Trilhas (A–D) são só inspiração**, não obrigatórias — o brief incentiva explicitamente cruzar
  fronteiras entre elas ou propor tarefas inéditas. O nome do evento, "Ideias em Rede", e a exigência
  de equipe multi-Colab deixam claro que o objetivo é **contribuição científica real e colaboração em
  rede**, não encaixe em uma categoria fixa. `bookworm` naturalmente atravessa a Trilha A (auditoria
  documental/RAG) e a Trilha B (mineração de opiniões/atores), por causa da UDV — isso não precisa ser
  "escolhido", é consequência do desenho.

## 2. Visão de longo prazo

Ver [`CONSTITUTION.md`](CONSTITUTION.md) por inteiro. Resumo mínimo: o dataset contém três verdades
parciais que não devem ser confundidas — **documental** (o que a transcrição sustenta), **editorial**
(o que foi selecionado e redigido na matéria) e **anotada sob recuperação** (o que um especialista
validou a partir de só 4 chunks). A tese central é aprender um espaço latente compacto e multifacetado
(tema/atores/posições/argumentos/relações/temporalidade/relevância/evidência/incerteza) que supere
representações genéricas em recuperação, comparação, clusterização, avaliação de resumo e auditoria
editorial — sem nunca ser apresentado como substituto da transcrição original. A cláusula de
integridade do `CONSTITUTION.md` (seção 24) é não negociável em qualquer fase do projeto.

## 3. Escopo e progresso

Ver [`ROADMAP.md`](ROADMAP.md): escopo priorizado (P0/P1/P2), o que já está pronto, e os próximos
passos. Esse conteúdo vive lá, não aqui, porque muda a cada milestone; este documento (`CLAUDE.md`) só
guarda as regras de como trabalhar.

## 4. Regras de engenharia

- **Postura de trabalho científico (vale sempre, não só quando alguém lembrar):** isto é um trabalho
  científico, mesmo tendo uma aplicação prática em vista (o desafio). Nada de otimismo ou pressa em
  resultado: nenhuma suposição (formato de dado, taxa de acerto de uma heurística, cobertura de um
  método) vira conclusão antes de ser calculada e mostrada de verdade. Resultado ruim ou parcial é
  relatado como é, nunca suavizado. Todo conceito novo introduzido em qualquer documento ou notebook
  precisa ser explicado pelo **problema que resolve**, não só pelo mecanismo. Toda conclusão, em
  qualquer artefato (não só notebooks), precisa ser rastreável e reproduzível: a célula, o script ou o
  comando que a gerou tem que existir e poder ser executado de novo.
- Python 3.12, `uv` para ambiente/lock/execução; tipagem estrita; `pathlib` para caminhos; configuração
  por arquivos; sementes determinísticas.
- Nunca presuma schema do dataset: inspecione o JSONL real antes de codar contra ele (`inspect-schema`:
  chaves, tipos, nulos, cardinalidades, exemplos truncados).
- Sem dados brutos versionados no Git, sem credenciais, sem chamadas remotas nos testes, componentes de
  LLM sempre atrás de interface.
- Trabalhe por milestones pequenos e completos: código executável + testes + docs + métricas + artefato
  versionado + decisão registrada (ADR em `bookworm/docs/adr/` antes de mudar arquitetura).
- Segurança científica (não negociável): não usar o conjunto de teste para escolher hiperparâmetros; não
  ajustar EDI depois de ver avaliação final; não mudar splits silenciosamente; não reportar só a melhor
  seed; não inferir intenção política; não chamar ausência na matéria de "alucinação"; não chamar label
  NLI de verdade global.
- Dependências pesadas (`torch`, `faiss-cpu`, `hdbscan`, `umap-learn`, `sentence-transformers` etc.) só
  entram quando o milestone P1/P2 correspondente for de fato atacado — comece o ambiente enxuto (baseline
  TF-IDF/SVD roda em CPU sem elas).
- "CI local" (rodar lint/tipos/testes antes de cada milestone) é suficiente — não é necessário montar
  infraestrutura de CI real dado o prazo do desafio.
- **Código nunca referencia `CONSTITUTION.md` ou `CLAUDE.md`**: nada de comentários ou docstrings do
  tipo "conforme CONSTITUTION.md seção X" ou "regra do CLAUDE.md" dentro de código-fonte ou notebooks.
  Esses documentos orientam quem escreve o código; o código em si não os cita.
- **Nunca atribua autoria ou julgamento a "este agente"/IA dentro de `bookworm/` ou `challenge/`**
  (notebooks, código, dados): isso é vazamento de informação meta, mesmo sem citar
  `CONSTITUTION.md`/`CLAUDE.md` pelo nome, e vale tanto para frases em prosa quanto para conteúdo gerado
  por IA hardcoded onde deveria haver trabalho humano (ex.: julgamento de uma checagem manual). Quando um
  julgamento humano for necessário para validar algo, o script que mora em `challenge/utils/` deve gerar
  só o material bruto/esqueleto (ex.: um JSON com o campo de julgamento vazio), nunca a avaliação já
  preenchida; o preenchimento é feito à mão, editando o artefato. Essa regra não vale para os arquivos
  `claude_*.py` na raiz (ver "Onde as coisas ficam"), que existem justamente para guardar esse tipo de
  trabalho de agente.

## 5. Convenções de idioma

- **Código** (variáveis, classes, funções, módulos) sempre em **inglês**, tanto em `bookworm/` quanto
  em `challenge/`.
- **Comentários no código não devem existir**; se existirem mesmo assim, em inglês.
- **Notebooks** (`challenge/`, incluindo `challenge/concepts/`): código segue a mesma regra acima
  (inglês), mas as células de texto/markdown são em **português (pt-BR)**, e os poucos comentários de
  código que existirem dentro do notebook também em pt-BR — diferente da regra geral, porque notebooks
  são material didático/exploratório.
- Sempre que "português" for mencionado em qualquer documento deste repositório, significa **pt-BR**
  especificamente (não português europeu).
- **Nunca use travessão/em-dash (—) em texto corrido** (markdown, documentação, células de prosa):
  troque por vírgula, ponto e vírgula, dois pontos ou ponto final. Essa regra é sobre a pontuação de
  prosa, não sobre hífens de separação, marcadores de lista ou intervalos numéricos, que continuam
  normais.
- **Nada de clichês, sensacionalismo ou frases de efeito em prosa técnica** (notebooks, documentação,
  conclusões): descreva o achado e a causa diretamente. Em especial, nunca use a estrutura retórica
  "não é X, é Y" ("não é uma falha, é uma oportunidade") como recurso de estilo, nem comentários sobre
  "o que este resultado representa" quando isso não acrescenta informação nova. Se a frase pode ser
  cortada sem perder conteúdo, corte a frase inteira, não só reformule.

## 6. Convenção de notebooks

Todo notebook do repositório (o notebook de EDA em `challenge/`, futuros `challenge/concepts/*.ipynb`
etc.) segue esta sequência fixa:

1. `md` H1 (`#`): título do notebook (ex.: para a EDA, "Análise Exploratória de Dados").
2. `md`: explicação básica do que o arquivo faz, sem entrar em muitos detalhes.
3. `code`: imports gerais, usados na maior parte do notebook.
4. `code` (só se aplicável): qualquer client ou `dotenv` que precise rodar de antemão.
5. A partir daqui vêm as seções, cada uma iniciada por um subtítulo `##` (H2). Evite muitas quebras,
   no máximo `###` (H3), e só se fizer muito sentido.

Toda conclusão em célula markdown precisa estar apoiada por célula(s) de código, no mesmo notebook, que
produzem o número/evidência citado: nunca uma afirmação solta sem a célula que a sustenta. Quando surgir
uma nova investigação sobre um tema que já tem notebook (ex.: mais uma pergunta sobre o dataset), ela
entra como seção nova nesse mesmo notebook, em vez de espalhar em notebooks novos por achado.

Regras por célula, em qualquer notebook:
- imports específicos de uma seção vêm logo após o markdown daquela seção, não no topo do notebook;
- cada célula de código tem **uma responsabilidade só** — nunca várias funções na mesma célula;
- evite imports dentro de função/classe.

## 7. Checklist de entregáveis do desafio

Vive em [`ROADMAP.md`](ROADMAP.md), seção "Checklist de entregáveis do desafio".
