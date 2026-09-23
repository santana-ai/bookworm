# Guia de anotação: validação humana das UDVs

Cada linha da planilha liga uma afirmação que o conjunto de dados atribui a um participante de uma
audiência pública a um trecho da transcrição escolhido pelo programa. A anotação mede com que
frequência esse trecho sustenta de fato a afirmação, e com que frequência o programa deixou de
encontrar a fala de alguém que falou. Todas as linhas seguem o mesmo critério, sem tentar adivinhar
como o trecho foi escolhido.

## Fluxo

Os comandos rodam dentro de `challenge/`; `<rodada>` é o nome da rodada de UDVs escolhida. O
gerador só aceita uma rodada cujo limiar do codificador foi calibrado sem nenhuma audiência de teste
e sem audiências do conjunto amostrado, e recusa as outras antes de ler qualquer registro; hoje só
`udv_v1` passa nessa checagem.

1. Gerar a amostra:
   `uv run --no-sync python -m utils.generate_validation_sample --run-name <rodada> --final-test`
2. Anotar `artifacts/validation/human_validation_v1_<rodada>/annotation.csv`.
3. Pelo menos 24 horas depois da última edição dessa planilha, gerar a segunda rodada:
   `uv run --no-sync python -m utils.generate_validation_sample --stage repeat --run-name <rodada> --final-test`
4. Anotar `reannotation.csv`, na mesma pasta, sem consultar a primeira rodada.
5. Calcular o relatório:
   `uv run --no-sync python -m utils.precision_report --sample-dir artifacts/validation/human_validation_v1_<rodada> --final-test`

O relatório só é calculado com as duas planilhas completas, para que nenhum número de precisão seja
visto antes do fim da anotação. Não abra `annotation_key.json`, `reannotation_key.json` nem os
arquivos `*_report.json` antes disso: eles mostram como cada trecho foi escolhido.

## Arquivos

- `annotation.csv`: primeira rodada, aberta no Numbers ou no Excel.
- Transcrições completas: `artifacts/cache/validation/human_validation_v1_<rodada>/transcripts/<hearing_id>.txt`.
  Cada turno de fala começa com uma linha `[turno N | caracteres início–fim]` seguida do cabeçalho
  (`O SR. ...`, `A SRA. ...`). Para achar um trecho, busque as primeiras palavras dele no arquivo.
- `reannotation.csv`: segunda rodada, com 20 linhas da primeira em outra ordem e com outros
  identificadores.

## Colunas

| Coluna | Conteúdo |
| --- | --- |
| `item_id` | identificador da linha |
| `hearing_id` | número da audiência, que é também o nome do arquivo da transcrição |
| `pergunta` | `trecho_sustenta` ou `pessoa_falou`: qual pergunta responder e quais valores valem em `julgamento` |
| `participante`, `cargo` | a pessoa a quem a afirmação é atribuída |
| `afirmacao` | o que o conjunto de dados diz que a pessoa afirmou; em `pessoa_falou`, quando há mais de uma, elas vêm numeradas (`[1]`, `[2]`) |
| `trecho` | a passagem da transcrição a julgar (vazia em `pessoa_falou`) |
| `contexto_antes`, `contexto_depois` | até cerca de 350 caracteres do mesmo turno antes e depois do trecho; `[início do turno]` e `[fim do turno]` marcam os limites do turno e `…` indica que o texto continua. Em `pessoa_falou`, `contexto_antes` lista os cabeçalhos de fala detectados na audiência, com o número de turnos de cada um entre colchetes |
| `link` | posição do trecho: caracteres na transcrição original, número do turno (o mesmo das linhas `[turno N]` do arquivo) e cabeçalho do turno |
| `julgamento`, `existe_trecho_melhor`, `observacao` | preenchidas por você |

## Pergunta `trecho_sustenta`

O trecho, dito por esta pessoa, sustenta a afirmação atribuída a ela?

Julgue só a relação entre o trecho e a afirmação. Não julgue se a afirmação é verdadeira no mundo, se
está bem redigida nem se você concorda com ela. O contexto serve para entender o trecho (a que um
"isso" se refere, se há negação ou ironia).

- `correta`: o trecho, lido com o contexto imediato, sustenta o conteúdo principal da afirmação,
  inclusive a posição (a favor, contra) e os números ou nomes que ela cita.
  - Afirmação: "Defendeu que as creches municipais funcionem até as 19 horas." Trecho: "Precisamos
    que as creches fiquem abertas até as sete da noite, porque as mães que trabalham no comércio
    saem tarde."
  - Afirmação: "Criticou o atraso no repasse de verbas do transporte escolar." Trecho: "O dinheiro do
    transporte escolar chegou com quatro meses de atraso, e isso é inaceitável."
- `parcial`: o trecho sustenta só uma parte da afirmação, ou o mesmo assunto e a mesma posição sem o
  detalhe principal (número, prazo, destinatário), ou o apoio está nas frases vizinhas do contexto e
  não no trecho.
  - Afirmação: "Pediu a contratação de fiscais ambientais e a criação de um fundo de
    reflorestamento." Trecho: "Hoje temos um fiscal para cada 50 mil hectares; é preciso contratar
    mais gente." O trecho cobre os fiscais, não o fundo.
  - Afirmação: "Afirmou que o programa reduziu a evasão escolar em 30%." Trecho: "O programa ajudou
    muito a manter os alunos na escola." Mesma direção, sem o número.
- `incorreta`: o trecho não sustenta a afirmação: trata de outro assunto, tem a posição oposta, é só
  cumprimento ou procedimento da sessão, ou é fala de outra pessoa.
  - Afirmação: "Defendeu a privatização da companhia de saneamento." Trecho: "A companhia de
    saneamento tem de continuar pública, sob controle do Estado." Posição oposta.
  - Afirmação: "Cobrou a instalação de radares na rodovia." Trecho: "Agradeço ao presidente pelo
    convite e cumprimento os colegas da Mesa." Sem relação.

Confira o cabeçalho do turno na coluna `link`. `O SR. PRESIDENTE (Fulano de Tal. ...)` é fala de
Fulano de Tal. Se o cabeçalho for de outra pessoa, marque `incorreta` e escreva `outra pessoa` em
`observacao`.

### `existe_trecho_melhor`

Obrigatória em `trecho_sustenta`.

- `sim`: na fala desta mesma pessoa nesta audiência há uma passagem que sustenta a afirmação melhor
  que o trecho; copie as primeiras 10 palavras dela em `observacao`.
- `nao`: você procurou na fala da pessoa e não há passagem melhor.
- `nao_procurei`: você não procurou além do contexto mostrado. Use este valor, e não `nao`, sempre que
  não tiver procurado.

## Pergunta `pessoa_falou`

A transcrição registra palavras ditas por esta pessoa nesta audiência?

Estas linhas não têm trecho: o programa não encontrou fala utilizável da pessoa. Abra a transcrição
da audiência e procure pelo nome, pelos sobrenomes, pelo cargo e pela instituição. A lista de
cabeçalhos em `contexto_antes` ajuda, mas contém só os cabeçalhos que o programa reconheceu; um
cabeçalho escrito em formato diferente pode faltar nela, e é esse o tipo de caso que a pergunta
procura. A afirmação aparece só para ajudar a busca e não é julgada. Deixe `existe_trecho_melhor`
vazia.

- `falou`: há pelo menos um turno com palavras desta pessoa, mesmo que o nome esteja escrito de outro
  jeito (abreviado, sem um sobrenome, nome social, `PRESIDENTE` com o nome entre parênteses).
  - Participante "Carla Mendes Duarte, presidente da associação de moradores". Na transcrição:
    `A SRA. CARLA DUARTE - Bom dia a todos. Represento a associação...`. É a mesma pessoa, com um
    sobrenome a menos.
  - Participante "Rui Tavares, deputado". Na transcrição: `O SR. PRESIDENTE (Rui Tavares. PSD - MG) -
    Passo a palavra ao próximo orador.` Ele fala como presidente da sessão.
- `nao_falou`: a pessoa só é mencionada por outros, não aparece, ou o único registro dela é uma
  rubrica sem palavras; neste último caso, escreva `só rubrica` em `observacao`.
  - Participante "Otávio Lemos, secretário estadual". A transcrição só diz: "Registro a ausência do
    Sr. Otávio Lemos, que justificou a falta."
  - Participante "João Pinto, representante sindical". Único registro:
    `O SR. JOÃO PINTO - (Intervenção fora do microfone. Inaudível.)`.
- `nao_sei`: a transcrição não permite decidir; explique o motivo em `observacao`.
  - Participante "Ana Paula Souza". A transcrição tem `A SRA. ANA PAULA` e `A SRA. ANA SOUZA`, de
    instituições diferentes, e nada indica qual das duas é ela.
  - O presidente anuncia "Com a palavra a Sra. Márcia Reis", e o turno seguinte está marcado como
    `(Não identificado) -`.

## Como preencher

- Preencha só `julgamento`, `existe_trecho_melhor` e `observacao`. O relatório confere `item_id`,
  `hearing_id`, `pergunta`, `participante`, `afirmacao` e `trecho` e recusa a planilha se algum deles
  mudar.
- Não ordene nem filtre uma coluna isolada: isso troca os julgamentos de linha. Para ordenar,
  selecione a tabela inteira.
- Maiúsculas, acentos e espaços nos valores são aceitos (`Não falou` vale como `nao_falou`).
- Salve como CSV em UTF-8, com o mesmo nome de arquivo (no Excel, "CSV UTF-8"; no Numbers, exportar
  para CSV com codificação Unicode UTF-8). Ponto e vírgula, vírgula ou tabulação como separador
  funcionam.
- Depois que `reannotation.csv` for gerada, não edite mais `annotation.csv`; o relatório registra
  quando isso acontece.
