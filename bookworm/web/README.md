# Demonstração web

Esta pasta guarda a demonstração da verificação de atribuição: para cada afirmação que uma matéria
atribui a alguém, a página mostra onde procuramos essa fala na transcrição da audiência e o que
encontramos. O público é quem não conhece o projeto, então a página explica cada passo com cartões e
barbantes em vez de tabelas.

| Pasta | Conteúdo |
| --- | --- |
| [`app/`](app/) | A demonstração: lista das 206 matérias e a rede de barbantes de cada audiência. |
| [`mockups/`](mockups/) | Os protótipos usados para escolher a direção visual; a direção F virou `app/`. |

## Gerar os dados

A página não recalcula nada: ela lê arquivos JSON gravados pela biblioteca. O comando `export-site`
grava um arquivo por audiência (`hearings/<id>.json`, o mesmo formato de `export-hearing`) e um
`index.json` com o resumo que a lista de matérias usa. Ele lê os embeddings só do cache gravado por
`build-udvs`, então não carrega modelo. O comando roda a partir da pasta do projeto que tem
`configs/udv.toml`, porque os caminhos do TOML são relativos a ela. Neste repositório essa pasta é
`challenge/`, cujo ambiente não tem a biblioteca instalada; `--project ../bookworm` usa o ambiente de
`bookworm/` sem sair de `challenge/`:

```bash
cd challenge
uv run --project ../bookworm bookworm export-site --config configs/udv.toml --run-name udv_v1 \
    --split-manifest artifacts/splits/temporal_v1.json \
    --verifier-report artifacts/udv/udv_v1_verifier_report.json
```

`--verifier-report` é opcional. Com ele, cada afirmação ganha o bloco `signals` que a pasta da
afirmação mostra (a nota do verificador, as oito perguntas e a cópia em inglês que o verificador leu);
os valores são lidos dos arquivos que o relatório do verificador registra, conferidos pelo sha256, sem
carregar modelo. Esses arquivos, e o cache de tradução em `challenge/artifacts/cache/translation/`,
precisam estar no lugar; se faltar algum, se um sha256 não bater ou se os ids das UDVs não forem os da
execução, o comando para com código 2. Sem a opção, os arquivos gravados são os mesmos, byte a byte,
de antes dela existir.

`--profiles` é opcional e acrescenta os perfis de atores (ver [`docs/profiles.md`](../docs/profiles.md)).
Ele recebe o JSONL gravado por `profile-actors`; `--actors-config` aponta para a configuração que
ligou as falas aos atores (por padrão `configs/hearing_actors.toml`) e `--profiles-run` dá o nome da
rodada que a página mostra (por padrão, o nome do arquivo). Para os perfis de treino gerados com Qwen:

```bash
cd challenge
uv run --project ../bookworm bookworm export-site --config configs/udv.toml --run-name udv_v1 \
    --split-manifest artifacts/splits/temporal_v1.json \
    --verifier-report artifacts/udv/udv_v1_verifier_report.json \
    --profiles mlx_alternative/runs/qwen38_27b/actor_profiles/actor_profiles_train.jsonl \
    --profiles-run qwen38_27b --actors-config configs/hearing_actors.toml
```

Com a opção, a exportação grava também `actors.json` e um arquivo por ator em `profiles/<slug>.json`.
O comando recusa um arquivo de perfis que não bata com as falas da configuração: para cada perfil,
`n_statements` precisa ser a soma dos turnos da pessoa nas audiências de `hearing_ids`. Sem a opção,
uma exportação anterior com perfis tem `actors.json` e `profiles/` apagados, para a página não
mostrar perfis de outra execução.

Num projeto que tenha a biblioteca como dependência (`uv add --editable ../bookworm`), o mesmo comando
é `uv run bookworm export-site ...`. A exportação precisa do LDS em `challenge/dataset/`, da execução
em `challenge/artifacts/udv/` e do cache de embeddings em `challenge/artifacts/cache/embeddings/`; se
faltar um arquivo do cache, o comando para com código 2 e diz qual é.

Sem `--output`, os arquivos vão para `bookworm/web/app/data/` da árvore de código de onde a biblioteca
foi instalada; `--output` escolhe outro diretório, e `--overwrite` regrava uma exportação existente. A
pasta `app/data/` está no `.gitignore` porque os arquivos contêm as transcrições inteiras (cerca de
77 MB para `udv_v1`, ou 81 MB com `--verifier-report`). O formato dos arquivos está em
[`docs/data_model.md`](../docs/data_model.md#diretório-de-demonstração-export-site).

Antes de desenhar, a página confere cada arquivo contra esse formato: todos os campos que ela usa, com
os tipos e os valores de `tier` e `support_type` documentados, e a coerência entre eles. Cada
`actor.name` precisa estar em `people`; `person_not_resolved` precisa corresponder a
`resolved: false`; só `quote_found`, `semantic_match_high` e `semantic_match_weak` têm `evidence`, e só
`quote_found` tem `support_type` `direct_quote`; a evidência precisa estar dentro da transcrição e num
turno da própria pessoa; no índice, a soma de `tiers` precisa ser `n_udvs`. Quando a audiência traz `signals`, a página confere
também: o bloco do nível da audiência existe e tem perguntas com ids únicos; toda afirmação tem o seu
bloco; `scored` é verdadeiro exatamente nas afirmações com `evidence` e, nessas, os quatro campos estão
preenchidos (e nas outras, `null`); `supported` é igual a `probability >= threshold`; e cada valor das
perguntas está entre 0 e 1 para todos os ids de pergunta, nas duas leituras. Uma audiência que não está
em `index.json`, ou cujo bloco `run` difere do dele, é recusada, porque o arquivo pode ter sobrado de
outra exportação. As posições das frases candidatas podem ser `null`, como diz o formato; essas frases
aparecem na folha da busca, mas não acendem nenhum tracinho do caderno. A página também exige que
`transcript_chars` seja igual ao tamanho da transcrição medido pelo navegador. A biblioteca conta
posições em caracteres Unicode, e o navegador em unidades UTF-16; as duas contagens só coincidem
quando a transcrição não tem caracteres fora do plano básico (como emojis), e é isso que garante que
cada posição aponta para o texto certo. Quando uma conferência falha, a página mostra o arquivo e o
campo em vez de desenhar a audiência com valores supostos.

## Servir a página

A página é estática: HTML, CSS e módulos JavaScript, sem etapa de build e sem `npm`. Qualquer servidor
de arquivos serve; com Python:

```bash
cd bookworm/web/app
python -m http.server 8000
```

Depois, abra `http://localhost:8000/`. Cada audiência tem um endereço próprio, `#h` seguido do número
(`http://localhost:8000/#h70`), que pode ser compartilhado, e cada afirmação tem a sua pasta em `#h`
seguido do número da audiência, `-u` e o número da afirmação (`#h70-u1`). Abrir `index.html` direto do disco não
funciona, porque o navegador bloqueia a leitura de `data/` por `file://`; nesse caso, e quando os dados
ainda não foram gerados, a página mostra o erro e o comando que falta.

Os outros endereços são `#arquivo` (o mapa do caso), `#perfil/<slug>` (o perfil de um ator, como
`#perfil/vanessa-negrini`) e `#h<id>-t<turno>-p<início>-<fim>`, que abre o leitor da transcrição na
frase indicada; é para esse endereço que a evidência de um perfil aponta.

As bibliotecas vêm do cdnjs com versão fixa e verificação de integridade (`d3` 7.9.0 e `gsap` 3.12.5), e
as fontes vêm do Google Fonts. Sem acesso a elas a página continua funcionando, sem as animações e com
as fontes de reserva do sistema.

## O que a página mostra

### Resumo primeiro, análise completa depois

Toda página abre com um resumo curto, pensado para ser entendido em poucos segundos, e guarda o resto
atrás de um único botão, "Ver análise completa" (na lista de matérias, "Ver todas as matérias"). O
botão é um `<button>` com `aria-expanded`; aberto, ele vira "Recolher", e um segundo "Recolher" aparece
no fim da parte aberta. Abrir mostra toda a análise de uma vez, sem outros blocos para abrir dentro
dela. Cada tipo de página lembra, no `localStorage` do navegador, se foi deixado aberto ou fechado
(chaves `bookworm.expand.<página>`); quando o navegador não deixa gravar, a página abre fechada e
funciona igual.

| Página | Resumo | Análise completa |
| --- | --- | --- |
| Lista de matérias | Busca, ordem, sorteio, a legenda dos quadradinhos e as 12 primeiras matérias da ordem e da busca atuais. | As outras matérias, na mesma ordem. |
| Mural da audiência | Manchete, os números (afirmações, pessoas, quantas em cada resultado), os perfis de quem fala, o aviso curto e a parede com a história. | Como ler os fios, o aviso completo, os atalhos de teclado, "Pergunte à audiência" e a lista de todas as afirmações, cada uma com link para a sua pasta. |
| Pasta da afirmação | Uma linha de veredito com as duas perguntas separadas ("Trecho encontrado por citação direta · verificador: sustenta (0,91)"), o cartão da afirmação, só a frase escolhida em amarelo e dois medidores: "Onde está o trecho?" (resultado da busca e cosseno com o corte) e "O trecho sustenta a afirmação?" (nota do verificador com o corte). | As fichas de todas as afirmações, o aviso sobre as notas e a pasta inteira descrita abaixo. |
| Perfil | Nome, cargo, duas frases de síntese calculadas a partir dos dados, os números de participação, o selo de texto gerado por modelo e três posições, cada uma com um selo da evidência. | O dossiê inteiro descrito abaixo. |
| Mapa do caso | Os números da exportação, a busca, os quatro atores com mais afirmações nas matérias e as três audiências mais recentes; com uma busca, os primeiros resultados. | Todos os perfis e todas as audiências. |
| Leitor da transcrição | Só a frase marcada, com o turno e quem fala. | As frases antes e depois e os botões "Ler mais antes" e "Ler mais depois". |

Um link que aponta para algo dentro da parte fechada abre essa parte antes de rolar até o alvo: é o
caso dos números dos itens e dos selos das posições no perfil e do endereço `#perfil/<slug>/item-<n>`,
que abre o perfil já na evidência do item `n`.

A síntese do perfil não é texto do modelo: ela junta o período das audiências, quantos itens o modelo
escreveu e quantos deles têm frase parecida ou afirmação de matéria ligada. As três posições são as
três primeiras da seção "Posições" (ou da primeira seção, quando ela não existe), pondo antes as que
têm afirmação de matéria ligada e depois as que têm frase parecida, na ordem em que o modelo as
escreveu. Essa ordem mostra primeiro o que tem evidência; não diz quais posições são mais importantes
para a pessoa.

**Lista de matérias.** Cada matéria aparece como um recorte com a data, a manchete, quantas afirmações
ela atribui a quantas pessoas e um quadradinho por afirmação, agrupado pelo resultado. A busca procura
palavras no título, no assunto e nos nomes de quem participou, sem diferenciar acentos. A lista pode
ser ordenada pela data da matéria, e o botão "Sortear uma matéria" abre uma das que estão na lista.

**Rede de barbantes.** A audiência vira um mural de cortiça:

| Objeto | O que é |
| --- | --- |
| A matéria | O texto da matéria, com as palavras que coincidem com cada afirmação sublinhadas. |
| Cartão de afirmação | Um registro UDV: a afirmação, a quem a matéria a atribui e, depois da verificação, o carimbo com o resultado. |
| Etiqueta de quem disse | O participante, com o cargo e em quantos turnos fala. |
| Caderno de falas | Todas as frases que a pessoa diz na audiência, em ordem, uma por tracinho (em audiências longas, cada tracinho reúne várias frases, e o caderno diz quantas). |
| Trecho | A frase escolhida como evidência, com o turno e a posição na transcrição, e o botão "Ler em volta". |

Os barbantes são as ligações: a matéria publica a afirmação, a afirmação é atribuída a alguém, a pessoa
fala no caderno, o trecho está num ponto do caderno e a afirmação é ligada ao trecho com o estilo do
resultado. Quem não foi achado entre os falantes fica com fios soltos.

A história percorre cada afirmação em até cinco cenas (a matéria, quem disse, a busca, o trecho e o
resultado), com os botões Próximo e Voltar, as setas do teclado ou "Tocar sozinho". "Ver a rede
inteira" mostra todas as ligações de uma vez; ali, tocar num cartão mostra só ele e seus fios, e tocar
num fio segue o barbante até a outra ponta. Os botões + e −, as teclas + e −, o gesto de pinça e
Ctrl ou Cmd com a roda do mouse aproximam e afastam a parede; na rede inteira, arrastar a parede mostra
outras partes.

"Pergunte à audiência" procura palavras nas frases da transcrição (BM25, sem acentos e sem palavras
muito comuns) e mostra as cinco frases que mais combinam, com a posição de cada uma na audiência. Nada é
gerado: todo texto mostrado foi dito na audiência.

**Pasta da afirmação.** Uma página por afirmação. Ela é aberta pelo botão "Abrir a pasta desta afirmação" na legenda da
parede, que aparece no exemplo pronto, na cena do resultado de cada afirmação e, na rede inteira,
quando um cartão de afirmação é puxado. Dentro dela há "Voltar à audiência", os botões de afirmação
anterior e próxima e uma ficha por afirmação, com o número, o nome e a cor do resultado. Voltar à
audiência reabre a história na cena do resultado da afirmação que estava aberta; entrar em `#h70` por
outro caminho abre a história do começo, como antes. Na análise completa, a pasta tem quatro partes,
lidas da esquerda para a direita em telas largas e uma embaixo da outra no celular:

| Parte | O que mostra |
| --- | --- |
| O que a matéria diz | O cartão da afirmação, com a pessoa, o cargo e o carimbo. Quando há aspas, a citação inteira aparece sublinhada, com as palavras achadas na fala marcadas, e uma linha diz que só o começo das aspas foi procurado. |
| O que a pessoa disse | A frase escolhida em amarelo, no meio de até duas frases antes e duas depois do mesmo turno (até 600 caracteres de cada lado), com o turno e a posição na transcrição. Embaixo, menor, a cópia em inglês que o verificador leu, marcada como tradução automática, com o nome do modelo. |
| O que as medidas dizem | A régua do cosseno com o corte da execução, a régua do verificador com o corte escolhido no treino, uma frase dizendo se as duas medidas ficam do mesmo lado dos seus cortes, a tabela das oito perguntas (a pergunta em pt-BR, o valor na leitura em português e na cópia em inglês) e a linha do outro modelo de NLI. Uma linha da tabela fica marcada em vermelho quando as duas leituras diferem em 0,30 ou mais. |
| Outras frases parecidas que a pessoa disse | As frases candidatas, em ordem de nota, com uma barra por nota e o corte marcado; a escolhida fica destacada, e a última linha diz a distância da segunda para a primeira (ou, em `quote_found`, em que posição a frase escolhida pelas aspas ficaria pelo sentido). |

Uma afirmação sem trecho (`no_evidence`, `person_not_resolved`) mostra só o cartão e a explicação do
motivo, com o mesmo texto da legenda da parede. Uma audiência exportada sem `--verifier-report` mostra
a pasta sem a parte das medidas e com uma linha dizendo que a segunda opinião, a do verificador, não
foi calculada. Um aviso curto no topo repete que ninguém conferiu os resultados e que nenhuma nota é a
chance de a afirmação estar certa.

### Como os resultados são descritos

Os nomes dizem o que o procedimento fez, e a página avisa, na lista e em cada audiência, que os
resultados ainda não foram conferidos por uma pessoa.

| `tier` | Carimbo no cartão | Na lista de matérias |
| --- | --- | --- |
| `quote_found` | Achamos o começo das aspas na fala | com o começo das aspas achado na fala |
| `semantic_match_high` | Achamos um trecho parecido, com a nota | com um trecho parecido |
| `semantic_match_weak` | Só achamos algo pouco parecido, com a nota | só com algo pouco parecido |
| `no_evidence` | Não havia frase para comparar | sem trecho achado |
| `person_not_resolved` | Não achamos a pessoa entre quem fala | sem trecho achado |

Em `quote_found`, o que foi achado igual na fala é o começo da citação (6 ou mais palavras), não a
afirmação inteira, e o cartão diz quantas palavras das aspas coincidem. A nota é a similaridade de
cosseno entre a afirmação e a frase, de 0 a 1, com o corte da execução (`run.threshold`) marcado na
régua do cartão. A página a apresenta como medida de semelhança e diz que ela não é a chance de a
afirmação estar certa. A nota aparece com duas casas decimais; quando o arredondamento a colocaria do
outro lado do corte (0,448 viraria 0,45 com corte 0,45), ela ganha uma ou duas casas a mais, para que
o número mostrado nunca contradiga o resultado. Uma pessoa que não foi achada entre os falantes pode não
ter falado ou aparecer com outro nome na transcrição, e a legenda diz isso. Os turnos são numerados a
partir de 1 em toda a página.

## Mapa do caso e perfis

**Mapa do caso.** Uma aba fixa na borda direita (no celular, um botão no canto de baixo) e a tecla `M`
abrem o mapa de qualquer página; `M` de novo volta para onde se estava, e `/` põe o cursor na busca.
O mapa lista os perfis em ordem alfabética, cada um com uma barra que divide os itens do perfil em
três partes (com UDV, só com frase parecida, sem frase), e as audiências da mais recente para a mais
antiga, com links para o mural, para as pastas e para o perfil de cada pessoa citada que tem perfil.
A busca procura por nome, cargo, título ou número da audiência, sem acento e sem diferença entre
maiúsculas e minúsculas. No topo de todas as páginas, uma trilha (`Mapa do caso › Audiência 167 ›
VANESSA NEGRINI › Perfil`) mostra o caminho e volta a qualquer ponto dele.

**Perfil.** O perfil de um ator é montado como um dossiê: uma ficha com o monograma no lugar da foto
(a página não busca imagens), a tabela de dados (cargo, audiências, turnos, itens), a proveniência
(modelo, rodada, versão do prompt, data, tokens e sha256 do arquivo de perfis), um cartão que explica
como o perfil foi feito, os cartões de cada seção do texto e a folha "Evidência de cada item". Um aviso
no topo diz que o texto foi escrito por um modelo de linguagem a partir das falas e que a evidência
precisa ser conferida. Cada item do texto tem um número que leva à sua evidência; a evidência é a
frase das falas da pessoa mais parecida com o item, com link para o leitor da transcrição, e, quando
existe, a afirmação da matéria (UDV) ligada a ele, com o nível do resultado, a nota do verificador e o
link para a pasta. As notas adesivas só mostram números calculados na exportação (por exemplo, quantos
itens têm UDV). Tudo isso fica na análise completa; o resumo do topo está descrito em "Resumo primeiro,
análise completa depois".

**Ligações.** A pasta de uma afirmação tem o link "Ver o perfil de ..." quando a pessoa tem perfil, e o
cabeçalho do mural lista os perfis de quem fala na audiência. Do perfil, cada audiência leva ao mural,
cada frase ao leitor da transcrição e cada UDV à sua pasta.

## Como a parede se organiza

As audiências vão de 3 a 31 afirmações, de 2 a 12 participantes e de 7 a 4.898 turnos. Os protótipos
tinham uma página por turno e posições ajustadas à audiência 70, o que não cabe nas audiências maiores.
A página faz a disposição a partir dos dados de cada audiência:

- cada participante forma um grupo em linha, com três colunas: os cartões das afirmações, a etiqueta da
  pessoa com os trechos e o caderno de falas. O caderno substitui as páginas por turno, então o tamanho
  do grupo depende do número de afirmações e trechos, e não do tamanho da transcrição;
- a altura de cada cartão é medida no próprio texto, então afirmações longas aumentam o cartão em vez de
  serem cortadas; os grupos ficam empilhados sem sobreposição;
- os barbantes só ligam colunas vizinhas e passam pelo espaço entre elas, então não atravessam
  cartões; dentro de um grupo, a ordem dos cartões e dos trechos é escolhida para reduzir os cruzamentos
  entre fios (todas as ordens são testadas nos grupos pequenos; nos maiores, as afirmações seguem a
  ordem da matéria e cada trecho fica perto das afirmações que liga);
- em telas largas os grupos se dividem dos dois lados da matéria, equilibrando as alturas; quando a
  parede tem menos de 760 px de largura ou não é pelo menos 20% mais larga do que alta (como num celular
  de 400 px), ficam todos de um lado, e a parede passa a ser explorada por aproximação e arrasto, sem
  rolagem horizontal da página;
- de longe, cada objeto mostra só um rótulo (número da afirmação, nome, "trecho"); de perto, o
  conteúdo inteiro. Nomes, cargos e títulos dos cadernos não são cortados: o cartão cresce para
  caber o texto. Os textos encurtados de propósito são o trecho (até 560 caracteres em volta das
  palavras iguais, com [...] e o botão "Ler em volta"), as frases candidatas da folha da busca (uma
  linha cada) e, no celular, as aspas e a frase da folha da busca (até 200 e 240 caracteres em volta
  das palavras iguais);
- cada cena termina no objeto principal dela em tamanho de leitura: o cartão na matéria e no
  resultado, a etiqueta da pessoa em quem disse, a folha na busca e o trecho no trecho. Nas cenas da
  matéria e de quem disse, quando os objetos ligados não cabem juntos em tamanho de leitura, como no
  celular, a animação mostra o conjunto de longe enquanto o fio é amarrado e depois se aproxima do
  objeto principal. Um cartão mais alto que a parede aparece a partir do começo (no resultado, a partir
  do fim, onde fica o carimbo);
- na rede inteira, o cartão puxado é aumentado até dar para ler, mesmo quando os vizinhos estão longe;
- os botões de aproximar e afastar ficam numa coluna à direita (telas largas) ou numa fileira no pé da
  parede (celular), e os enquadramentos deixam essa faixa livre, para que os botões não cubram o texto.

A página respeita `prefers-reduced-motion` (sem animações, com as mesmas cenas), mostra o foco do
teclado em todos os controles e mostra estados de carregamento e de erro.

## Estrutura de `app/`

| Arquivo | Papel |
| --- | --- |
| `index.html` | Estrutura da página, fontes e bibliotecas. |
| `css/app.css` | Cores, tipografia, lista de matérias e estados. |
| `css/wall.css` | A parede: cartões, barbantes, câmera, legenda, busca. |
| `css/case.css` | A pasta da afirmação. |
| `css/profile.css` | O dossiê do perfil. |
| `css/nav.css` | Trilha, aba do mapa, links para os perfis e o mapa do caso. |
| `css/disclose.css` | O botão "Ver análise completa" e os quadros de números dos resumos. |
| `js/main.js` | Rotas (`#h<id>`, `#h<id>-u<n>`, `#h<id>-t<turno>-p<início>-<fim>`, `#perfil/<slug>`, `#perfil/<slug>/item-<n>`, `#arquivo`), trilha, atalhos de teclado, carregamento, volta da pasta à cena do resultado e estados de erro. |
| `js/data.js` | Leitura de `data/index.json`, `data/hearings/<id>.json`, `data/actors.json` e `data/profiles/<slug>.json` e conferência do formato. |
| `js/home.js` | Lista de matérias: busca, ordem, sorteio. |
| `js/model.js` | Deriva do JSON da audiência os cartões, grupos, trechos, cadernos e onde cada afirmação aparece na matéria. |
| `js/copy.js` | Os textos dos resultados, do aviso de conferência, das oito perguntas e dos resumos. |
| `js/disclose.js` | O botão de abrir e recolher, a memória do estado no `localStorage` e a abertura por link. |
| `js/case.js` | A pasta da afirmação. |
| `js/profile.js` | O perfil do ator. |
| `js/atlas.js` | O mapa do caso e a busca por ator. |
| `js/text.js` | Formatação em pt-BR e comparação de palavras. |
| `js/wall/wall.js` | Monta a parede e liga os eventos. |
| `js/wall/objects.js`, `layout.js`, `strings.js` | Cartões, disposição e barbantes. |
| `js/wall/camera.js`, `motion.js` | Câmera, aproximação, arrasto e o balanço dos fios. |
| `js/wall/scene.js`, `story.js` | O que aparece em cada cena, a história, a rede inteira e as legendas. |
| `js/wall/ask.js`, `legend.js`, `markup.js` | A busca na audiência, o leitor da transcrição, a legenda e o HTML da parede. |

## Limitações conhecidas

- Os níveis de resultado não foram validados por anotação humana; a página diz isso, mas não mostra
  nenhuma medida de acerto, porque ela ainda não existe.
- Nas audiências com mais afirmações, a visão da rede inteira fica pequena e alguns rótulos de longe se
  encostam; aproximar resolve.
- No celular, cada cena termina num objeto só; para ver as ligações com os vizinhos ao mesmo tempo, é
  preciso afastar a parede com o botão de menos.
- Um cartão ou caderno mais alto que a parede aparece em parte; na história, o celular não arrasta a
  parede (o arrasto fica para a rolagem da página), então o resto se vê afastando a parede ou na rede
  inteira.
- As frases candidatas da folha da busca aparecem numa linha cada, cortadas com reticências; a frase
  escolhida aparece inteira no trecho.
- A busca da seção "Pergunte à audiência" é por palavras, não por sentido.
- O botão da pasta fica na legenda, não nos cartões: na rede inteira, ele só aparece depois que um
  cartão de afirmação é puxado.
- A cópia em inglês é a que o verificador leu, gravada no cache de tradução; a página não traduz nada,
  e um erro de tradução muda juntas todas as respostas da leitura em inglês.
- O limite de 0,30 que marca uma pergunta em que as duas leituras discordam foi escolhido para a
  leitura da página, não medido.
- A ligação entre um item do perfil e a fala é lexical (TF-IDF das palavras), escolhida frase a frase;
  uma frase parecida pode não sustentar o item, e o item pode resumir várias falas. Os limites de 0,15
  (frase) e 0,30 (UDV por texto parecido) foram escolhidos para a leitura da página, não medidos.
- Na rodada `qwen38_27b` de treino, só 144 dos 4.928 itens dos 264 perfis se ligam a uma UDV; 4.348 têm
  só uma frase parecida e 436 não têm nenhuma. A maior parte do texto dos perfis se confere lendo a
  transcrição, sem uma afirmação da matéria ao lado.
- Os perfis só cobrem o split de treino e os atores que falam em mais de uma audiência; a simulação de
  atores não aparece na página.
- O estado aberto ou fechado é lembrado por tipo de página, não por endereço: quem abre a análise de uma
  pasta encontra as próximas pastas abertas.
- A legenda da parede não tem link para os perfis; eles aparecem no cabeçalho do mural e nas pastas.
