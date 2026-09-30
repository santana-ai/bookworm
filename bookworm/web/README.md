# Demonstração web

Esta pasta guarda a demonstração da verificação de atribuição: para cada afirmação que uma matéria
atribui a alguém, a página mostra onde procuramos essa fala na transcrição da audiência e o que
encontramos. O público é quem não conhece o projeto, então a página explica cada passo com cartões e
barbantes em vez de tabelas.

A demonstração fica em [`app/`](app/): a lista das 206 matérias e a rede de barbantes de cada
audiência. Os protótipos visuais usados para escolher essa direção ficaram só no histórico do Git
(pasta `bookworm/web/mockups/`, removida na versão 1.0.0).

## Início rápido

Os dados já gerados estão versionados em `app/data/`, então para só ver a demo basta:

```bash
cd bookworm/web/app && python3 -m http.server 8000
```

e abrir `http://localhost:8000/`. Os comandos abaixo regeram esses dados a partir da rodada.

Com a rodada `udv_v2` e o cache de embeddings no lugar (ver [guia de reprodução](../../docs/reproduce.md)):

```bash
cd experiments
uv run bookworm export-site --config configs/udv_v2.toml --run-name udv_v2 \
    --split-manifest artifacts/splits/temporal_v1.json \
    --verifier-report artifacts/udv/udv_v2_verifier_report.json \
    --profiles artifacts/mlx_runs/qwen38_27b/actor_profiles/actor_profiles_train.jsonl \
    --profiles-run qwen38_27b --actors-config configs/hearing_actors.toml \
    --human-validation artifacts/udv/udv_v2_precision_final.json
cd ../bookworm/web/app && python3 -m http.server 8000
```

Os arquivos de notas do verificador são grandes e ficam fora do Git; se faltarem, `experiments/README.md`
explica como restaurá-los da etiqueta `research-2026-09-28`.

Depois, abra `http://localhost:8000/`. As seções abaixo explicam cada opção, as conferências que a
página faz antes de desenhar e o que ela mostra.

## Gerar os dados

A página não recalcula nada: ela lê arquivos JSON gravados pela biblioteca. O comando `export-site`
grava um arquivo por audiência (`hearings/<id>.json`, o mesmo formato de `export-hearing`) e um
`index.json` com o resumo que a lista de matérias usa. Ele lê os embeddings só do cache gravado por
`build-udvs`, então não carrega modelo. O comando roda a partir da pasta do projeto que tem
`configs/udv_v2.toml`, porque os caminhos do TOML são relativos a ela. Neste repositório essa pasta é
`experiments/`, cujo ambiente instala a biblioteca como dependência editável:

```bash
cd experiments
uv run bookworm export-site --config configs/udv_v2.toml --run-name udv_v2 \
    --split-manifest artifacts/splits/temporal_v1.json \
    --verifier-report artifacts/udv/udv_v2_verifier_report.json
```

`udv_v2` é a execução final; `udv_v1` continua exportável com `--config configs/udv.toml --run-name
udv_v1` e o relatório `udv_v1_verifier_report.json`. O comando confere a execução contra a configuração
passada em `--config`: a seção `pipeline` do arquivo de cobertura precisa ser a que essa configuração
descreve (unidade candidata, extensão da citação e padrões de aspas) e o corte de cosseno precisa ser o
dela. Com a configuração de outra execução, o comando para com código 2 e diz em que campos elas
diferem.

`--verifier-report` é opcional. Com ele, cada afirmação ganha o bloco `signals` que a pasta da
afirmação mostra (a nota do verificador, as oito perguntas e a cópia em inglês que o verificador leu);
os valores são lidos dos arquivos que o relatório do verificador registra, conferidos pelo sha256, sem
carregar modelo. Esses arquivos, e o cache de tradução em `experiments/artifacts/cache/translation/`,
precisam estar no lugar; se faltar algum, se um sha256 não bater ou se os ids das UDVs não forem os da
execução, o comando para com código 2. Sem a opção, os arquivos gravados são os mesmos, byte a byte,
de antes dela existir.

`--profiles` é opcional e acrescenta os perfis de atores (ver [`docs/profiles.md`](../docs/profiles.md)).
Ele recebe o JSONL gravado por `generate-profiles`; `--actors-config` aponta para a configuração que
ligou as falas aos atores (por padrão `configs/hearing_actors.toml`) e `--profiles-run` dá o nome da
rodada que a página mostra (por padrão, o nome do arquivo). Para os perfis de treino gerados com Qwen:

```bash
cd experiments
uv run bookworm export-site --config configs/udv_v2.toml --run-name udv_v2 \
    --split-manifest artifacts/splits/temporal_v1.json \
    --verifier-report artifacts/udv/udv_v2_verifier_report.json \
    --profiles artifacts/mlx_runs/qwen38_27b/actor_profiles/actor_profiles_train.jsonl \
    --profiles-run qwen38_27b --actors-config configs/hearing_actors.toml
```

Com a opção, a exportação grava também `actors.json` e um arquivo por ator em `profiles/<slug>.json`.
O comando recusa um arquivo de perfis que não bata com as falas da configuração: para cada perfil,
`n_statements` precisa ser a soma dos turnos da pessoa nas audiências de `hearing_ids`. Sem a opção,
uma exportação anterior com perfis tem `actors.json` e `profiles/` apagados, para a página não
mostrar perfis de outra execução.

`--human-validation` é opcional e recebe o relatório final da conferência humana
(`artifacts/udv/udv_v2_precision_final.json`). Com ele, `index.json` ganha o bloco `validation`: por
resultado da busca, quantas afirmações a pessoa que conferiu julgou corretas, parciais e incorretas,
com os intervalos de Wilson; os dois critérios fixados antes da conferência e se foram atingidos; se foi
uma pessoa só; quantos julgamentos foram herdados da primeira versão da busca e a suposição por trás
dessa herança; a checagem de quem falou nas afirmações sem trecho; o julgamento de cada afirmação
conferida; e, com `--verifier-report`, as mesmas contagens por faixa de apoio do verificador (ver
"Apoio do verificador" abaixo). As contagens por faixa são uma junção, feita na exportação, dos
julgamentos com as notas que o verificador já tinha gravado; nenhum modelo roda. O comando para com
código 2 quando o relatório não bate com a execução: uma afirmação conferida que não existe nela, um
julgamento fora das opções da pergunta, uma afirmação julgada duas vezes ou contagens por resultado
diferentes das que o próprio relatório registra. Sem a opção, a página diz que ainda não houve
conferência humana.

A exportação precisa do LDS em `experiments/dataset/`, da execução
em `experiments/artifacts/udv/` e do cache de embeddings em `experiments/artifacts/cache/embeddings/`; se
faltar um arquivo do cache, o comando para com código 2 e diz qual é.

Sem `--output`, os arquivos vão para `bookworm/web/app/data/` da árvore de código de onde a biblioteca
foi instalada; `--output` escolhe outro diretório, e `--overwrite` regrava uma exportação existente. A
pasta `app/data/` é versionada, com a exportação de `udv_v2`, para que a demo rode sem o dataset e sem
modelo; os arquivos contêm as transcrições inteiras do PublicHearingBR: 58.810.449
bytes para `udv_v2` com `--verifier-report`, os perfis e a conferência humana, segundo o resumo gravado em
[`experiments/artifacts/web/export_site_udv_v2.json`](../../experiments/artifacts/web/export_site_udv_v2.json). O formato dos arquivos está em
[`docs/data_model.md`](../docs/data_model.md#diretório-de-demonstração-export-site).

### Tamanho dos arquivos

Cada transcrição é gravada uma vez só. As frases de cada turno guardam só as posições de início e fim
(`[início, fim]`), e o navegador recorta o texto da transcrição, trocando cada sequência de espaços por
um espaço; o texto só vai junto (`[início, fim, texto]`) quando o recorte não devolve a frase gravada,
e `[null, null, texto]` marca uma frase sem posição. Antes, cada frase levava o texto inteiro, e a
transcrição ia duas vezes em cada arquivo. Medido nesta exportação de `udv_v2` (verificador, perfis e,
depois, conferência humana), com `gzip -6` para o tamanho comprimido:

| | Antes | Depois |
| --- | --- | --- |
| Arquivos de audiência, mediana | 362.001,5 bytes | 229.722 bytes |
| Arquivos de audiência, maior (audiência 6) | 3.439.667 bytes | 2.450.736 bytes |
| Arquivos de audiência, soma | 83.712.499 bytes | 54.022.444 bytes |
| Exportação inteira | 88.438.867 bytes | 58.810.449 bytes |
| Audiência 6, comprimida | 695.563 bytes | 420.366 bytes |
| Arquivos de audiência comprimidos, soma | 21.545.734 bytes | 13.300.437 bytes |

Com a rede limitada a 1,6 Mbit/s e 150 ms de latência e a CPU 4 vezes mais lenta (Chrome, 390 x 844,
servidor sem compressão), o tempo até a página sair do estado de carregamento foi, antes e depois:
`#h6` 22.187 e 17.737 ms; `#h70` 6.878 e 6.634 ms; lista de matérias 4.218 e 4.425 ms; perfil de
Erika Kokay 5.620 e 5.866 ms. Os tempos de depois são a mediana de três medições, que variaram menos de
20 ms; os de antes são de uma medição. A lista e o perfil ficaram mais lentos porque `index.json`
ganhou o bloco `validation` (de 184.172 para 201.313 bytes) e `actors.json` os nomes da matéria e os
cortes do verificador (de 157.029 para 167.969 bytes).

### Conferências antes de desenhar

A página confere cada arquivo contra esse formato antes de desenhar, e, quando uma conferência falha,
mostra o arquivo e o campo em vez de desenhar a audiência com valores supostos. Ela exige:

- todos os campos que usa, com os tipos e os valores de `tier` e `support_type` documentados;
- cada `actor.name` em `people`, e `person_not_resolved` correspondendo a `resolved: false`;
- `evidence` só em `quote_found`, `semantic_match_high` e `semantic_match_weak`, e `support_type`
  `direct_quote` só em `quote_found`;
- a evidência dentro da transcrição e num turno da própria pessoa;
- no índice, a soma de `tiers` igual a `n_udvs`;
- com `signals`: o bloco do nível da audiência com perguntas de ids únicos; um bloco por afirmação;
  `scored` verdadeiro exatamente nas afirmações com `evidence`, e nessas os quatro campos preenchidos
  (nas outras, `null`); `supported` igual a `probability >= threshold`; cada valor das perguntas entre 0
  e 1, para todos os ids, nas duas leituras;
- a audiência presente em `index.json`, com o mesmo bloco `run`, porque um arquivo fora do índice pode
  ter sobrado de outra exportação;
- `transcript_chars` igual ao tamanho da transcrição medido pelo navegador. A biblioteca conta posições
  em caracteres Unicode e o navegador em unidades UTF-16; as duas contagens só coincidem quando a
  transcrição não tem caracteres fora do plano básico (como emojis), e é isso que garante que cada
  posição aponta para o texto certo.

- cada frase de um turno com as posições dentro desse turno;
- com `validation`: resultados conhecidos, `correta + parcial + incorreta` igual a `udvs` em cada
  resultado, dois cortes em `verifier_bands` e, para cada afirmação conferida, uma pergunta e um
  julgamento;
- `similar_speakers`, quando existe, com nome, turnos e semelhança; `verifier_cuts` dos perfis com o
  corte alto acima do baixo.

As posições das frases candidatas podem ser `null`, como diz o formato; essas frases aparecem na folha
da busca, mas não acendem nenhum tracinho do caderno.

## Servir a página

A página é estática: HTML, CSS e módulos JavaScript, sem etapa de build e sem `npm`. Qualquer servidor
de arquivos serve; com Python:

```bash
cd bookworm/web/app
python3 -m http.server 8000
```

O `http.server` do Python não comprime as respostas. Os arquivos JSON comprimem para cerca de um quarto
do tamanho, então um servidor público deve ligar a compressão; por exemplo, no nginx:

```nginx
gzip on;
gzip_types application/json text/css application/javascript;
```

ou, no Caddy, `encode gzip` no bloco do site. Para testar localmente com compressão,
`npx http-server bookworm/web/app -g` serve os arquivos `.gz` quando existem ao lado dos originais.

Depois, abra `http://localhost:8000/`. Cada audiência tem um endereço próprio, `#h` seguido do número
(`http://localhost:8000/#h70`), que pode ser compartilhado, e cada afirmação tem a sua pasta em `#h`
seguido do número da audiência, `-u` e o número da afirmação (`#h70-u1`). Abrir `index.html` direto do disco não
funciona, porque o navegador bloqueia a leitura de `data/` por `file://`; nesse caso, e quando os dados
ainda não foram gerados, a página mostra o erro e o comando que falta.

Os outros endereços são `#arquivo` (o mapa do caso), `#perfil/<slug>` (o perfil de um ator, como
`#perfil/vanessa-negrini`) e `#h<id>-t<turno>-p<início>-<fim>`, que abre o leitor da transcrição na
frase indicada; é para esse endereço que a evidência de um perfil aponta. O turno do endereço é o
mesmo número que a página mostra, contado a partir de 1 (`#h70-t8-p…` abre o turno 8). Quando as
posições não ficam dentro desse turno, a página não abre o leitor: diz quais posições e qual turno o
endereço pede, com links para a audiência e, se se veio de um perfil, de volta a ele.

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
| Mural da audiência | Manchete, os números (afirmações, pessoas, quantas em cada resultado), os perfis de quem fala, o resumo da conferência humana e a parede com a história. | Como ler os fios, o aviso completo, os atalhos de teclado, "Pergunte à audiência" e a lista de todas as afirmações, com o resultado da busca, a semelhança e o apoio de cada uma, em ordem da matéria ou com "Menos apoio primeiro". |
| Pasta da afirmação | Uma linha de veredito com as duas perguntas separadas ("Trecho parecido pelo sentido · apoio incerto 0,29"), o cartão da afirmação, só o trecho escolhido em amarelo, o julgamento da conferência humana quando a afirmação foi conferida e dois medidores: "Onde está o trecho?" (resultado da busca e semelhança com o corte) e "Quanto o trecho apoia a afirmação?" (apoio do verificador nas três faixas e quantos trechos dessa faixa a conferência humana julgou corretos). | As fichas de todas as afirmações, o texto completo da conferência humana, um glossário curto e a pasta inteira descrita abaixo. |
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
| Cartão de afirmação | Uma afirmação da matéria, a quem a matéria a atribui e, depois da verificação, o carimbo com o resultado, a régua da semelhança e o selo de apoio do verificador. |
| Etiqueta de quem disse | O participante, com o cargo e em quantos turnos fala. |
| Caderno de falas | Todas as frases que a pessoa diz na audiência, em ordem, uma por tracinho (em audiências longas, cada tracinho reúne várias frases, e o caderno diz quantas). |
| Trecho | A frase escolhida como evidência, com o turno e o nome inteiro de quem fala, e o botão "Ler em volta", que abre o leitor já com as frases de antes e de depois. |

Os barbantes são as ligações: a matéria publica a afirmação, a afirmação é atribuída a alguém, a pessoa
fala no caderno, o trecho está num ponto do caderno e a afirmação é ligada ao trecho com o estilo do
resultado. Quem não foi achado entre os falantes fica com fios soltos.

A história percorre cada afirmação em até cinco cenas (a matéria, quem disse, a busca, o trecho e o
resultado), com os botões Próximo e Voltar, as setas do teclado ou "Tocar sozinho". "Ver a rede
inteira" mostra todas as ligações de uma vez; ali, tocar num cartão mostra só ele e seus fios, e tocar
num fio segue o barbante até a outra ponta. Os botões + e −, as teclas + e −, o gesto de pinça e
Ctrl ou Cmd com a roda do mouse aproximam e afastam a parede; na rede inteira, arrastar a parede mostra
outras partes. No celular, depois de cada Próximo ou Voltar, a página rola para que a parede e a
legenda da cena fiquem juntas na tela.

Pelo teclado, Tab leva a um cartão da parede de cada vez (foco em rodízio): as setas passam para o
cartão seguinte ou anterior, e Enter puxa o fio do cartão na rede inteira (na história, Enter passa
para a rede inteira com esse cartão puxado). Com o foco fora dos cartões, as setas para a direita e
para a esquerda continuam avançando e voltando a história.

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
| O que a pessoa disse | O trecho escolhido em amarelo (uma frase em `udv_v1`, duas frases seguidas ou a citação inteira em `udv_v2`), no meio de até duas frases antes e duas depois do mesmo turno (até 600 caracteres de cada lado), com o turno e a posição na transcrição. Embaixo, menor, a cópia em inglês que o verificador leu, marcada como tradução automática, com o nome do modelo. |
| O que as medidas dizem | A régua da semelhança com o corte da execução, a régua do apoio com as três faixas e os dois cortes explicados em palavras, quantos trechos da mesma faixa a conferência humana julgou corretos, um parágrafo que separa as duas perguntas (onde está o trecho e quanto ele apoia a afirmação) e aponta quando as respostas discordam, a tabela das oito perguntas (a pergunta em pt-BR, o valor na leitura em português e na cópia em inglês) e a linha do outro modelo de NLI. Uma linha da tabela fica marcada em vermelho quando as duas leituras diferem em 0,30 ou mais. |
| Outras frases (ou trechos) parecidos que a pessoa disse | As unidades candidatas da execução (frases em `udv_v1`, janelas de duas frases em `udv_v2`), em ordem de nota, com uma barra por nota e o corte marcado; a escolhida fica destacada, e a última linha diz a distância da segunda para a primeira (ou, em `quote_found`, em que posição a frase escolhida pelas aspas ficaria pelo sentido). |

Uma afirmação sem trecho (`no_evidence`, `person_not_resolved`) mostra o cartão, a explicação do
motivo uma vez só e os dois medidores com "não se aplica". Em `no_evidence`, a explicação traz os
turnos da pessoa como estão na transcrição (por exemplo, "(Manifestação em LIBRAS.)") e, quando o
turno seguinte é de um intérprete, esse turno também, porque a fala pode ter sido registrada na voz
dele. Em `person_not_resolved`, ela lista até três nomes de quem fala na transcrição com grafia
parecida com o da matéria (semelhança de texto de pelo menos 0,75, calculada na exportação), como
nomes para conferir: a grafia parecida não confirma que é a mesma pessoa. Uma audiência exportada sem
`--verifier-report` mostra a pasta sem a parte das medidas e com uma linha dizendo que o apoio do
verificador não foi calculado.

### Como os resultados são descritos

Os nomes dizem o que o procedimento fez. Cada resultado tem um nome só, usado em toda a página, numa
forma longa (veredito, pasta) e numa curta (carimbos, quadradinhos, listas); os textos ficam em
`js/copy.js`.

| `tier` | Forma longa | Forma curta |
| --- | --- | --- |
| `quote_found` | Aspas achadas na fala | aspas achadas na fala |
| `semantic_match_high` | Trecho parecido pelo sentido | trecho parecido |
| `semantic_match_weak` | Só um trecho pouco parecido | pouco parecido |
| `no_evidence` | Sem frase para comparar | sem frase (nos quadradinhos, "sem trecho") |
| `person_not_resolved` | Pessoa não achada entre quem fala | pessoa não achada (nos quadradinhos, "sem trecho") |

Três medidas aparecem na página, cada uma com seu nome: **semelhança** (o cosseno entre a afirmação e
o trecho, que decide entre trecho parecido e pouco parecido), **apoio** (a nota do verificador) e,
nos perfis, **palavras em comum** (a ligação lexical entre um item do perfil e uma frase). Nenhuma
delas é chamada de "nota" nos resumos. Termos técnicos (UDV, cosseno, `legacy_random`, TF-IDF, nome da
rodada, posições em caracteres) só aparecem na análise completa, com uma explicação em palavras; o
glossário da pasta define UDV como Unidade Deliberativa Verificável, uma afirmação que a matéria
atribui a alguém, ligada a essa pessoa e ao trecho da fala que a busca escolheu.

Em `quote_found`, o que foi achado igual na fala é o começo da citação (6 ou mais palavras), não a
afirmação inteira, e o cartão diz quantas palavras das aspas coincidem. A semelhança vai de 0 a 1, com
o corte da execução (`run.threshold`) marcado na régua do cartão, e a página diz que ela não é a chance
de a afirmação estar certa. A semelhança aparece com duas casas decimais; quando o arredondamento a colocaria do
outro lado do corte (0,448 viraria 0,45 com corte 0,45), ela ganha uma ou duas casas a mais, para que
o número mostrado nunca contradiga o resultado. Uma pessoa que não foi achada entre os falantes pode não
ter falado ou aparecer com outro nome na transcrição, e a legenda diz isso. Os turnos são numerados a
partir de 1 em toda a página, inclusive nos endereços.

### Apoio do verificador

O apoio é a nota que o verificador treinado dá ao trecho como sustentação da afirmação, de 0 a 1. A
página a mostra em três faixas, com uma paleta própria (roxo, com um ícone de círculo vazio, meio
cheio ou cheio), separada das cores dos resultados da busca:

| Faixa | Valores em `udv_v2` | Corretas na conferência humana |
| --- | --- | --- |
| apoio fraco | abaixo de 0,2429 | 5 de 15 (mais 3 parciais) |
| apoio incerto | de 0,2429 a 0,7478 | 23 de 48 (mais 15 parciais) |
| apoio forte | a partir de 0,7478 | 48 de 58 (mais 7 parciais) |

Os dois cortes são lidos do bloco `signals.verifier` de cada audiência: 0,2429 é o corte calibrado para
premissas de UDV (`udv_threshold`, regra `legacy_random`: o ponto médio entre a nota típica de pares
certos e a de pares em que o trecho foi sorteado de outra fala, em audiências de treino) e 0,7478 é o
corte do treino do verificador no benchmark NLI. As contagens da última coluna vêm do bloco
`validation.verifier_bands` de `index.json`, e a página só as mostra quando os cortes gravados ali são
os mesmos da audiência. São contagens de uma amostra do conjunto de teste, julgada por uma pessoa: dizem
com que frequência os trechos daquela faixa estavam certos na amostra, não a chance de uma afirmação
específica estar certa. Quando o verificador e a pergunta "O trecho sustenta a afirmação?" (P4)
caem em lados opostos, a pasta diz que o apoio combina as oito respostas nas duas leituras e mostra o
valor de P4.

### Conferência humana

Com `--human-validation`, os resumos da lista, do mural e da pasta dizem o que foi conferido: uma
pessoa julgou o trecho de 121 afirmações do conjunto de teste (aspas achadas na fala, 25 de 35 corretas e 6
parciais; trecho parecido, 50 de 80 corretas e 19 parciais; pouco parecido, 1 de 6 correta), e os dois
critérios fixados antes da conferência, em 23/09/2026, não foram atingidos: o limite inferior do
intervalo de 95% para a parte de corretas nas aspas achadas ficou em 55% (o critério pedia 90%) e o
das corretas e parciais nos trechos parecidos sem aspas curtas ficou em 71% (o critério pedia 75%). A
análise completa da pasta traz os intervalos por resultado, os critérios, a herança (101 dos 127
julgamentos vêm da primeira versão da busca, supondo que acrescentar uma frase do mesmo turno não tira
o apoio que o trecho dava; os outros 26 foram julgados de novo) e a checagem de 13 afirmações sem
trecho, em 11 das quais a pessoa citada fala na transcrição (a resposta é dada por pessoa e copiada
para as afirmações dela, então essas 13 não são independentes). Os números vêm todos do arquivo; nada
disso está escrito no JavaScript.

## Mapa do caso e perfis

**Mapa do caso.** Uma aba fixa na borda direita (no celular, um botão na barra do topo, que não cobre
o conteúdo) e a tecla `M` abrem o mapa de qualquer página; a aba fica dentro do `<header>` da página; `M` de novo volta para onde se estava, e `/` põe o cursor na busca.
O mapa lista os perfis em ordem alfabética, cada um com uma barra que divide os itens do perfil em
três partes (ligados a uma afirmação da matéria, só com frase parecida, sem frase) e o número de
afirmações que as matérias atribuem à pessoa, e as audiências da mais recente para a mais
antiga, com links para o mural, para as pastas e para o perfil de cada pessoa citada que tem perfil.
A busca procura por nome, cargo, título ou número da audiência, sem acento e sem diferença entre
maiúsculas e minúsculas. No topo de todas as páginas, uma trilha (`Mapa do caso › Audiência 167 ›
Perfil de Vanessa Negrini`) mostra o caminho e volta a qualquer ponto dele. Os nomes dos atores são os
que as matérias usam (`display_name`); quando nenhuma matéria cita a pessoa, o nome da transcrição
aparece com só as iniciais maiúsculas, e a ficha do perfil mostra também a grafia da transcrição.

**Perfil.** O perfil de um ator é montado como um dossiê: uma ficha com o monograma no lugar da foto
(a página não busca imagens), a tabela de dados (cargo, audiências, turnos, itens), a proveniência
(modelo, rodada, versão do prompt, data, tokens e sha256 do arquivo de perfis), um cartão que explica
como o perfil foi feito, os cartões de cada seção do texto e a folha "Evidência de cada item". Um aviso
no topo diz que o texto foi escrito por um modelo de linguagem a partir das falas e que a evidência
precisa ser conferida. Cada item do texto tem um número que leva à sua evidência; a evidência é a
frase das falas da pessoa mais parecida com o item, com link para o leitor da transcrição, e, quando
existe, a afirmação da matéria ligada a ele, com o resultado da busca, o selo de apoio do verificador e
o link para a pasta. As notas adesivas só mostram números calculados na exportação (por exemplo, quantos
itens não têm afirmação da matéria ligada). As três posições do resumo são numeradas de 1 a 3, cada uma
com o número do item no perfil ("item 3 do perfil") e o texto inteiro. Tudo isso fica na análise completa; o resumo do topo está descrito em "Resumo primeiro,
análise completa depois".

**Ligações.** A pasta de uma afirmação tem o link "Ver o perfil de ..." quando a pessoa tem perfil, e o
cabeçalho do mural lista os perfis de quem fala na audiência. Do perfil, cada audiência leva ao mural,
cada frase ao leitor da transcrição e cada UDV à sua pasta.

## Como a parede se organiza

As audiências vão de 3 a 31 afirmações, de 2 a 12 participantes e de 7 a 4.898 turnos. Uma página
por turno com posições fixas, como nos protótipos, não cabe nas audiências maiores, então a página
faz a disposição a partir dos dados de cada audiência:

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
| `js/copy.js` | Os textos dos resultados, das medidas, das faixas de apoio, da conferência humana, das oito perguntas, do glossário e dos resumos. |
| `js/support.js` | O selo, o ícone e a régua das faixas de apoio do verificador. |
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

- A conferência humana foi feita por uma pessoa só, numa amostra do conjunto de teste, e 101 dos 127
  julgamentos foram herdados da primeira versão da busca; não há medida de concordância entre
  anotadores, e os intervalos são largos (nas faixas de apoio, 15 a 58 afirmações cada).
- As faixas de apoio usam dois cortes que vêm de artefatos: 0,7478 é o corte do treino no benchmark NLI
  (`experiments/artifacts/udv/udv_v2_verifier_report.json`, `primary.fit.threshold`) e 0,2429 o corte
  para premissas de UDV (`experiments/artifacts/calibration/udv_verifier_threshold_v1.json`,
  `udv_threshold`). Nenhum dos dois foi escolhido para separar faixas nesta página, e a faixa incerta
  junta trechos que a conferência julgou corretos e incorretos em proporções parecidas.
- Os nomes parecidos de quem fala são sugestões de grafia; a página não liga a afirmação a esses
  turnos nem recalcula o resultado.
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
- As visualizações novas propostas na revisão de usabilidade (por exemplo, a distribuição do apoio na
  lista de matérias) ainda não foram feitas.
- A cópia em inglês é a que o verificador leu, gravada no cache de tradução; a página não traduz nada,
  e um erro de tradução muda juntas todas as respostas da leitura em inglês.
- O limite de 0,30 que marca uma pergunta em que as duas leituras discordam foi escolhido para a
  leitura da página, não medido.
- A ligação entre um item do perfil e a fala é lexical (TF-IDF das palavras), escolhida frase a frase;
  uma frase parecida pode não sustentar o item, e o item pode resumir várias falas. Os limites de 0,15
  (frase) e 0,30 (UDV por texto parecido) foram escolhidos para a leitura da página, não medidos.
- Na rodada `qwen38_27b` de treino, com `udv_v2`, só 202 dos 4.928 itens dos 264 perfis se ligam a
  uma UDV; 4.290 têm só uma frase parecida e 436 não têm nenhuma
  (`experiments/artifacts/web/export_site_udv_v2.json`). A maior parte do texto dos perfis se confere lendo a
  transcrição, sem uma afirmação da matéria ao lado.
- Os perfis só cobrem o split de treino e os atores que falam em mais de uma audiência; a simulação de
  atores não aparece na página.
- O estado aberto ou fechado é lembrado por tipo de página, não por endereço: quem abre a análise de uma
  pasta encontra as próximas pastas abertas.
- A legenda da parede não tem link para os perfis; eles aparecem no cabeçalho do mural e nas pastas.
