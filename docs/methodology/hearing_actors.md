# Falas por ator entre audiências

Este documento descreve como separar a fala completa de cada pessoa ao longo das 206 audiências do
PublicHearingBR e justifica cada regra com medidas feitas sobre o dataset.

## Objetivo

Queremos um perfil por ator: juntar tudo o que uma pessoa disse em todas as audiências de que
participou, extrair desse texto um perfil estruturado com um LLM e representar o perfil como
embedding. O primeiro passo, descrito aqui, é produzir a entrada desse pipeline: para cada pessoa, a
lista de audiências em que falou e, em cada uma, a fala completa, sem recortes.

Como o interesse é o comportamento de uma pessoa ao longo do tempo, a separação gera dois arquivos: um
com quem fala em uma única audiência e outro com quem fala em duas ou mais. O segundo é o material do
perfil; o primeiro serve de comparação.

## Como reproduzir

Os números deste documento saem de dois comandos, rodados dentro de `experiments/`:

```
uv run python -m experiments.actors.measure_hearing_actors
uv run python -m experiments.actors.build_speeches
```

O primeiro mede o dataset e justifica as regras; a saída fica em
`artifacts/hearing_actors/measurements.json`. O segundo aplica as regras e gera os dois arquivos de
falas em `artifacts/cache/hearing_actors/` (não versionados, porque reproduzem quase toda a
transcrição), além de `artifacts/hearing_actors/ambiguous_names.json` (pares de nomes que restam para
revisão) e `artifacts/hearing_actors/actor_speeches_stats.json` (números da rodada, incluindo a
verificação de que todo `text` é igual a `transcricao[start_char:end_char]`). A configuração dos dois
fica em `configs/hearing_actors.toml` (hash do LDS, corte de palavras dos turnos de presidente,
chaves descartadas, caminhos de saída e a seção `[merges]`, com os grupos de mescla e as
reatribuições confirmados na revisão dos pares). Cada rodada leva menos de 5 segundos.

## A unidade: turno de fala

A transcrição marca cada troca de falante com um cabeçalho, como `O SR. JORGE SOLLA (Bloco/PT - BA) -`
ou `A SRA. PRESIDENTE (Erika Kokay. PT - DF) -`. Um turno é o texto entre um cabeçalho e o seguinte. A
função `split_into_turns` (`bookworm/src/bookworm/transcript/turns.py`), a mesma usada pela UDV, encontra 17.264 turnos
nas 206 audiências.

O turno resolve o problema de saber quem disse o quê sem depender da matéria: a fala de uma pessoa numa
audiência é a soma dos seus turnos. Quando alguém interrompe, a fala de uma pessoa fica dividida em
vários turnos, por isso a fala completa numa audiência é a concatenação de todos eles, em ordem.

A pessoa de cada turno é identificada pelo nome do cabeçalho, sem acentos e em maiúsculas. Nos turnos
de presidente vale o nome entre parênteses. Com essa chave há 1.901 falantes distintos.

## Quem aparece em mais de uma audiência

| audiências por falante | falantes | com partido e UF no cabeçalho |
|---|---|---|
| exatamente 1 | 1.619 (85%) | |
| 2 ou mais | 282 | 209 |
| 3 ou mais | 145 | 134 |
| 5 ou mais | 49 | 47 |

A maioria dos falantes aparece uma única vez; para eles, não há comportamento entre audiências a
observar. O falante mais recorrente é Erika Kokay (30 audiências); o seguinte aparece em 16, e os
demais em 9 ou menos. Somando todas as audiências, a mediana é de 1.329 palavras por falante (p10 368,
p90 3.460).

## Por que não filtrar por partido

O cabeçalho com partido e UF é a forma como a transcrição da Câmara anota deputados federais. As
medidas sustentam essa leitura: os 10.355 turnos com partido no cabeçalho terminam todos em
"Partido - UF"; os 654 cabeçalhos com título de ministro nunca trazem partido; e das 354 pessoas da
matéria que casam com um falante com partido, 353 têm "deputado" ou "deputada" no cargo (a exceção,
Danilo Forte, aparece como "Presidente da Comissão de Desenvolvimento Econômico").

Filtrar por partido, porém, descarta gente que também aparece em várias audiências. Dos 282 falantes
recorrentes, 73 não têm partido no cabeçalho (11 deles em 3 ou mais audiências). Dois não são pessoas:
"INTÉRPRETE" e "Manifestação em língua estrangeira. Tradução simultânea.", cuja fala pertence a outro
participante. Os demais incluem ministros (Nísia Trindade, Waldez Góes, Luiz Marinho), secretários
nacionais (Alexandre da Silva, da pessoa idosa, em 5 audiências), um deputado estadual (Carlos
Giannazi, 4 audiências) e dirigentes de sindicatos, associações e organizações da sociedade civil.
Deputados estaduais, vereadores e ministros não recebem partido no cabeçalho.

Por isso o critério de entrada é o número de audiências, não o partido. Cada registro traz o campo
`has_party_header`, que permite analisar deputados federais e demais participantes em separado.

## Turnos de presidente

Quem preside a sessão fala muito: 42,7% das palavras dos falantes recorrentes estão em turnos de
presidente (Aureo Ribeiro 83,6%, Bia Kicis 81,0%, Rogério Correia 85,4%). Boa parte disso é condução
da sessão: passar a palavra, controlar o tempo, agradecer. Se esses turnos entrarem inteiros, o perfil
descreve quem conduz sessões, não o que a pessoa defende.

Descartar todos os turnos de presidente, porém, perde opinião real. Na rodada `udv_v1` da UDV, 2.105
opiniões da matéria têm evidência localizada na transcrição; 336 delas (16%) estão em turnos de
presidente, de 118 pessoas diferentes, e 56 dessas são citações literais (`quote_found`). Exemplo: a
matéria cita entre aspas Bia Kicis, na presidência, dizendo "Existe uma elite globalista que quer,
sim, forçar essa vacinação, inclusive em bebês, e o Brasil está sendo pioneiro nessa obrigação."

O tamanho do turno separa bem os dois casos. Dos 6.266 turnos de presidente, a mediana tem 28 palavras
(p75 83, p90 254), e 538 têm 300 palavras ou mais, em geral a abertura e o encerramento, onde o
presidente se posiciona.

| corte (palavras) | turnos de presidente mantidos | palavras de presidente mantidas | opiniões da matéria em turno de presidente mantidas |
|---|---|---|---|
| 50 | 2.264 de 6.266 | 89,0% | 332 de 336 |
| 100 | 1.373 | 79,1% | 319 |
| 200 | 796 | 66,1% | 277 |
| 300 | 538 | 56,3% | 256 |

A regra adotada descarta turnos de presidente com menos de 50 palavras e mantém os demais, marcados
com `role: "chair"`. O corte remove 64% dos turnos de presidente e perde 4 das 336 opiniões que a
matéria tirou de turnos de presidente.

O corte remove turnos, não palavras: 89% das palavras de presidente estão nos turnos longos. Depois
dele, turnos de presidente ainda somam 40,4% das palavras do arquivo de falantes recorrentes (contra
3,0% no de falantes de uma audiência). A condução da sessão que resta dentro desses turnos precisa ser
ignorada na etapa de extração do perfil, e o campo `role` permite medir o efeito de incluí-los ou não.

## Identidade entre audiências

Dentro de uma audiência há poucos falantes, e a UDV aceita que um nome contido em outro seja a mesma
pessoa ("Rodrigo Marinho" e "Rodrigo Saraiva Marinho"). Entre audiências essa regra deixa de ser
segura. Há 59 pares de nomes, em audiências diferentes, em que um nome está contido no outro. Alguns
são a mesma pessoa (Deyvid Bacelar e Deyvid Souza Bacelar da Silva), outros não (Luiz Lima e Gustavo
Luiz de Lima Correia). Em 45 dos 59 pares nenhum dos dois nomes tem partido no cabeçalho, e só em 1
ambos têm (Roberto Monteiro e Roberto Monteiro Pai).

A regra adotada junta falantes só pelo nome normalizado exato e lista os pares ambíguos num arquivo à
parte, para revisão manual. Isso deixa o erro visível em vez de misturar duas pessoas num mesmo perfil
sem registro.

O arquivo gerado, `artifacts/hearing_actors/ambiguous_names.json`, usa um critério mais amplo que o
dos 59 pares acima: compara os atores mantidos dos dois arquivos, inclui nomes de um único token e
pares que dividem audiência, e considera também pares com as mesmas palavras em ordem diferente. A
primeira rodada, ainda sem mescla configurada, resultou em 153 pares: 1 de palavras iguais ("PAULO
PEDRO" e "PEDRO PAULO"), 4 que dividem ao menos uma audiência e 148 em que um nome está contido no
outro sem audiência em comum.

Os 153 pares foram revisados um a um com evidência do próprio dataset: o cargo em
`metadados.envolvidos`, a apresentação que o presidente faz antes do primeiro turno da pessoa, a
auto-apresentação no início da fala, o partido e a UF do cabeçalho e, nos casos de mudança de cargo,
a data no carimbo da matéria (Paulo Teixeira é deputado nas audiências de 2022 e Ministro do
Desenvolvimento Agrário na de agosto de 2023; Rodrigo Agostinho é deputado em agosto de 2022 e
presidente do Ibama nas de 2023). Nenhuma identidade foi conferida contra registros externos à base.
O resultado: 41 pares confirmados como a mesma pessoa, que formam os 37 grupos de `[merges.groups]`
na configuração; 110 pares mantidos como pessoas distintas; 1 par resolvido por divisão de chave
(abaixo); e 1 par sem resolução (WELLINGTON, audiência 112, e WELLINGTON LOPES, audiência 177, ambos
em pauta de motoristas de aplicativo), que fica de fora da mescla.

A revisão também encontrou um falso positivo da própria mescla por nome exato: a chave ALEXANDRE
SAMPAIO juntava o presidente da associação de vítimas da mineração em Maceió (audiência 14) e o
diretor da CNC (audiência 38), que é a mesma pessoa de ALEXANDRE SAMPAIO DE ABREU (audiência 108,
mesmo cargo). A seção `[merges.reassignments]` resolve o caso movendo só o turno da audiência 38
para a chave ALEXANDRE SAMPAIO DE ABREU; a audiência 14 permanece um ator próprio. O critério de
pares também deixou um caso de fora: "DUARTE JR." e "DUARTE GONCALVES JR" não formam par porque o
ponto de "JR." sobrevive à normalização; o caso entrou no grupo de mescla do Deputado Duarte Jr.
pela revisão do par "DUARTE" vs "DUARTE GONCALVES JR".

O build aplica a seção `[merges]` ao coletar os turnos: cada chave de alias vira a chave canônica do
grupo e cada reatribuição move os turnos de uma chave numa audiência específica para outra chave. O
build falha se alguma entrada configurada não casar com nenhum turno mantido, o que protege contra
erro de digitação nas chaves. Na rodada atual a mescla move 130 turnos por alias e 1 por
reatribuição, e o número de atores cai de 1.891 para 1.851. Com a mescla aplicada, o
`ambiguous_names.json` regenerado lista os 81 pares que restam entre as chaves finais (1 de palavras
iguais, 80 de nome contido).

## Regras de separação

1. Entram todos os falantes identificados pelo cabeçalho do turno; turnos de presidente contam para a
   pessoa entre parênteses.
2. Saem as chaves que não são pessoa: `INTÉRPRETE`, "Manifestação em língua estrangeira. Tradução
   simultânea." e "Manifestação em língua estrangeira. Tradução não Simultânea." (esta terceira, com
   4 turnos na audiência 99, apareceu ao aplicar as regras ao dataset inteiro).
3. Saem os turnos de presidente com menos de 50 palavras.
4. Saem os turnos formados só por marcação de palco, como "(Palmas.)".
5. Saem os turnos vazios, em que o cabeçalho é imediatamente seguido pelo cabeçalho do turno
   seguinte (falante interrompido antes de dizer qualquer coisa).
6. Turnos descartados não contam como presença na audiência.
7. A mesma pessoa é juntada entre audiências pelo nome normalizado exato e pela seção `[merges]` da
   configuração; os pares de nomes que restam ficam em `ambiguous_names.json` para revisão.
8. O texto de cada turno é o turno inteiro, igual a `transcricao[start_char:end_char]`, sem limpeza.

Com essas regras, 13.190 dos 17.264 turnos são mantidos (saem 4.001 turnos curtos de presidente, 32
turnos de intérprete ou tradução, 36 turnos só de marcação de palco e 5 turnos vazios), e os arquivos
ficam assim:

| arquivo | atores | com partido no cabeçalho | palavras como orador | palavras como presidente |
|---|---|---|---|---|
| `actors_single_hearing.jsonl` | 1.550 | 135 | 2.118.165 | 68.846 |
| `actors_multi_hearing.jsonl` | 301 | 209 | 847.078 | 493.766 |

## Formato

Uma linha por pessoa. Os nomes dos campos seguem os da UDV (`actor`, `hearing_id`, `start_char`),
para que os dois artefatos possam ser cruzados por audiência e posição na transcrição.

```json
{
  "actor": "Jorge Solla",
  "has_party_header": true,
  "party_uf": ["Bloco/PT - BA"],
  "hearings": [
    {
      "hearing_id": 12,
      "full_speech": "...",
      "turns": [
        {"turn_index": 3, "role": "speaker", "start_char": 1520, "end_char": 2890, "text": "..."}
      ]
    }
  ]
}
```

- `text`: o turno inteiro, exatamente `transcricao[start_char:end_char]`.
- `full_speech`: todos os turnos mantidos da pessoa naquela audiência, em ordem, separados por uma
  linha em branco.
- `role`: `speaker` ou `chair`.
- `party_uf`: lista, porque o cabeçalho de um mesmo deputado varia entre audiências ("PT - DF" e
  "Bloco/PT - DF"); vazia quando `has_party_header` é falso. Atores mesclados acumulam os cabeçalhos
  observados em todas as grafias, inclusive um divergente que a revisão tratou como provável erro de
  transcrição ("DUARTE JR." fica com "Bloco/PSB - MA" e "Bloco/PODE - MG", este vindo só da
  audiência 115).

## Limitações

- O descarte das chaves de intérprete e tradução remove o único registro textual de quem falou por
  meio delas: 6 participantes que falaram em Libras (2 na audiência 53, 4 na 111; os turnos com o
  nome deles contêm só "(Manifestação em LIBRAS.)" e a tradução está nos turnos de `INTÉRPRETE`) e 5
  convidados estrangeiros (audiências 89, 99, 179 e 195) ficam sem nenhum turno mantido. Nos 16
  turnos de "Tradução simultânea." o cabeçalho traz o nome do convidado e o texto é a fala traduzida,
  então essa parte é recuperável se a regra mudar; a lista de turnos descartados por chave, com nome
  e audiência, está em `actor_speeches_stats.json`.
- A chave de pessoa é o nome escrito no cabeçalho, corrigido pela seção `[merges]`. O critério que
  gera os pares para revisão exige subconjunto de palavras e é sensível a pontuação (o ponto de
  "JR." escondeu um par), então grafias diferentes da mesma pessoa que não formam par continuam
  separadas sem aviso. Homônimos exatos em audiências diferentes ficam juntos sem aviso; a revisão
  encontrou e corrigiu um caso (ALEXANDRE SAMPAIO), sem garantia de que seja o único.
- A revisão dos pares usou só evidência interna do dataset. O par WELLINGTON e WELLINGTON LOPES
  segue sem resolução e não é mesclado, e nenhuma identidade foi conferida contra registros externos
  da Câmara.
- A regra de que partido e UF no cabeçalho indicam deputado federal foi checada contra o cargo das
  pessoas citadas na matéria (353 de 354), não contra uma lista oficial de deputados.
- O corte de 50 palavras foi avaliado só contra as opiniões que a matéria escolheu citar. Quanto texto
  de condução da sessão ainda passa nos turnos longos de presidente não foi medido; isso exige ler uma
  amostra desses turnos.
- Quando a transcrição omite o cabeçalho de um falante, a fala dele entra no turno anterior. Esse erro
  vem da segmentação e não é detectado pela checagem de `text`, que confirma a cópia do texto, não a
  atribuição do turno.
