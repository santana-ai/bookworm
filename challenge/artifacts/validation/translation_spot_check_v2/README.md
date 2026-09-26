# Checagem manual da tradução (translation_spot_check_v2)

A planilha `spot_check.csv` tem 80 linhas, feitas a partir de 40 trechos de transcrição sorteados
(semente 42) entre as unidades de tradução dos trechos de validação. Cada trecho foi traduzido para o
inglês por dois tradutores automáticos, facebook/nllb-200-distilled-600M e facebook/m2m100_418M, nas
revisões fixadas em `configs/translation.toml`.

Cada frase aparece duas vezes, em linhas diferentes e fora de ordem, cada vez com a tradução de um
dos dois tradutores; as duas traduções podem até coincidir. Julgue cada linha por si só, sem procurar
a outra linha da mesma frase para comparar, e não tente adivinhar qual tradutor produziu a tradução.

Cada unidade é uma frase da transcrição; quando uma frase tem menos de quatro palavras ou termina em
abreviação (Sr., Dra., V.Exa., art.), ela aparece unida à frase seguinte, ou, no fim do trecho, à
frase anterior, exatamente como foi traduzida. Nas traduções, apóstrofos e aspas tipográficos
aparecem na forma simples (' e ").

Para cada linha, leia `texto_pt` inteiro e depois `traducao_en`, e preencha:

- `adequacao` (1 a 4): quanto do sentido de `texto_pt` está em `traducao_en`.
  4: todo o sentido. 3: quase todo, com perda ou erro que não muda a afirmação.
  2: parte do sentido, ou um erro que muda a afirmação (negação perdida, sujeito trocado, número errado).
  1: pouco ou nada do sentido, ou sentido oposto.
- `fluencia` (1 a 4): quão correto é o inglês, lido sozinho, sem olhar o português.
  4: inglês correto. 3: compreensível, com erros pequenos. 2: difícil de entender. 1: incompreensível.
- `observacao` (opcional): o tipo de erro, por exemplo nome próprio traduzido, negação perdida,
  repetição, trecho omitido, termo técnico ou jurídico errado.

Julgue a tradução da frase como ela está, mesmo quando a frase original estiver incompleta ou mal
transcrita; nesse caso, anote em `observacao`. Não use outro tradutor automático como referência e
não abra `spot_check_key.json` antes de terminar: ele diz qual tradutor produziu cada linha e traz os
identificadores das audiências e as marcações automáticas de cada item.

A planilha é regenerada por `python -m utils.translation spot-check`, que se recusa a sobrescrever
um arquivo com qualquer julgamento preenchido.
