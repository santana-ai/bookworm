# Mockups da demo de verificação de atribuição

Protótipos visuais da demo que mostra, para uma audiência, como cada afirmação atribuída pela matéria
a um participante é ligada ao trecho da transcrição que a sustenta. Serviram para escolher a direção do
front; não são a demo final. A direção escolhida foi a F, e a demonstração que a leva para todas as
matérias está em [`../app/`](../app/), descrita em [`../README.md`](../README.md).

| Arquivo | Direção |
| --- | --- |
| `mock_f.html` | F, rede de barbantes: cartões e fios formando a rede da audiência |
| `mock_g.html` | G, caderno de mapa mental: a mesma rede desenhada à mão |
| `mock_d.html` | D, passeio pelo grafo: mapa de linhas com câmera percorrendo a verificação |
| `mock_e.html` | E, parede de evidências: cenas que prendem cada item num quadro |
| `mock_a.html`, `mock_b.html`, `mock_c.html` | primeira rodada, mais técnica |

Cada arquivo é um fragmento HTML (um `<style>`, uma `<section class="mk-X" id="mock-X">` e um
`<script>`) que lê os dados de `window.HEARING`. Os dados não ficam no repositório, porque incluem a
transcrição completa da audiência; eles são gerados pela biblioteca:

```
uv run bookworm export-hearing --config configs/udv.toml --run-name udv_v1 --hearing 70 --output hearing70.json
```

Para abrir um fragmento, monte uma página que defina `window.HEARING` com o conteúdo desse JSON,
carregue `https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js` e
`https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js`, e inclua o fragmento depois.

Limitações conhecidas: os fragmentos foram feitos com a rodada `udv_v0` da audiência 70; A, B e E
numeram os turnos a partir de 0 e D, F e G a partir de 1; os níveis de confiança ainda não foram
validados por anotação humana.
