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
| `mock_h.html` | H, pasta do caso: uma afirmação por vez, com o voto do leitor antes dos carimbos |
| `mock_i.html` | I, os dois juízes: a audiência inteira num plano de semelhança contra verificador, com cortes arrastáveis |
| `mock_p1.html` | P1, dossiê do ator: linha do tempo, afirmações por split e o lugar do perfil e da conferência |

Cada arquivo é um fragmento HTML (um `<style>`, uma `<section class="mk-X" id="mock-X">` e um
`<script>`) que lê os dados de `window.HEARING`. Os dados não ficam no repositório, porque incluem a
transcrição completa da audiência; eles são gerados pela biblioteca:

```
uv run bookworm export-hearing --config configs/udv.toml --run-name udv_v1 --hearing 70 --output hearing70.json
```

Para abrir um fragmento, monte uma página que defina `window.HEARING` com o conteúdo desse JSON,
carregue `https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js` e
`https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js`, e inclua o fragmento depois.

## Segunda rodada: H, I e P1

H e I leem, além de `window.HEARING`, um objeto `window.SIGNALS` com o que a verificação produziu para
a mesma audiência; P1 lê só `window.ACTOR`. Nenhum dos dois objetos é exportado pela biblioteca ainda,
então os mockups foram abertos com dados montados a partir destes arquivos:

- `SIGNALS`: `challenge/artifacts/udv/udv_v1_verifier.jsonl` e o relatório do verificador (nota e corte
  do verificador principal, com as métricas do benchmark NLI), os arquivos de
  `challenge/artifacts/experiments/nli_verifier/udv_v1_verifier/scores/` (respostas às oito perguntas
  para os dois modelos, mDeBERTa XNLI e cosseno), a bateria de perguntas de `challenge/configs/nli_verifier.toml`
  e o cache de tradução NLLB (`challenge/artifacts/cache/translation/`), procurado pelo texto com espaços
  normalizados. Por afirmação: `primary_probability`, `supported_at_train_threshold`, `p4_supports`,
  `premise_pt`/`hypothesis_pt`, `translation`, `laya.<modelo>.answers.<pergunta>` (`choice`,
  `probabilities`, `signal`, `against`) e `panel`, `xnli`, `cosine_serafim`; e, por afirmação, as aspas
  extraídas da proposição e o prefixo conferido. `more_opinions` fica vazio até o estudo
  `confidence_v2` terminar.
- `ACTOR`: as falas da pessoa em todas as audiências (`collect_actor_speeches` com
  `challenge/configs/hearing_actors.toml`), o split de cada audiência (`challenge/artifacts/splits/temporal_v1.json`)
  e as afirmações de `udv_v1` ligadas à pessoa pela mesma regra de turno da conferência de perfis,
  cada uma com o grupo que `validate-profiles` daria a ela (`in_prompt`, `held_out`,
  `split_not_evaluated`, `not_in_prompt`, `tier_not_evaluated`). `profile` segue o formato do
  `ProfileRecord`, `validation.pairs` o dos pares da conferência e `simulation` o da avaliação da
  simulação; os três ficam `null` enquanto não existirem, e os campos `*_expected` guardam o que já se
  sabe sem eles (blocos do prompt, audiências de treino que o perfil vai ler, 264 candidatos e o acaso
  de 1/264 para acerto no topo e 0,023 para MRR, 25% na múltipla escolha).

P1 nunca mostra texto de perfil que não venha de `ACTOR.profile`: sem ele, as quatro notas mostram só a
estrutura e o aviso de que o perfil ainda não foi gerado.

O front não tem modo escuro, e os mockups também não.

Limitações conhecidas: os fragmentos de A a G foram feitos com a rodada `udv_v0` da audiência 70; A, B e E
numeram os turnos a partir de 0 e D, F e G a partir de 1; os níveis de confiança ainda não foram
validados por anotação humana. H e I usam a rodada `udv_v1` da audiência 70; P1 foi aberto com
Erika Kokay, Danilo Forte e Rodrigo Agostinho.
