export function wallMarkup() {
  return `
<div class="wl-top">
  <div class="wl-top-l">
    <h2 class="wl-vh">Rede de barbantes</h2>
    <p class="wl-lede">Cada cartão é uma afirmação da matéria. Os barbantes ligam o cartão a quem disse e ao trecho da audiência onde procuramos essa fala.</p>
  </div>
  <div class="wl-top-r">
    <div class="wl-tools">
      <div class="wl-pick">
        <label class="wl-vh" for="wl-pick">Escolha uma afirmação</label>
        <select id="wl-pick" class="wl-pick-sel"></select>
      </div>
      <button type="button" class="wl-btn wl-all">Ver a rede inteira</button>
    </div>
  </div>
</div>
<div class="wl-story">
  <div class="wl-wrap" tabindex="0" role="group" aria-roledescription="parede" aria-label="Parede com a rede de barbantes. As setas para a direita e para a esquerda avançam e voltam; mais e menos aproximam e afastam.">
    <div class="wl-world">
      <div class="wl-cork" aria-hidden="true"></div>
      <div class="wl-glabs" aria-hidden="true"></div>
      <svg class="wl-sv wl-sv-lo" aria-hidden="true" focusable="false"></svg>
      <div class="wl-objs"></div>
      <svg class="wl-sv wl-sv-hi" aria-hidden="true" focusable="false"></svg>
      <div class="wl-tags"></div>
    </div>
    <div class="wl-zoom" role="group" aria-label="Aproximar e afastar a parede">
      <button type="button" class="wl-zin" aria-label="Aproximar">+</button>
      <button type="button" class="wl-zout" aria-label="Afastar">&minus;</button>
      <button type="button" class="wl-zfit is-fit" aria-label="Enquadrar a rede inteira">ver<br>tudo</button>
    </div>
    <p class="wl-hint is-off" aria-hidden="true"></p>
    <div class="wl-legend" hidden></div>
    <div class="wl-reader" role="dialog" aria-labelledby="wl-reader-h" hidden>
      <h3 class="wl-reader-h" id="wl-reader-h">A transcrição em volta do trecho</h3>
      <p class="wl-reader-pos"></p>
      <div class="wl-reader-body" tabindex="0"></div>
      <div class="wl-reader-ctl">
        <button type="button" class="wl-btn wl-rd-before">Ler mais antes</button>
        <button type="button" class="wl-btn wl-rd-after">Ler mais depois</button>
        <button type="button" class="wl-btn wl-btn-go wl-rd-close">Fechar</button>
      </div>
    </div>
  </div>
  <div class="wl-cap">
    <div class="wl-cap-main">
      <p class="wl-cap-scene"></p>
      <p class="wl-cap-text" aria-live="polite"></p>
    </div>
    <div class="wl-nav">
      <button type="button" class="wl-btn wl-back">Voltar</button>
      <button type="button" class="wl-btn wl-btn-go wl-next">Começar</button>
      <button type="button" class="wl-btn wl-auto" aria-pressed="false">Tocar sozinho</button>
      <a class="wl-btn wl-case" hidden><span class="wl-case-w">Abrir a pasta<span class="wl-case-x"> desta afirmação</span></span><span class="wl-case-n">Pasta</span></a>
    </div>
  </div>
</div>
<ul class="wl-key" aria-label="Como ler os fios"></ul>
<p class="wl-disc"></p>
<p class="wl-keys">No teclado, as setas para a direita e para a esquerda avançam e voltam, e as teclas + e &minus; aproximam e afastam a parede. Na rede inteira, arraste a parede para ver outras partes.</p>
<div class="wl-ask" role="search" aria-labelledby="wl-ask-h">
  <h3 class="wl-ask-h" id="wl-ask-h">Pergunte à audiência</h3>
  <p class="wl-ask-note">Busca por palavras nesta audiência. Nada é gerado: só mostramos frases que foram ditas.</p>
  <p class="wl-ask-aud"></p>
  <form class="wl-askform">
    <label class="wl-vh" for="wl-q">Sua pergunta</label>
    <textarea id="wl-q" class="wl-q" rows="2" autocomplete="off" spellcheck="false"></textarea>
    <button type="submit" class="wl-btn wl-btn-go">Procurar</button>
  </form>
  <p class="wl-ask-sug"></p>
  <p class="wl-ask-terms" aria-live="polite"></p>
  <figure class="wl-astrip" hidden>
    <div class="wl-astrip-row" aria-hidden="true"></div>
    <div class="wl-astrip-ends" aria-hidden="true"><span>início da audiência</span><span>fim</span></div>
    <figcaption class="wl-astrip-cap"></figcaption>
  </figure>
  <ol class="wl-apages"></ol>
</div>`;
}
