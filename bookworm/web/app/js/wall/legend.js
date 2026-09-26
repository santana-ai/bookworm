import { esc, fmtScore } from "../text.js";

function swatch(t, k) {
  const d = t === "loose" ? "M3 3 Q12 3 20 10" : "M3 7 Q20 12 37 7";
  return (
    '<svg width="40" height="14" viewBox="0 0 40 14" aria-hidden="true"><g class="wl-s" data-t="' + t + '"' + (k ? ' data-k="' + k + '"' : "") + '><path class="sh" d="' + d + '"/><path class="ln" d="' + d + '"/>' +
    (t === "att" || k === "q" ? '<path class="l2" d="' + d + '"/>' : "") + (t === "loose" ? '<path class="wl-fray" d="M20 10 l2 3 M20 10 l-1 3"/>' : "") + "</g></svg>"
  );
}

export function installLegend(w) {
  const M = w.M;
  const legend = w.els.legend;
  let legendUser = null;
  const cut = fmtScore(M.CUT);

  function keyHtml() {
    const items = [
      ["pub", null, "a matéria publica"],
      ["att", null, "atribuída a"],
      ["fala", null, "fala na audiência (caderno de falas)"],
      ["sus", "q", "começo das aspas achado na fala"],
      ["sus", "h", "trecho parecido pelo sentido"],
      ["sus", "w", "o mais parecido, com nota abaixo de " + cut],
      ["esta", null, "onde o trecho está no caderno"],
      ["loose", null, "fio solto: fala não achada"],
    ];
    return "<li>Como ler os fios:</li>" + items.map((x) => "<li>" + swatch(x[0], x[1]) + "<span>" + esc(x[2]) + "</span></li>").join("");
  }

  function renderLegend() {
    const li = (t, k, txt) => "<li>" + swatch(t, k) + "<span>" + esc(txt) + "</span></li>";
    legend.innerHTML =
      '<button type="button" class="wl-lg-h" aria-expanded="' + (legend.classList.contains("is-closed") ? "false" : "true") + '">Como ler os fios</button><ul class="wl-lg-l">' +
      li("pub", null, "a matéria publica") + li("att", null, "atribuída a") + li("fala", null, "fala na audiência") + li("esta", null, "onde está no caderno") +
      '<li class="wl-lg-sub">cartão e trecho, conforme o resultado:</li>' +
      li("sus", "q", "aspas na fala") + li("sus", "h", "trecho parecido") + li("sus", "w", "pouco parecido") + li("loose", null, "solto: fala não achada") + "</ul>";
  }

  function autoLegend() {
    let closed = legendUser != null ? legendUser : false;
    if (legendUser == null && w.L && w.L.name === "tall") {
      const was = legend.hidden;
      legend.hidden = false;
      legend.classList.remove("is-closed");
      closed = legend.offsetHeight > 0.34 * w.SH;
      legend.hidden = was;
    }
    legend.classList.toggle("is-closed", closed);
    w.resetZoomBounds();
    const b = legend.querySelector(".wl-lg-h");
    if (b) b.setAttribute("aria-expanded", closed ? "false" : "true");
  }

  function toggleLegend() {
    legendUser = !legend.classList.contains("is-closed");
    autoLegend();
    if (!legend.hidden && ((w.st.mode === "net" && !w.st.pull) || w.st.mode === "rest")) w.camTo(w.frameIds(["all"]));
  }

  w.els.key.innerHTML = keyHtml();
  renderLegend();
  Object.assign(w, { renderLegend, autoLegend, toggleLegend });
}
