import { caseHash } from "./case.js";
import { ATLAS_COPY, DISCLOSE_HINTS } from "./copy.js";
import { bindDisclose, discloseBar, discloseEnd } from "./disclose.js";
import { statsHtml } from "./home.js";
import { countLabel, esc, fmtDate, foldText } from "./text.js";
import { icon, initials, profileHash } from "./profile.js";

function words(q) {
  return foldText(q)
    .split(/[^\p{L}\p{N}]+/u)
    .filter(Boolean);
}

function hits(hay, ws) {
  return ws.every((w) => hay.indexOf(w) >= 0);
}

function claimBar(c) {
  const seg = (k, n) => (n ? '<i data-k="' + k + '" style="flex-grow:' + n + '"></i>' : "");
  return '<span class="at-bar" aria-hidden="true">' + seg("udv", c.with_udv) + seg("pas", c.passage_only) + seg("none", c.without_evidence) + "</span>";
}

function actorCard(a) {
  const c = a.claims;
  return (
    '<li class="at-a"><a href="' + profileHash(a.slug) + '">' +
    '<span class="at-mono" aria-hidden="true">' + esc(initials(a.name)) + "</span>" +
    '<span class="at-a-n">' + esc(a.name) + "</span>" +
    (a.role ? '<span class="at-a-r">' + esc(a.role) + "</span>" : "") +
    '<span class="at-a-m">' + esc("fala em " + countLabel(a.n_hearings, "audiência", "audiências") + ", perfil leu " + a.n_hearings_in_profile) + "</span>" +
    claimBar(c) +
    '<span class="at-a-c">' + esc(countLabel(c.claims, "item", "itens") + ", " + c.with_udv + " com UDV") + "</span>" +
    "</a></li>"
  );
}

function hearingRow(h, people, bySlug) {
  const names = h.actors.map((name) => {
    const slug = people[name];
    if (!slug) return '<li class="at-p">' + esc(name) + "</li>";
    const a = bySlug.get(slug);
    return '<li class="at-p is-prof"><a href="' + profileHash(slug) + '" title="Perfil de ' + esc(a ? a.name : name) + '">' + icon("doc") + esc(name) + "</a></li>";
  });
  const date = fmtDate(h.article_date);
  return (
    '<li class="at-h">' +
    '<p class="at-h-k"><span>' + (date ? esc(date) : "sem data") + "</span><span>Audiência " + h.id + "</span></p>" +
    '<p class="at-h-t"><a href="#h' + h.id + '">' + esc(h.title) + "</a></p>" +
    '<p class="at-h-l"><a class="at-go" href="#h' + h.id + '">Mural</a>' +
    (h.n_udvs ? '<a class="at-go" href="' + caseHash(h.id, 1) + '">' + esc(countLabel(h.n_udvs, "pasta", "pastas")) + "</a>" : "") + "</p>" +
    '<ul class="at-ps" aria-label="Quem a matéria cita">' + names.join("") + "</ul>" +
    "</li>"
  );
}

export function createAtlas(host, index, actors, onRender) {
  const people = actors ? actors.people : {};
  const list = actors ? actors.actors.slice().sort((a, b) => a.name.localeCompare(b.name, "pt-BR")) : [];
  const bySlug = new Map(list.map((a) => [a.slug, a]));
  const hearings = index.hearings.slice().sort((a, b) => (b.article_date || "").localeCompare(a.article_date || "") || b.id - a.id);
  const aHay = list.map((a) => foldText([a.name, a.role || ""].join(" ")));
  const hHay = hearings.map((h) => foldText([h.title, h.assunto, "audiencia " + h.id, h.actors.join(" ")].join(" ")));
  const totalUdvs = index.hearings.reduce((s, h) => s + h.n_udvs, 0);
  const pr = actors ? actors.profiles : null;
  const stats = [
    { n: index.hearings.length, label: "audiências" },
    { n: totalUdvs, label: "afirmações atribuídas" },
    { n: list.length, label: actors ? "perfis de atores" : "perfis nesta exportação" },
  ];
  host.innerHTML =
    '<div class="at-head">' +
    '<h1 class="at-title" id="at-title" tabindex="-1">' + icon("doc") + "Mapa do caso</h1>" +
    '<p class="lede">' + ATLAS_COPY.lede + "</p>" +
    statsHtml(stats, "Números desta exportação") +
    '<div class="tools" role="search" aria-label="Procurar no mapa"><div class="field field-q"><label for="at-q">Procurar ator ou audiência</label>' +
    '<input id="at-q" type="search" autocomplete="off" spellcheck="false" placeholder="Nome, título ou número da audiência"></div></div>' +
    '<p class="count" id="at-count" aria-live="polite"></p>' +
    "</div>" +
    '<div class="at-grid at-sum">' +
    (actors ? '<section class="at-sec" aria-labelledby="at-sa-h"><h2 class="at-sec-h" id="at-sa-h" data-at-sa-h></h2><ol class="at-al" id="at-sal"></ol></section>' : "") +
    '<section class="at-sec" aria-labelledby="at-sh-h"><h2 class="at-sec-h" id="at-sh-h" data-at-sh-h></h2><ol class="at-hl" id="at-shl"></ol></section>' +
    "</div>" +
    discloseBar("atlas", "at-more", DISCLOSE_HINTS.atlas) +
    '<div class="at-more" id="at-more">' +
    (actors && pr ? '<p class="at-stats">' + esc("Perfis da rodada " + pr.run + ", escritos por " + pr.models.join(", ") + ". " + ATLAS_COPY.profiles) + "</p>" : "") +
    '<div class="at-grid">' +
    '<section class="at-sec at-actors" aria-labelledby="at-a-h"><h2 class="at-sec-h" id="at-a-h">Todos os perfis de atores</h2>' +
    (actors
      ? '<p class="at-legend"><span><i data-k="udv"></i>item com UDV</span><span><i data-k="pas"></i>só frase parecida</span><span><i data-k="none"></i>sem frase</span></p><ol class="at-al" id="at-al"></ol>'
      : '<p class="empty">Esta exportação não tem perfis. Para incluí-los, rode <code>bookworm export-site</code> com <code>--profiles</code>; o comando está em <code>web/README.md</code>.</p>') +
    "</section>" +
    '<section class="at-sec at-hearings" aria-labelledby="at-h-h"><h2 class="at-sec-h" id="at-h-h">Todas as audiências</h2><ol class="at-hl" id="at-hl"></ol></section>' +
    "</div>" + discloseEnd("atlas", "at-more") + "</div>";
  bindDisclose(host, "atlas");
  const sal = host.querySelector("#at-sal");
  const shl = host.querySelector("#at-shl");
  const topActors = list.slice().sort((a, b) => b.n_udvs - a.n_udvs || b.n_hearings - a.n_hearings || a.name.localeCompare(b.name, "pt-BR"));
  const q = host.querySelector("#at-q");
  const al = host.querySelector("#at-al");
  const hl = host.querySelector("#at-hl");
  const count = host.querySelector("#at-count");
  let timer = 0;

  function render() {
    const ws = words(q.value);
    const shownA = list.filter((a, i) => hits(aHay[i], ws));
    const shownH = hearings.filter((h, i) => hits(hHay[i], ws));
    if (al) al.innerHTML = shownA.map(actorCard).join("") || '<li class="empty">Nenhum ator com essas palavras.</li>';
    hl.innerHTML = shownH.map((h) => hearingRow(h, people[String(h.id)] || {}, bySlug)).join("") || '<li class="empty">Nenhuma audiência com essas palavras.</li>';
    const pickA = ws.length ? shownA : topActors;
    const hRow = (h) => hearingRow(h, people[String(h.id)] || {}, bySlug);
    if (sal) {
      host.querySelector("[data-at-sa-h]").textContent = ws.length ? ATLAS_COPY.foundActors : ATLAS_COPY.topActors;
      sal.innerHTML = pickA.slice(0, ATLAS_COPY.nActors).map(actorCard).join("") || '<li class="empty">Nenhum ator com essas palavras.</li>';
    }
    host.querySelector("[data-at-sh-h]").textContent = ws.length ? ATLAS_COPY.foundHearings : ATLAS_COPY.recentHearings;
    shl.innerHTML = shownH.slice(0, ATLAS_COPY.nHearings).map(hRow).join("") || '<li class="empty">Nenhuma audiência com essas palavras.</li>';
    count.textContent = ws.length
      ? "Na busca: " + countLabel(shownA.length, "perfil", "perfis") + " e " + countLabel(shownH.length, "audiência", "audiências") + ". Aqui aparecem os primeiros; a análise completa mostra todos."
      : ATLAS_COPY.count;
    if (onRender) onRender();
  }

  q.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(render, 120);
  });
  render();
  return { focusSearch: () => q.focus(), input: q };
}
