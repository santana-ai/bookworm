import { bucketSentence, NOT_CHECKED, NOT_CHECKED_SHORT, statementsSentence, TIER_ORDER } from "./copy.js";
import { DataError, loadHearing, loadIndex } from "./data.js";
import { createHome, dotsHtml } from "./home.js";
import { esc, fmtDate, hashToken } from "./text.js";
import { createWall } from "./wall/wall.js";

const $ = (sel) => document.querySelector(sel);
const view = { home: $("#home"), hearing: $("#hearing"), state: $("#state"), host: $("#wall-host") };
const hv = { k: $("[data-hv-k]"), title: $("#hv-title"), sum: $("[data-hv-sum]"), nav: $("[data-hv-nav]") };
const APP_TITLE = "Rede de barbantes";
let home = null;
let siteIndex = null;
let homeScroll = 0;
let wall = null;
let routeSeq = 0;

document.querySelectorAll("[data-note]").forEach((el) => {
  el.innerHTML = "<b>Ainda sem conferência humana.</b> " + esc(el.classList.contains("is-compact") ? NOT_CHECKED_SHORT : NOT_CHECKED);
});

function setState(html) {
  view.state.innerHTML = html;
}

function loadingHtml(text) {
  return '<div class="spinner"><i aria-hidden="true"></i><span>' + esc(text) + "</span></div>";
}

function errorHtml(title, lines, action, homeLink) {
  return (
    '<div class="state-card"><h2>' + esc(title) + "</h2>" + lines.map((l) => "<p>" + l + "</p>").join("") +
    (action ? '<button type="button" class="btn btn-go" data-retry>' + esc(action) + "</button>" : "") +
    (homeLink ? '<a class="btn" href="#">Ver todas as matérias</a>' : "") + "</div>"
  );
}

function dataHelp() {
  return "Gere os dados com o comando <code>bookworm export-site</code> e sirva a pasta <code>web/app</code> com <code>python -m http.server</code>; os comandos completos estão em <code>web/README.md</code>.";
}

function describe(error) {
  if (error instanceof DataError) {
    if (error.kind === "missing") return "O arquivo <code>" + esc(error.url) + "</code> não existe.";
    if (error.kind === "network") return "Não conseguimos buscar <code>" + esc(error.url) + "</code>. Se a página foi aberta direto do disco, sirva a pasta por HTTP.";
    if (error.kind === "format") return "O arquivo <code>" + esc(error.url) + "</code> não tem o formato esperado" + (error.detail ? " (<code>" + esc(error.detail) + "</code>)" : "") + ".";
    return "O servidor respondeu com erro ao buscar <code>" + esc(error.url) + "</code>.";
  }
  return "Algo deu errado ao montar a página.";
}

function dropWall() {
  if (wall) {
    wall.destroy();
    wall = null;
  }
  view.host.innerHTML = "";
}

function tierCounts(udvs) {
  const out = {};
  TIER_ORDER.forEach((t) => {
    out[t] = 0;
  });
  udvs.forEach((u) => {
    out[u.tier] = (out[u.tier] || 0) + 1;
  });
  return out;
}

function focusQuiet(el) {
  try {
    el.focus({ preventScroll: true });
  } catch (error) {
    el.focus();
  }
}

async function showHome(seq) {
  dropWall();
  view.hearing.hidden = true;
  document.title = APP_TITLE;
  if (home) {
    view.home.hidden = false;
    setState("");
    window.scrollTo(0, homeScroll);
    return;
  }
  view.home.hidden = true;
  setState(loadingHtml("Carregando as matérias…"));
  try {
    const index = await loadIndex();
    if (seq !== routeSeq) return;
    siteIndex = index;
    home = createHome(
      { q: $("#home-q"), sort: $("#home-sort"), lucky: $("#home-lucky"), count: $("#home-count"), list: $("#home-list"), empty: $("#home-empty") },
      index,
      (id) => {
        location.hash = "#h" + id;
      },
    );
    setState("");
    view.home.hidden = false;
  } catch (error) {
    if (seq !== routeSeq) return;
    setState(errorHtml("Não conseguimos carregar a lista de matérias", [describe(error), dataHelp()], "Tentar de novo"));
  }
}

function navHtml(id) {
  if (!home) return "";
  const list = home.ordered();
  const at = list.findIndex((h) => h.id === id);
  const prev = at > 0 ? list[at - 1] : null;
  const next = at >= 0 && at < list.length - 1 ? list[at + 1] : null;
  return (
    (prev ? '<a class="btn" href="#h' + prev.id + '" title="' + esc(prev.title) + '"><span aria-hidden="true">&larr;</span><span>Anterior<span class="nav-long">, mais antiga</span></span></a>' : "") +
    '<button type="button" class="btn" data-lucky>Sortear outra</button>' +
    (next ? '<a class="btn" href="#h' + next.id + '" title="' + esc(next.title) + '"><span>Próxima<span class="nav-long">, mais recente</span></span><span aria-hidden="true">&rarr;</span></a>' : "")
  );
}

async function showHearing(id, seq) {
  if (!view.home.hidden) homeScroll = window.scrollY;
  dropWall();
  view.home.hidden = true;
  view.hearing.hidden = true;
  setState(loadingHtml("Abrindo a audiência " + id + "…"));
  window.scrollTo(0, 0);
  if (!home) {
    try {
      const index = await loadIndex();
      if (seq !== routeSeq) return;
      siteIndex = index;
      home = createHome(
        { q: $("#home-q"), sort: $("#home-sort"), lucky: $("#home-lucky"), count: $("#home-count"), list: $("#home-list"), empty: $("#home-empty") },
        index,
        (hid) => {
          location.hash = "#h" + hid;
        },
      );
    } catch (error) {
      home = null;
    }
  }
  if (siteIndex && !siteIndex.hearings.some((h) => h.id === id)) {
    setState(errorHtml("Não achamos a audiência " + id, ["A audiência " + id + " não está na lista de matérias desta exportação (<code>data/index.json</code>).", "Volte para a lista e escolha outra matéria."], null, true));
    return;
  }
  let H;
  try {
    H = await loadHearing(id, siteIndex ? siteIndex.run : null);
  } catch (error) {
    if (seq !== routeSeq) return;
    const missing = error instanceof DataError && error.kind === "missing";
    setState(errorHtml(missing ? "Não achamos a audiência " + id : "Não conseguimos abrir a audiência " + id, [describe(error), missing ? "Volte para a lista e escolha outra matéria." : dataHelp()], missing ? null : "Tentar de novo", true));
    return;
  }
  if (seq !== routeSeq) return;
  if (!H.udvs.length) {
    setState(errorHtml("A audiência " + id + " não pode ser mostrada", ["Esta matéria não tem afirmações atribuídas a participantes."], null, true));
    return;
  }
  const date = fmtDate(H.hearing.article_date);
  const headline = H.hearing.materia.split(/\n/).map((s) => s.replace(/\s+/g, " ").trim()).find(Boolean) || H.hearing.assunto.replace(/\s+/g, " ").trim();
  const tiers = tierCounts(H.udvs);
  hv.k.textContent = (date ? "Matéria de " + date + " · " : "") + "Audiência " + H.hearing.id;
  hv.title.textContent = headline;
  hv.sum.innerHTML = '<span class="dots" aria-hidden="true">' + dotsHtml(tiers) + "</span>" + esc(statementsSentence(H.udvs.length, H.people.length) + ": " + bucketSentence(tiers));
  hv.nav.innerHTML = navHtml(H.hearing.id);
  document.title = (date ? "Matéria de " + date : "Audiência " + id) + " · " + APP_TITLE;
  setState("");
  view.hearing.hidden = false;
  try {
    wall = createWall(view.host, H);
  } catch (error) {
    dropWall();
    view.hearing.hidden = true;
    setState(errorHtml("Não conseguimos montar a rede da audiência " + id, [describe(error)], "Tentar de novo"));
    if (window.console) console.warn(error);
    return;
  }
  focusQuiet(hv.title);
}

function route() {
  const seq = ++routeSeq;
  const id = hashToken(location.hash);
  if (id == null) {
    if (location.hash && location.hash !== "#") history.replaceState(null, "", location.pathname + location.search);
    showHome(seq);
    return;
  }
  showHearing(id, seq);
}

document.addEventListener("click", (e) => {
  const t = e.target;
  if (!t || !t.closest) return;
  if (t.closest("[data-skip]")) {
    e.preventDefault();
    const target = view.hearing.hidden ? $("#home-h") : hv.title;
    focusQuiet(target && !target.closest("[hidden]") ? target : $("#main"));
    return;
  }
  if (t.closest("[data-retry]")) {
    route();
    return;
  }
  if (t.closest("[data-lucky]") && home) {
    const list = home.ordered();
    const cur = hashToken(location.hash);
    const pool = list.filter((h) => h.id !== cur);
    if (pool.length) location.hash = "#h" + pool[Math.floor(Math.random() * pool.length)].id;
    return;
  }
  const a = t.closest('a[href="#"]');
  if (a) {
    e.preventDefault();
    if (location.hash && location.hash !== "#") location.hash = "";
    else route();
  }
});

window.addEventListener("hashchange", route);
route();
