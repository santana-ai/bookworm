import { bucketSentence, NOT_CHECKED, NOT_CHECKED_SHORT, statementsSentence, TIER_ORDER } from "./copy.js";
import { DataError, loadActors, loadHearing, loadIndex, loadProfile } from "./data.js";
import { createAtlas } from "./atlas.js";
import { caseToken, renderCase } from "./case.js";
import { createHome, dotsHtml } from "./home.js";
import { passageToken, profileHash, profileToken, renderProfile } from "./profile.js";
import { esc, fmtDate, hashToken } from "./text.js";
import { createWall } from "./wall/wall.js";

const $ = (sel) => document.querySelector(sel);
const view = { home: $("#home"), hearing: $("#hearing"), kase: $("#case"), profile: $("#profile"), atlas: $("#atlas"), state: $("#state"), host: $("#wall-host") };
const hv = { k: $("[data-hv-k]"), title: $("#hv-title"), sum: $("[data-hv-sum]"), nav: $("[data-hv-nav]"), prof: $("[data-hv-prof]") };
const crumbsEl = $("[data-crumbs]");
const MAP_HASH = "#arquivo";
const APP_TITLE = "Rede de barbantes";
let home = null;
let siteIndex = null;
let homeScroll = 0;
let wall = null;
let routeSeq = 0;
let resume = null;
let actorsData;
let lastHearing = null;
let lastRoute = "";

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

function dropCase() {
  view.kase.hidden = true;
  view.kase.innerHTML = "";
  view.profile.hidden = true;
  view.profile.innerHTML = "";
  view.atlas.hidden = true;
  view.atlas.innerHTML = "";
}

function crumb(label, href) {
  return href ? '<a href="' + esc(href) + '">' + esc(label) + "</a>" : '<span aria-current="page">' + esc(label) + "</span>";
}

function setCrumbs(parts) {
  const items = [["Mapa do caso", parts.length ? MAP_HASH : null]].concat(parts);
  crumbsEl.innerHTML = "<ol>" + items.map((p) => "<li>" + crumb(p[0], p[1]) + "</li>").join("") + "</ol>";
  document.querySelector("[data-map-tab]").classList.toggle("is-here", !parts.length);
}

async function ensureActors(seq) {
  if (actorsData !== undefined) return actorsData;
  await ensureHome(seq);
  try {
    actorsData = await loadActors(siteIndex ? siteIndex.run : null);
  } catch (error) {
    actorsData = null;
    if (window.console) console.warn(error);
  }
  return actorsData;
}

function actorBySlug(slug) {
  return actorsData ? actorsData.actors.find((a) => a.slug === slug) || null : null;
}

function profileOfUdv(udvId) {
  const slug = actorsData && actorsData.udvs[udvId];
  const a = slug ? actorBySlug(slug) : null;
  return a ? { href: profileHash(a.slug), name: a.name, slug: a.slug } : null;
}

function hearingProfiles(id) {
  const map = actorsData ? actorsData.people[String(id)] || {} : {};
  const seen = new Set();
  const out = [];
  Object.keys(map).forEach((name) => {
    const slug = map[name];
    if (seen.has(slug)) return;
    seen.add(slug);
    const a = actorBySlug(slug);
    if (a) out.push(a);
  });
  return out;
}

async function showHome(seq) {
  dropWall();
  dropCase();
  resume = null;
  view.hearing.hidden = true;
  document.title = APP_TITLE;
  setCrumbs([["Matérias", null]]);
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

async function ensureHome(seq) {
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
}

async function openHearing(id, seq) {
  if (!view.home.hidden) homeScroll = window.scrollY;
  dropWall();
  dropCase();
  view.home.hidden = true;
  view.hearing.hidden = true;
  setState(loadingHtml("Abrindo a audiência " + id + "…"));
  window.scrollTo(0, 0);
  await ensureHome(seq);
  if (seq !== routeSeq) return null;
  if (siteIndex && !siteIndex.hearings.some((h) => h.id === id)) {
    setState(errorHtml("Não achamos a audiência " + id, ["A audiência " + id + " não está na lista de matérias desta exportação (<code>data/index.json</code>).", "Volte para a lista e escolha outra matéria."], null, true));
    return null;
  }
  let H;
  try {
    H = await loadHearing(id, siteIndex ? siteIndex.run : null);
  } catch (error) {
    if (seq !== routeSeq) return null;
    const missing = error instanceof DataError && error.kind === "missing";
    setState(errorHtml(missing ? "Não achamos a audiência " + id : "Não conseguimos abrir a audiência " + id, [describe(error), missing ? "Volte para a lista e escolha outra matéria." : dataHelp()], missing ? null : "Tentar de novo", true));
    return null;
  }
  if (seq !== routeSeq) return null;
  return H;
}

async function showHearing(id, seq, passage) {
  const H = await openHearing(id, seq);
  if (!H) return;
  await ensureActors(seq);
  if (seq !== routeSeq) return;
  lastHearing = id;
  setCrumbs([["Audiência " + id, null]]);
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
  const profs = hearingProfiles(H.hearing.id);
  hv.prof.hidden = !profs.length;
  hv.prof.innerHTML = profs.length
    ? "<span>Perfis de quem fala aqui:</span> " + profs.map((a) => '<a class="hv-prof-a" href="' + profileHash(a.slug) + '">' + esc(a.name) + "</a>").join("")
    : "";
  document.title = (date ? "Matéria de " + date : "Audiência " + id) + " · " + APP_TITLE;
  setState("");
  view.hearing.hidden = false;
  try {
    wall = createWall(view.host, H);
    if (passage) {
      if (!wall.openPassage(passage)) setState(errorHtml("Não achamos o trecho pedido", ["As posições " + passage.start + " a " + passage.end + " do turno " + (passage.turn + 1) + " não existem na audiência " + id + "."], null, false));
    } else if (resume && resume.id === H.hearing.id && resume.j < H.udvs.length) wall.openStatement(resume.j);
    resume = null;
  } catch (error) {
    dropWall();
    view.hearing.hidden = true;
    setState(errorHtml("Não conseguimos montar a rede da audiência " + id, [describe(error)], "Tentar de novo"));
    if (window.console) console.warn(error);
    return;
  }
  focusQuiet(hv.title);
}

async function showCase(id, n, seq) {
  const H = await openHearing(id, seq);
  if (!H) return;
  await ensureActors(seq);
  if (seq !== routeSeq) return;
  lastHearing = id;
  setCrumbs([["Audiência " + id, "#h" + id], ["Pasta da afirmação " + n, null]]);
  if (!(n >= 1 && n <= H.udvs.length)) {
    setState(errorHtml("Não achamos a afirmação " + n + " da audiência " + id, ["A audiência " + id + " tem " + H.udvs.length + " afirmações.", '<a href="#h' + id + '">Voltar à audiência</a>.'], null, true));
    return;
  }
  let shown;
  try {
    shown = renderCase(view.kase, H, n, profileOfUdv);
  } catch (error) {
    dropCase();
    setState(errorHtml("Não conseguimos montar a pasta da afirmação " + n, [describe(error)], "Tentar de novo", true));
    if (window.console) console.warn(error);
    return;
  }
  resume = { id, j: n - 1 };
  document.title = shown.title + " · " + APP_TITLE;
  setState("");
  view.kase.hidden = false;
  focusQuiet(shown.heading);
}

async function showProfile(slug, seq) {
  dropWall();
  dropCase();
  view.home.hidden = true;
  view.hearing.hidden = true;
  setState(loadingHtml("Abrindo o perfil…"));
  window.scrollTo(0, 0);
  const actors = await ensureActors(seq);
  if (seq !== routeSeq) return;
  if (!actors) {
    setState(errorHtml("Esta exportação não tem perfis", ["O arquivo <code>data/actors.json</code> não existe ou não pôde ser lido. Gere os dados com <code>bookworm export-site --profiles</code>; o comando completo está em <code>web/README.md</code>."], null, true));
    return;
  }
  const a = actorBySlug(slug);
  if (!a) {
    setState(errorHtml("Não achamos o perfil pedido", ["Nenhum ator com perfil tem o endereço <code>" + esc(slug) + "</code>.", '<a href="' + MAP_HASH + '">Ver o mapa do caso</a>.'], null, true));
    return;
  }
  let P;
  try {
    P = await loadProfile(slug);
  } catch (error) {
    if (seq !== routeSeq) return;
    setState(errorHtml("Não conseguimos abrir o perfil de " + a.name, [describe(error), dataHelp()], "Tentar de novo", true));
    return;
  }
  if (seq !== routeSeq) return;
  const from = lastHearing != null && P.hearings.some((h) => h.id === lastHearing) ? lastHearing : null;
  setCrumbs((from != null ? [["Audiência " + from, "#h" + from]] : []).concat([[a.name, null], ["Perfil", null]]));
  let shown;
  try {
    shown = renderProfile(view.profile, P, { runLabel: "rodada " + P.provenance.run });
  } catch (error) {
    dropCase();
    setState(errorHtml("Não conseguimos montar o perfil de " + a.name, [describe(error)], "Tentar de novo", true));
    if (window.console) console.warn(error);
    return;
  }
  document.title = shown.title + " · " + APP_TITLE;
  setState("");
  view.profile.hidden = false;
  focusQuiet(shown.heading);
}

async function showAtlas(seq) {
  dropWall();
  dropCase();
  view.home.hidden = true;
  view.hearing.hidden = true;
  setCrumbs([]);
  setState(loadingHtml("Abrindo o mapa do caso…"));
  window.scrollTo(0, 0);
  await ensureHome(seq);
  const actors = await ensureActors(seq);
  if (seq !== routeSeq) return;
  if (!siteIndex) {
    setState(errorHtml("Não conseguimos carregar a lista de matérias", [dataHelp()], "Tentar de novo"));
    return;
  }
  createAtlas(view.atlas, siteIndex, actors);
  document.title = "Mapa do caso · " + APP_TITLE;
  setState("");
  view.atlas.hidden = false;
  focusQuiet($("#at-title"));
}

function route() {
  const seq = ++routeSeq;
  const hash = location.hash;
  if (hash !== MAP_HASH) lastRoute = hash;
  if (hash === MAP_HASH) {
    showAtlas(seq);
    return;
  }
  const slug = profileToken(hash);
  if (slug) {
    showProfile(slug, seq);
    return;
  }
  const passage = passageToken(hash);
  if (passage) {
    showHearing(passage.id, seq, passage);
    return;
  }
  const kase = caseToken(location.hash);
  if (kase) {
    showCase(kase.id, kase.n, seq);
    return;
  }
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
    const target = !view.kase.hidden ? $("#cs-title") : !view.profile.hidden ? $("#pf-title") : !view.atlas.hidden ? $("#at-title") : view.hearing.hidden ? $("#home-h") : hv.title;
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
  const jump = t.closest("[data-jump]");
  if (jump) {
    const el = document.getElementById(jump.dataset.jump);
    if (el) {
      el.scrollIntoView({ block: "center", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
      focusQuiet(el);
      el.classList.remove("is-flash");
      void el.offsetWidth;
      el.classList.add("is-flash");
    }
    return;
  }
  const a = t.closest('a[href="#"]');
  if (a) {
    e.preventDefault();
    if (location.hash && location.hash !== "#") location.hash = "";
    else route();
  }
});

function typing(el) {
  if (!el || el.nodeType !== 1) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
}

document.addEventListener("keydown", (e) => {
  if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || typing(e.target)) return;
  if (e.key === "m" || e.key === "M") {
    e.preventDefault();
    if (location.hash === MAP_HASH) location.hash = lastRoute && lastRoute !== MAP_HASH ? lastRoute : "";
    else location.hash = MAP_HASH;
    return;
  }
  if (e.key === "/" && !view.atlas.hidden) {
    const q = $("#at-q");
    if (q) {
      e.preventDefault();
      q.focus();
    }
  }
});

window.addEventListener("hashchange", route);
route();
