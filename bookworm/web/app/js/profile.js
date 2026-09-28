import { CLAIM_STATES, claimState, DISCLOSE_HINTS, PROFILE_BADGES, PROFILE_MATCH_NOTE, PROFILE_NOTE, PROFILE_NOTE_SHORT, PROFILE_TOP_NOTE, PROFILE_TOP_TITLE, SPLIT_NAMES, tierOf, UDV_RULES } from "./copy.js";
import { bindDisclose, discloseBar, discloseEnd } from "./disclose.js";
import { caseHash } from "./case.js";
import { countLabel, esc, fmtCut, fmtDate, fmtInt, fmtScore, joinPt, plural, shorten, tno } from "./text.js";

const CLAIM_SHORT = 220;
const QUOTE_MAX = 260;
const LEFT_SECTIONS = ["posições"];
const INITIAL_SKIP = new Set(["de", "da", "do", "das", "dos", "e", "dep.", "prof.", "profa.", "dr.", "dra.", "sr.", "sra."]);
const DOT = { q: "q", h: "h", w: "w", n: "u", u: "u" };

const ICON = {
  cal: '<path d="M4 6h16v14H4z"/><path d="M4 10h16M8 3v5M16 3v5"/>',
  flag: '<path d="M5 21V4"/><path d="M5 4h11l-2 4 2 4H5"/>',
  scale: '<path d="M12 4v16M7 20h10"/><path d="M5 8h14"/><path d="M5 8l-3 6h6zM19 8l-3 6h6z"/>',
  link: '<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1"/><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/>',
  chip: '<path d="M7 7h10v10H7z"/><path d="M10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4"/>',
  tag: '<path d="M3 12V4h8l10 10-8 8z"/><circle cx="7.5" cy="8.5" r="1.3"/>',
  doc: '<path d="M6 3h9l4 4v14H6z"/><path d="M15 3v4h4M9 12h7M9 16h7"/>',
  hand: '<path d="M4 20l4-1 11-11-3-3L5 16z"/><path d="M14 7l3 3"/>',
};

export function profileHash(slug) {
  return "#perfil/" + slug;
}

const PROFILE_RE = /^#?perfil\/([a-z0-9-]+)(?:\/item-(\d+))?$/;

export function profileToken(value) {
  const m = String(value || "").match(PROFILE_RE);
  return m ? m[1] : null;
}

export function profileItemToken(value) {
  const m = String(value || "").match(PROFILE_RE);
  return m && m[2] ? Number(m[2]) : null;
}

export function passageHash(p) {
  return "#h" + p.hearing_id + "-t" + p.turn + "-p" + p.start + "-" + p.end;
}

export function passageToken(value) {
  const m = String(value || "").match(/^#?h(\d+)-t(\d+)-p(\d+)-(\d+)$/);
  return m ? { id: Number(m[1]), turn: Number(m[2]), start: Number(m[3]), end: Number(m[4]) } : null;
}

export function icon(name) {
  return '<svg class="pf-ic" viewBox="0 0 24 24" aria-hidden="true" focusable="false">' + ICON[name] + "</svg>";
}

export function initials(name) {
  const words = String(name)
    .replace(/\([^)]*\)/g, " ")
    .split(/\s+/)
    .filter((w) => w && !INITIAL_SKIP.has(w.toLowerCase()) && /\p{L}/u.test(w));
  if (!words.length) return "?";
  const pick = words.length === 1 ? [words[0]] : [words[0], words[words.length - 1]];
  return pick.map((w) => w.match(/\p{L}/u)[0].toUpperCase()).join("");
}

function pct(n, total) {
  return total ? ((100 * n) / total).toFixed(1) + "%" : "0%";
}

function verifierText(v, cut) {
  if (!v) return "verificador não calculado";
  const side = cut == null ? "" : v.supported ? ", acima do corte " + fmtCut(cut) : ", abaixo do corte " + fmtCut(cut);
  return "verificador " + fmtScore(v.probability, cut == null ? undefined : cut) + side;
}

function inkHtml(u, cut) {
  const t = tierOf(u.tier);
  return '<div class="pf-ink" data-t="' + t.k + '"><span>' + esc(t.label) + "</span><small>" + esc(verifierText(u.verifier, cut)) + "</small></div>";
}

function period(hearings) {
  const dates = hearings.map((h) => h.article_date).filter(Boolean).sort();
  if (!dates.length) return "";
  if (dates[0] === dates[dates.length - 1]) return fmtDate(dates[0]);
  return fmtDate(dates[0]) + " a " + fmtDate(dates[dates.length - 1]);
}

function allClaims(P) {
  const out = [];
  P.sections.forEach((s) => {
    s.claims.forEach((c) => {
      out.push({ c, section: s.title, n: out.length + 1 });
    });
  });
  return out;
}

function pickQuote(P) {
  const withEv = P.udvs.filter((u) => u.evidence);
  if (withEv.length) {
    const rank = (u) => (u.in_profile ? 2 : 0) + (u.verifier ? u.verifier.probability : 0);
    const best = withEv.slice().sort((a, b) => rank(b) - rank(a))[0];
    return { text: best.evidence.text, hearing: best.hearing_id, turn: best.evidence.turn, from: "evidência da afirmação " + best.n + " da matéria" + (best.in_profile ? "" : ", numa audiência que o perfil não leu") };
  }
  const passages = allClaims(P).map((x) => x.c.passage).filter(Boolean);
  if (!passages.length) return null;
  const best = passages.slice().sort((a, b) => b.score - a.score)[0];
  return { text: best.text, hearing: best.hearing_id, turn: best.turn, from: "frase mais parecida com um item do perfil" };
}

function factRow(label, value) {
  return value ? '<tr><th scope="row">' + esc(label) + "</th><td>" + value + "</td></tr>" : "";
}

function headHtml(P, ctx) {
  const a = P.actor;
  const pv = P.provenance;
  const heard = P.hearings;
  const turns = heard.reduce((s, h) => s + h.turns, 0);
  const q = pickQuote(P);
  const others = a.article_names.filter((n) => n !== a.name);
  const rows =
    factRow("Na matéria", a.role ? esc(a.role) : "") +
    factRow("Na transcrição", a.party_uf.length ? esc(a.party_uf.join("; ")) : "") +
    factRow("Audiências", esc(countLabel(heard.length, "audiência", "audiências")) + (period(heard) ? ", " + esc(period(heard)) : "")) +
    factRow("Falas", esc(countLabel(turns, "turno", "turnos"))) +
    factRow("Perfil lido de", esc(countLabel(pv.n_hearings, "audiência", "audiências") + " de treino, " + countLabel(pv.n_statements, "turno", "turnos"))) +
    factRow("Foco", esc(focusLine(P))) +
    (others.length ? factRow("Nome na matéria", esc(joinPt(others))) : "");
  return (
    '<article class="pf-sheet pf-head" aria-labelledby="pf-title"><span class="pf-tape is-a" aria-hidden="true"></span><span class="pf-tape is-b" aria-hidden="true"></span>' +
    '<figure class="pf-pola" role="img" aria-label="Sem foto: as iniciais ' + esc(initials(a.name)) + '"><div class="pf-pola-img" aria-hidden="true"><svg class="pf-sil" viewBox="0 0 100 100"><circle cx="50" cy="38" r="17"/><path d="M16 96c3-22 17-34 34-34s31 12 34 34"/></svg><span>' + esc(initials(a.name)) + '</span></div><figcaption>' + esc(a.name) + "<small>sem foto</small></figcaption></figure>" +
    '<div class="pf-head-main">' +
    '<p class="pf-k">Dossiê de ator · ' + esc(ctx.runLabel) + "</p>" +
    '<h2 class="pf-name"><span>Perfil:</span> ' + esc(a.name) + "</h2>" +
    (q
      ? '<blockquote class="pf-quote"><p>“' + esc(shorten(q.text, QUOTE_MAX)) + '”</p><footer>Trecho da transcrição, audiência ' + q.hearing + ", turno " + tno(q.turn) + " (" + esc(q.from) + ")</footer></blockquote>"
      : "") +
    '<table class="pf-facts"><tbody>' + rows + "</tbody></table>" +
    "</div>" +
    '<p class="pf-gen"><span>Gerado por modelo</span><small>confira a evidência</small></p>' +
    "</article>"
  );
}

function focusLine(P) {
  const c = P.counts;
  return countLabel(c.claims, "item", "itens") + " no perfil: " + c.with_udv + " " + CLAIM_STATES.udv.short + ", " + c.passage_only + " " + CLAIM_STATES.passage.short + ", " + c.without_evidence + " " + CLAIM_STATES.none.short;
}

function notesHtml(P) {
  const pv = P.provenance;
  const when = fmtDate(pv.generated_at.slice(0, 10));
  const items = [
    "escrito por " + pv.model.split("/").pop(),
    "rodada " + pv.run + ", prompt " + pv.prompt_version,
    when ? "gerado em " + when : "",
    fmtInt(pv.input_tokens) + " tokens lidos, " + fmtInt(pv.output_tokens) + " escritos",
    "gerado a partir das falas; confira a evidência",
  ].filter(Boolean);
  return (
    '<aside class="pf-notes" aria-labelledby="pf-notes-h"><span class="pf-tape is-c" aria-hidden="true"></span>' +
    '<h2 class="pf-notes-h" id="pf-notes-h">' + icon("hand") + "Proveniência</h2>" +
    "<ul>" + items.map((t) => "<li>" + esc(t) + "</li>").join("") + "</ul></aside>"
  );
}

function diagramHtml(P) {
  const pv = P.provenance;
  const c = P.counts;
  const box = (x, w, top, bottom) =>
    '<g><rect x="' + x + '" y="10" width="' + w + '" height="46" rx="3"/><text x="' + (x + w / 2) + '" y="29" class="b">' + esc(top) + '</text><text x="' + (x + w / 2) + '" y="46">' + esc(bottom) + "</text></g>";
  const arrow = (x) => '<path class="ar" d="M' + x + " 33h14m-5-5 5 5-5 5" + '"/>';
  return (
    '<svg class="pf-diag" viewBox="0 0 400 66" role="img" aria-label="Falas das audiências de treino, depois o modelo, depois o perfil, depois a conferência por palavras contra as falas e as afirmações das matérias">' +
    box(2, 78, "falas", countLabel(pv.n_statements, "turno", "turnos")) + arrow(84) +
    box(102, 84, "modelo", "LLM") + arrow(190) +
    box(208, 74, "perfil", countLabel(c.claims, "item", "itens")) + arrow(286) +
    box(304, 94, "conferir", "por palavras") +
    "</svg>"
  );
}

function resultsHtml(P) {
  const c = P.counts;
  const row = (k, label, n) =>
    '<li><span class="pf-rb" data-k="' + k + '"><i style="width:' + pct(n, c.claims) + '"></i></span><b>' + fmtInt(n) + "</b><span>" + esc(label) + "</span></li>";
  return (
    '<div class="pf-res"><p class="pf-res-h">Resultados da conferência</p><ul>' +
    row("udv", CLAIM_STATES.udv.label, c.with_udv) +
    row("pas", CLAIM_STATES.passage.label, c.passage_only) +
    row("none", CLAIM_STATES.none.label, c.without_evidence) +
    "</ul></div>"
  );
}

function howHtml(P) {
  return (
    '<section class="pf-sheet pf-how" aria-labelledby="pf-how-h"><span class="pf-tape is-d" aria-hidden="true"></span>' +
    '<h2 class="pf-h" id="pf-how-h">' + icon("chip") + "Como o perfil foi feito</h2>" +
    '<p class="pf-small">Um modelo de linguagem leu as falas desta pessoa nas audiências de treino e escreveu o perfil. Depois, cada item foi comparado, por palavras, com as frases dessas falas e com as afirmações que as matérias atribuem à pessoa.</p>' +
    diagramHtml(P) + resultsHtml(P) +
    "</section>"
  );
}

function sticky(text, cls) {
  return '<p class="pf-sticky' + (cls ? " " + cls : "") + '">' + esc(text) + "</p>";
}

function hearingRows(P) {
  return P.hearings
    .slice()
    .sort((a, b) => (a.article_date || "").localeCompare(b.article_date || "") || a.id - b.id)
    .map((h) => {
      const udvs = P.udvs.filter((u) => u.hearing_id === h.id);
      const split = SPLIT_NAMES[h.split] || h.split || "";
      return (
        '<li class="pf-hr' + (h.in_profile ? " is-read" : "") + '">' +
        '<p class="pf-hr-k"><span>' + (h.article_date ? esc(fmtDate(h.article_date)) : "sem data") + '</span><a href="#h' + h.id + '">Audiência ' + h.id + "</a></p>" +
        '<p class="pf-hr-t">' + esc(h.title) + "</p>" +
        '<p class="pf-hr-m">' + esc(countLabel(h.turns, "turno", "turnos")) + " · " + (h.in_profile ? "<b>lida pelo perfil</b>" : "fora do perfil") + (split ? " (" + esc(split) + ")" : "") + "</p>" +
        (udvs.length
          ? '<p class="pf-hr-u">' + esc(plural(udvs.length, "Pasta:", "Pastas:")) + " " +
            udvs.map((u) => '<a href="' + caseHash(u.hearing_id, u.n) + '" title="' + esc(tierOf(u.tier).label) + '"><i class="dot" data-k="' + DOT[tierOf(u.tier).k] + '" aria-hidden="true"></i>' + u.n + "</a>").join("") + "</p>"
          : "") +
        "</li>"
      );
    })
    .join("");
}

function participationHtml(P) {
  const read = P.hearings.filter((h) => h.in_profile).length;
  return (
    '<section class="pf-card pf-part" aria-labelledby="pf-part-h">' +
    sticky("fala em " + countLabel(P.hearings.length, "audiência", "audiências") + "; o perfil leu " + read, "is-y") +
    '<h2 class="pf-h" id="pf-part-h">' + icon("cal") + "Participação</h2>" +
    '<ol class="pf-hl">' + hearingRows(P) + "</ol></section>"
  );
}

function themesHtml(P) {
  return (
    '<section class="pf-card pf-themes" aria-labelledby="pf-th-h"><h2 class="pf-h" id="pf-th-h">' + icon("tag") + "Temas das audiências</h2>" +
    '<p class="pf-small">O assunto de cada audiência em que a pessoa fala, como está nos metadados.</p><ul class="pf-tags">' +
    P.hearings.map((h) => '<li><a href="#h' + h.id + '">' + esc(h.assunto) + "</a></li>").join("") +
    "</ul></section>"
  );
}

function claimLi(x) {
  const st = claimState(x.c);
  return (
    '<li class="pf-cl" data-s="' + st.k + '"><button type="button" class="pf-cl-n" data-jump="pf-e' + x.n + '" aria-label="Ver a evidência do item ' + x.n + '">' + x.n + "</button>" +
    '<p class="pf-cl-t">' + esc(x.c.text) + '<span class="pf-cl-s">' + esc(st.short) + "</span></p></li>"
  );
}

function sectionCard(title, items, extra) {
  const id = "pf-s-" + items[0].n;
  return (
    '<section class="pf-card pf-sec" aria-labelledby="' + id + '">' + (extra || "") +
    '<h2 class="pf-h" id="' + id + '">' + icon(/posi/i.test(title) ? "flag" : "scale") + esc(title) + "</h2>" +
    '<ol class="pf-cls">' + items.map(claimLi).join("") + "</ol></section>"
  );
}

function sectionsHtml(P) {
  const claims = allClaims(P);
  const bySection = [];
  P.sections.forEach((s) => {
    bySection.push({ title: s.title, items: claims.filter((x) => x.section === s.title && s.claims.indexOf(x.c) >= 0) });
  });
  const left = [];
  const right = [];
  bySection.forEach((s) => {
    if (!s.items.length) return;
    const noUdv = s.items.filter((x) => !x.c.udv).length;
    const note = noUdv ? sticky(noUdv + " de " + s.items.length + " sem UDV ligada", "is-p") : "";
    (LEFT_SECTIONS.indexOf(s.title.toLowerCase()) >= 0 ? left : right).push(sectionCard(s.title, s.items, left.length + right.length === 0 ? note : ""));
  });
  return { left: left.join(""), right: right.join("") };
}

function evidenceItem(x, P, byId, cut) {
  const c = x.c;
  const st = claimState(c);
  const u = c.udv ? byId.get(c.udv.id) : null;
  const pa = c.passage;
  const passage = pa
    ? '<div class="pf-ev-pa"><p class="pf-ev-t"><mark>' + esc(pa.text) + "</mark></p>" +
      '<p class="pf-ev-m">Audiência ' + pa.hearing_id + ", turno " + tno(pa.turn) + ", nota de palavras " + fmtScore(pa.score) +
      (pa.start !== null && pa.end !== null ? ' · <a href="' + passageHash(pa) + '">Ler na transcrição</a>' : "") + "</p></div>"
    : '<div class="pf-ev-pa is-none"><p class="pf-ev-m">Sem frase parecida nas falas que o perfil leu.</p></div>';
  const udv = u
    ? '<div class="pf-ev-u">' + inkHtml(u, cut) +
      '<p class="pf-ev-m">' + esc(UDV_RULES[c.udv.rule]) + ".</p>" +
      '<p class="pf-ev-p">“' + esc(shorten(u.proposition, 180)) + "”</p>" +
      '<a class="pf-go" href="' + caseHash(u.hearing_id, u.n) + '">Pasta da afirmação ' + u.n + ", audiência " + u.hearing_id + "</a></div>"
    : '<div class="pf-ev-u is-none"><p class="pf-ev-m">Nenhuma afirmação da matéria ligada a este item.</p></div>';
  return (
    '<li class="pf-ev-i" id="pf-e' + x.n + '" data-s="' + st.k + '" tabindex="-1">' +
    '<div class="pf-ev-c"><span class="pf-no">' + x.n + '</span><p class="pf-ev-sec">' + esc(x.section) + '</p><p class="pf-ev-cl">' + esc(shorten(c.text, CLAIM_SHORT)) + "</p></div>" +
    passage + udv + "</li>"
  );
}

function evidenceHtml(P) {
  const byId = new Map(P.udvs.map((u) => [u.id, u]));
  const claims = allClaims(P);
  const none = P.counts.without_evidence;
  return (
    '<section class="pf-sheet pf-ev" aria-labelledby="pf-ev-h"><span class="pf-tape is-a" aria-hidden="true"></span><span class="pf-tape is-b" aria-hidden="true"></span>' +
    (none ? sticky(countLabel(none, "item sem frase parecida", "itens sem frase parecida") + " nas falas lidas", "is-g") : "") +
    '<h2 class="pf-h" id="pf-ev-h">' + icon("link") + "Evidência de cada item</h2>" +
    '<p class="pf-small">' + esc(PROFILE_MATCH_NOTE) + "</p>" +
    '<div class="pf-ev-cols" aria-hidden="true"><span>Item do perfil</span><span>Frase mais parecida nas falas</span><span>Afirmação da matéria (UDV)</span></div>' +
    '<ol class="pf-evl">' + claims.map((x) => evidenceItem(x, P, byId, P.verifier_threshold)).join("") + "</ol></section>"
  );
}

function udvsHtml(P) {
  if (!P.udvs.length) {
    return '<section class="pf-card pf-udvs" aria-labelledby="pf-u-h"><h2 class="pf-h" id="pf-u-h">' + icon("doc") + "O que as matérias atribuem</h2><p class=\"pf-small\">Nenhuma afirmação das matérias ficou ligada a esta pessoa pelos turnos de fala.</p></section>";
  }
  const cut = P.verifier_threshold;
  const rows = P.udvs
    .map(
      (u) =>
        '<li class="pf-ur' + (u.in_profile ? "" : " is-out") + '"><p class="pf-ur-k"><a href="' + caseHash(u.hearing_id, u.n) + '">Audiência ' + u.hearing_id + ", afirmação " + u.n + "</a>" + (u.in_profile ? "" : "<span>audiência fora do perfil</span>") + "</p>" +
        '<p class="pf-ur-p">' + esc(u.proposition) + "</p>" + inkHtml(u, cut) + "</li>",
    )
    .join("");
  return (
    '<section class="pf-card pf-udvs" aria-labelledby="pf-u-h"><h2 class="pf-h" id="pf-u-h">' + icon("doc") + "O que as matérias atribuem</h2>" +
    '<p class="pf-small">As afirmações que as matérias atribuem a ' + esc(P.actor.name) + ", ligadas pelos turnos em que a pessoa fala, com o resultado da busca e a nota do verificador de cada uma. Quando a matéria não menciona algo que está no perfil, isso só quer dizer que a matéria não escolheu aquele ponto.</p>" +
    '<ol class="pf-ul">' + rows + "</ol></section>"
  );
}

function firstSentence(text, max) {
  const m = String(text).match(/^(.+?[.!?])(\s|$)/);
  return shorten(m && m[1].length >= 40 ? m[1] : text, max);
}

function synthesisLines(P) {
  const c = P.counts;
  const when = period(P.hearings);
  const withPassage = c.claims - c.without_evidence;
  return [
    when ? "Fala em audiências de " + when + "." : "",
    "O modelo escreveu " + countLabel(c.claims, "item", "itens") + " a partir de " + countLabel(P.provenance.n_hearings, "audiência", "audiências") + "; " +
      fmtInt(withPassage) + " " + plural(withPassage, "tem", "têm") + " frase parecida nas falas e " + fmtInt(c.with_udv) + " " + plural(c.with_udv, "está ligado", "estão ligados") + " a afirmações de matéria.",
  ].filter(Boolean);
}

function topPositions(P) {
  const claims = allClaims(P);
  const pos = claims.filter((x) => /^posi/i.test(x.section));
  const pool = pos.length ? pos : claims;
  const rank = (x) => (x.c.udv ? 0 : x.c.passage ? 1 : 2);
  return pool
    .map((x, i) => ({ x, i }))
    .sort((a, b) => rank(a.x) - rank(b.x) || a.i - b.i)
    .slice(0, 3)
    .map((o) => o.x);
}

function badgeHtml(x) {
  const st = claimState(x.c);
  const label = PROFILE_BADGES[st.k] + (st.k === "pas" ? " (" + fmtScore(x.c.passage.score) + ")" : "");
  return '<button type="button" class="pf-badge" data-s="' + st.k + '" data-jump="pf-e' + x.n + '" aria-label="' + esc(label + ". Ver a evidência do item " + x.n) + '"><i aria-hidden="true"></i>' + esc(label) + "</button>";
}

function summaryHtml(P, ctx) {
  const a = P.actor;
  const read = P.hearings.filter((h) => h.in_profile).length;
  const turns = P.hearings.reduce((n, h) => n + h.turns, 0);
  const top = topPositions(P);
  const stat = (n, label) => '<li class="sm-stat"><b>' + esc(fmtInt(n)) + "</b><span>" + esc(label) + "</span></li>";
  return (
    '<article class="pf-sheet pf-sum" aria-labelledby="pf-title"><span class="pf-tape is-a" aria-hidden="true"></span><span class="pf-tape is-b" aria-hidden="true"></span>' +
    '<figure class="pf-pola" role="img" aria-label="Sem foto: as iniciais ' + esc(initials(a.name)) + '"><div class="pf-pola-img" aria-hidden="true"><svg class="pf-sil" viewBox="0 0 100 100"><circle cx="50" cy="38" r="17"/><path d="M16 96c3-22 17-34 34-34s31 12 34 34"/></svg><span>' + esc(initials(a.name)) + '</span></div><figcaption>' + esc(a.name) + "<small>sem foto</small></figcaption></figure>" +
    '<div class="pf-sum-main">' +
    '<p class="pf-k">Dossiê de ator · ' + esc(ctx.runLabel) + "</p>" +
    '<h1 class="pf-name" id="pf-title" tabindex="-1"><span>Perfil:</span> ' + esc(a.name) + "</h1>" +
    (a.role || a.party_uf.length ? '<p class="pf-role">' + esc(a.role || a.party_uf[0]) + "</p>" : "") +
    '<p class="pf-syn">' + synthesisLines(P).map(esc).join(" ") + "</p>" +
    '<ul class="sm-stats pf-stats" aria-label="Participação">' + stat(P.hearings.length, plural(P.hearings.length, "audiência", "audiências")) + stat(turns, plural(turns, "turno de fala", "turnos de fala")) + stat(read, plural(read, "lida pelo perfil", "lidas pelo perfil")) + "</ul>" +
    '<p class="pf-gen-s"><b>Gerado por modelo.</b> ' + esc(PROFILE_NOTE_SHORT) + "</p>" +
    "</div>" +
    (top.length
      ? '<section class="pf-top3" aria-labelledby="pf-top3-h"><h2 class="pf-h" id="pf-top3-h">' + icon("flag") + esc(PROFILE_TOP_TITLE) + '</h2><ol class="pf-top3-l">' +
        top.map((x) => '<li><span class="pf-no">' + x.n + '</span><div><p class="pf-top3-t">' + esc(firstSentence(x.c.text, 190)) + "</p>" + badgeHtml(x) + "</div></li>").join("") +
        '</ol><p class="pf-small">' + esc(PROFILE_TOP_NOTE) + "</p></section>"
      : "") +
    "</article>"
  );
}

export function renderProfile(host, P, ctx) {
  const cols = sectionsHtml(P);
  host.innerHTML =
    '<div class="pf-desk">' +
    '<div class="pf-lamp" aria-hidden="true"></div>' +
    summaryHtml(P, ctx) +
    discloseBar("profile", "pf-more", DISCLOSE_HINTS.profile) +
    '<div class="pf-more" id="pf-more">' +
    '<p class="note is-compact pf-warn"><b>Gerado por um modelo de linguagem.</b> ' + esc(PROFILE_NOTE) + "</p>" +
    '<div class="pf-top">' + headHtml(P, ctx) + notesHtml(P) + howHtml(P) + "</div>" +
    '<div class="pf-cols"><div class="pf-col">' + participationHtml(P) + themesHtml(P) + '</div><div class="pf-col">' + cols.left + '</div><div class="pf-col">' + cols.right + udvsHtml(P) + "</div></div>" +
    evidenceHtml(P) +
    discloseEnd("profile", "pf-more") +
    "</div>" +
    '<div class="pf-mug" aria-hidden="true"></div>' +
    "</div>";
  bindDisclose(host, "profile");
  return { title: "Perfil: " + P.actor.name, heading: host.querySelector("#pf-title") };
}
