import { NOT_CHECKED } from "../copy.js";
import { clamp, countLabel, esc, firstName, fmtInt, fmtScore, joinPt, markText, plural, shorten, tno } from "../text.js";

export const stId = (i) => "st" + i;
export const peId = (g) => "pe" + g.gi;
export const cadId = (g) => "cad" + g.gi;
export const paId = (k) => "pa" + k;

const PASSAGE_MAX = 560;
const SHEET_MAX = { wide: { quote: 420, sent: 560 }, tall: { quote: 200, sent: 240 } };

function hitWordsHtml(text, words, from, n, cls) {
  if (from < 0 || !n) return esc(text);
  let out = "";
  let last = 0;
  for (let j = from; j < Math.min(words.length, from + n); j++) {
    out += esc(text.slice(last, words[j].s)) + '<span class="' + (cls || "wl-w") + '">' + esc(text.slice(words[j].s, words[j].e)) + "</span>";
    last = words[j].e;
  }
  return out + esc(text.slice(last));
}

function clipWindow(text, words, from, n, max) {
  if (text.length <= max) return { a: 0, b: text.length };
  let a = 0;
  if (from >= 0 && words[from]) {
    const hs = words[from].s;
    const he = words[Math.min(words.length - 1, from + Math.max(0, n - 1))].e;
    a = clamp(Math.round((hs + he) / 2 - max / 2), 0, text.length - max);
    const sp = text.indexOf(" ", a);
    if (a > 0 && sp >= 0 && sp < a + 30) a = sp + 1;
  }
  let b = Math.min(text.length, a + max);
  const sp2 = text.lastIndexOf(" ", b);
  if (b < text.length && sp2 > b - 30) b = sp2;
  return { a, b };
}

function clippedHitHtml(text, words, from, n, cls, max) {
  const win = clipWindow(text, words, from, n, max);
  if (win.a === 0 && win.b === text.length) return hitWordsHtml(text, words, from, n, cls);
  const inside = [];
  words.forEach((wd, j) => {
    if (wd.s >= win.a && wd.e <= win.b) inside.push({ w: wd.w, s: wd.s - win.a, e: wd.e - win.a, j });
  });
  const f = inside.findIndex((x) => x.j === from);
  const body = hitWordsHtml(text.slice(win.a, win.b), inside, f, f < 0 ? 0 : n, cls);
  return (win.a > 0 ? "[…] " : "") + body + (win.b < text.length ? " […]" : "");
}

export function installObjects(w) {
  const M = w.M;
  const { S, GROUPS, PAS, article } = M;
  w.OBJ = {};
  w.OBJ_LIST = [];

  function mkObj(id, type, html, label) {
    const el = document.createElement("div");
    el.className = "wl-o wl-o-" + type;
    el.dataset.id = id;
    el.innerHTML =
      '<div class="wl-o-in">' + html + "</div>" +
      (label ? '<button type="button" class="wl-hit" data-pull="' + id + '" tabindex="-1" aria-label="' + esc(label) + '"></button>' : "");
    w.els.objs.appendChild(el);
    const o = { id, type, el, inn: el.firstChild, x: 0, y: 0, w: 100, h: 60, r: 0, lift: 0 };
    w.OBJ[id] = o;
    w.OBJ_LIST.push(o);
    return o;
  }

  function artParaHtml(pi) {
    let ranges = [];
    S.forEach((s) => {
      if (s.spot.pi === pi) ranges = ranges.concat(s.spot.ranges);
    });
    return markText(article.body[pi], ranges);
  }

  function buildArticle() {
    const h =
      '<div class="wl-art-paper"></div><span class="wl-pin is-l"></span><span class="wl-pin is-r"></span>' +
      '<div class="wl-near"><div class="wl-art-mast"><p class="wl-art-k"><span>' + (M.DATE ? "Matéria de " + esc(M.DATE) : "Matéria") + "</span><span>Audiência " + esc(M.H.hearing.id) + "</span></p>" +
      '<p class="wl-art-hed">' + esc(article.headline) + "</p>" + (article.subhead ? '<p class="wl-art-sub">' + esc(article.subhead) + "</p>" : "") + "</div>" +
      article.body.map((p, pi) => '<p class="wl-art-p" data-pi="' + pi + '">' + artParaHtml(pi) + "</p>").join("") +
      (article.credits.length ? '<p class="wl-art-p is-cred" data-pi="cred">' + article.credits.map(esc).join("<br>") + "</p>" : "") + "</div>" +
      '<div class="wl-far"><span class="wl-fl">A matéria<span class="wl-sub">' + (M.DATE ? " de " + esc(M.DATE) : "") + '</span></span><span class="wl-fl wl-sub wl-art-fhed">' + esc(shorten(article.headline, 70)) + "</span></div>";
    mkObj("art", "art", h, "A matéria" + (M.DATE ? " de " + M.DATE : "") + ": " + article.headline);
  }

  function rulerHtml(sc) {
    const c = clamp(M.CUT, 0, 1) * 100;
    const v = clamp(sc, 0, 1) * 100;
    return (
      '<div class="wl-rul" aria-hidden="true"><div class="wl-rul-bar"><span class="wl-rul-zone" style="left:' + c.toFixed(1) + '%"></span><span class="wl-rul-cut" style="left:' + c.toFixed(1) + '%"></span><span class="wl-rul-mk" style="left:' + v.toFixed(1) + '%"></span></div>' +
      '<div class="wl-rul-ax"><span style="left:0">0</span><span style="left:' + c.toFixed(1) + '%">' + fmtScore(M.CUT) + '</span><span style="left:100%">1</span></div><p class="wl-rul-l">nota de 0 a 1</p></div>'
    );
  }

  function stResHtml(s) {
    let left = "";
    if (s.isQuote && s.quote) left = '<p class="wl-qn">' + quoteCount(s.quote) + " das aspas, iguais na fala</p>";
    else if (s.ev) left = rulerHtml(s.ev.score);
    const sc = s.ev && !s.isQuote ? "<small>nota " + fmtScore(s.ev.score, M.CUT) + "</small>" : "";
    return '<div class="wl-st-res">' + left + '<div class="wl-ink" data-t="' + s.tier.k + '"><span>' + esc(s.tier.label) + "</span>" + sc + "</div></div>";
  }

  function buildStatement(s) {
    const u = s.u;
    const h =
      '<span class="wl-pin"></span><div class="wl-near"><p class="wl-st-k">Afirmação ' + (s.i + 1) + " · segundo a matéria,</p>" +
      '<p class="wl-st-who">' + esc(u.actor.name) + ':</p><p class="wl-st-prop">' + esc(u.proposition) + "</p>" + stResHtml(s) + "</div>" +
      '<div class="wl-far"><span class="wl-far-row"><span class="wl-no">' + (s.i + 1) + '</span><span class="wl-fl wl-sub">' + esc(firstName(u.actor.name)) + '</span></span><span class="wl-chip wl-sub" data-t="' + s.tier.k + '">' + esc(s.tier.short) + "</span></div>";
    mkObj(stId(s.i), "st", h, "Afirmação " + (s.i + 1) + ", atribuída a " + u.actor.name + ": " + shorten(u.proposition, 90) + ". Puxar este cartão.");
  }

  function personLine(g) {
    if (!g.resolved) return "Não achamos a fala na transcrição";
    return "Fala em " + g.turns.length + " dos " + fmtInt(M.NT) + " turnos";
  }

  function buildPerson(g) {
    const name = g.p.name;
    const role = g.p.role;
    const line = personLine(g);
    const h =
      '<span class="wl-pin"></span><div class="wl-near"><p class="wl-pe-k">' + (g.resolved ? "Quem disse" : "Quem a matéria cita") + '</p><p class="wl-pe-name">' + esc(name) + "</p>" +
      (role ? '<p class="wl-pe-role">' + esc(role) + "</p>" : "") + '<p class="wl-pe-turns">' + esc(line) + "</p></div>" +
      '<div class="wl-far"><span class="wl-fl wl-sub">' + esc(name) + '</span><span class="wl-fl wl-only-tiny">' + esc(firstName(name)) + "</span></div>";
    const o = mkObj(peId(g), "pe", h, name + ", " + line.toLowerCase() + ". Puxar este cartão.");
    if (!g.resolved) o.el.classList.add("is-unres");
  }

  function buildCad(g) {
    let alt = false;
    let prev = null;
    const dashes = g.dashTurn
      .map((t, d) => {
        if (prev !== null && t !== prev) alt = !alt;
        prev = t;
        return "<i data-d=\"" + d + '"' + (alt ? ' class="is-alt"' : "") + "></i>";
      })
      .join("");
    const nS = g.sents.length;
    const each = g.bin > 1 ? "cada tracinho reúne até " + g.bin + " frases" : "cada tracinho é uma frase";
    const nm = g.p.name;
    const h =
      '<span class="wl-pin"></span><div class="wl-near"><p class="wl-cad-k">Falas de</p><p class="wl-cad-h">' + esc(nm) + "</p>" +
      '<p class="wl-cad-n">' + esc(countLabel(g.turns.length, "turno", "turnos") + " · " + countLabel(nS, "frase", "frases")) + "</p>" +
      '<div class="wl-cad-d">' + dashes + '</div><p class="wl-cad-f">' + esc(each) + "</p></div>" +
      '<div class="wl-far"><span class="wl-fl">falas de ' + esc(firstName(nm)) + "</span></div>";
    mkObj(cadId(g), "cad", h, "Caderno com as falas de " + nm + ": " + countLabel(g.turns.length, "turno", "turnos") + ", " + countLabel(nS, "frase", "frases") + ". Puxar este caderno.");
  }

  function passageHtml(pa) {
    let s0 = null;
    pa.sts.forEach((i) => {
      if (!s0 && (S[i].isQuote || (S[i].qsem && S[i].qsem.run))) s0 = S[i];
    });
    if (s0 && s0.isQuote) return clippedHitHtml(pa.text, s0.quote.sw, s0.quote.si, s0.quote.np, "wl-w", PASSAGE_MAX);
    if (s0 && s0.qsem) return clippedHitHtml(pa.text, s0.qsem.sw, s0.qsem.si, s0.qsem.run, "wl-w", PASSAGE_MAX);
    return clippedHitHtml(pa.text, [], -1, 0, "wl-w", PASSAGE_MAX);
  }

  function buildPassage(pa) {
    const speaker = M.speakerName(pa.turn);
    const h =
      '<span class="wl-pin is-l"></span><span class="wl-pin is-r"></span><div class="wl-near"><p class="wl-pa-h"><span>Trecho · turno ' + tno(pa.turn) + "</span><span>" + esc(firstName(speaker)) + "</span></p>" +
      '<p class="wl-pa-t"><mark>' + passageHtml(pa) + '</mark></p><p class="wl-pa-f">Caracteres ' + fmtInt(pa.start) + " a " + fmtInt(pa.end) + " de " + fmtInt(M.T.length) + "</p></div>" +
      '<div class="wl-far"><span class="wl-fl">trecho<span class="wl-sub">, turno ' + tno(pa.turn) + "</span></span></div>";
    const o = mkObj(paId(pa.k), "pa", h, "Trecho do turno " + tno(pa.turn) + ", caracteres " + fmtInt(pa.start) + " a " + fmtInt(pa.end) + ". Puxar este trecho.");
    const rb = document.createElement("button");
    rb.type = "button";
    rb.className = "wl-read";
    rb.dataset.read = String(pa.k);
    rb.textContent = "Ler em volta";
    o.el.appendChild(rb);
  }

  function buildNote() {
    if (!M.otherTurns.length) return;
    const names = M.otherNames;
    const shown = names.slice(0, 5);
    const more = names.length - shown.length;
    const who = more > 0 ? shown.join(", ") + " e mais " + countLabel(more, "pessoa", "pessoas") : joinPt(shown);
    const h =
      '<div class="wl-near"><p class="wl-note-h">Também falam na audiência</p><p class="wl-note-t">' + esc(who) + ", em " + countLabel(M.otherTurns.length, "turno", "turnos") +
      ". Não ligamos essas falas a nenhuma afirmação da matéria.</p></div>" +
      '<div class="wl-far"><span class="wl-fl">Também falam</span></div>';
    mkObj("note", "note", h, null);
  }

  function buildSheet() {
    const o = mkObj("sheet", "sheet", '<span class="wl-pin is-l"></span><span class="wl-pin is-r"></span><div class="wl-sh-in"></div>', null);
    o.el.setAttribute("aria-hidden", "true");
    w.els.sheetIn = o.el.querySelector(".wl-sh-in");
  }

  function quoteCount(qd) {
    return (qd.qi === 0 ? "As " + qd.np + " primeiras" : qd.np) + " das " + qd.nq + " palavras";
  }

  function fillSheet(s) {
    const box = w.els.sheetIn;
    const name = s.u.actor.name;
    let h;
    if (s.isQuote && s.quote) {
      const qd = s.quote;
      const max = SHEET_MAX[w.L.name];
      h =
        '<p class="wl-qm-k">Na matéria, entre aspas</p><p class="wl-qm-art">“' + clippedHitHtml(qd.text, qd.qw, qd.qi, qd.np, "wl-qw", max.quote) + "”</p>" +
        '<p class="wl-qm-k">Na fala de ' + esc(name) + ", turno " + tno(s.ev.speaker_turn) + '</p><p class="wl-qm-sent">' + clippedHitHtml(s.ev.text, qd.sw, qd.si, qd.np, "wl-qw", max.sent) + "</p>" +
        '<p class="wl-qm-n">' + quoteCount(qd) + " entre aspas aparecem na fala, seguidas e iguais</p>";
    } else {
      const c = s.u.candidates.slice(0, 8);
      h =
        '<p class="wl-sh-h">Busca pelo sentido · <b>' + esc(countLabel(s.nSent, "frase", "frases")) + " de " + esc(name) + "</b></p>" +
        (c.length
          ? '<ol class="wl-slips">' +
            c.map((x, r) => '<li data-r="' + r + '"' + (s.ev && x.start === s.ev.start_char ? ' class="is-best"' : "") + "><b>" + fmtScore(x.score, M.CUT) + "</b><span>" + esc(x.text) + "</span><em>turno " + tno(x.turn) + "</em></li>").join("") +
            "</ol>"
          : '<p class="wl-sh-note">Não havia frase de ' + esc(name) + " com pelo menos 4 palavras para comparar.</p>") +
        (c.length ? '<p class="wl-sh-note">' + plural(c.length, "A mais parecida", "As " + c.length + " mais parecidas") + ", da nota maior para a menor. Quanto mais perto de 1, mais parecido o sentido; a nota não é a chance de estar certo.</p>" : "");
    }
    box.innerHTML = h;
  }

  buildArticle();
  S.forEach(buildStatement);
  GROUPS.forEach(buildPerson);
  GROUPS.forEach((g) => {
    if (g.resolved) buildCad(g);
  });
  PAS.forEach(buildPassage);
  buildNote();
  buildSheet();

  w.disclaimer = "Dados da audiência " + M.H.hearing.id + " (rodada " + M.RUN.name + ")" + ". " + NOT_CHECKED;
  Object.assign(w, { mkObj, fillSheet, quoteCount });
}
