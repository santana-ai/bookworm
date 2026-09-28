import { CASE_NOTE, NO_SECOND_OPINION, questionCopy, SPLIT_GAP } from "./copy.js";
import { LAYA_NAMES } from "./data.js";
import { buildModel } from "./model.js";
import { clamp, countLabel, esc, firstName, fmtDate, fmtInt, fmtScore, joinPt, markText, plural, seqIndex, tno, wordsOf } from "./text.js";

const CONTEXT_SENTENCES = 2;
const CONTEXT_MAX = 600;
const DOT_KEY = { q: "q", h: "h", w: "w", n: "u", u: "u" };

export function caseHash(id, n) {
  return "#h" + id + "-u" + n;
}

export function caseToken(value) {
  const m = String(value || "").match(/^#?h(\d+)-u(\d+)$/);
  return m ? { id: Number(m[1]), n: Number(m[2]) } : null;
}

function pct(x) {
  return (clamp(x, 0, 1) * 100).toFixed(1) + "%";
}

function fmtGap(x) {
  return x > 0 && x < 0.01 ? x.toFixed(4).replace(".", ",") : fmtScore(x);
}

function isCompany(role) {
  return /empresa|companhia/i.test(role || "");
}

function trimStart(text, max) {
  if (text.length <= max) return text;
  const cut = text.slice(text.length - max);
  const sp = cut.indexOf(" ");
  return "[…] " + (sp >= 0 && sp < 40 ? cut.slice(sp + 1) : cut);
}

function trimEnd(text, max) {
  if (text.length <= max) return text;
  const cut = text.slice(0, max);
  const sp = cut.lastIndexOf(" ");
  return (sp > max - 40 ? cut.slice(0, sp) : cut) + " […]";
}

function quoteMarks(s) {
  const u = s.u;
  const prop = u.proposition;
  const ranges = [];
  u.quotes.forEach((q) => {
    const at = prop.indexOf(q);
    if (at >= 0) ranges.push({ s: at, e: at + q.length, cls: "cs-q" });
  });
  let hit = null;
  if (s.isQuote && s.quote) hit = { text: s.quote.text, qw: s.quote.qw, from: s.quote.qi, n: s.quote.np };
  else if (s.qsem && s.qsem.np && s.qsem.run >= s.qsem.np) {
    const qw = wordsOf(s.qsem.text);
    hit = { text: s.qsem.text, qw, from: seqIndex(qw, wordsOf(s.qsem.prefix)), n: s.qsem.run };
  }
  if (hit && hit.from >= 0) {
    const at = prop.indexOf(hit.text);
    if (at >= 0) {
      const a = hit.qw[hit.from];
      const b = hit.qw[Math.min(hit.qw.length - 1, hit.from + hit.n - 1)];
      ranges.push({ s: at + a.s, e: at + b.e, cls: "wl-qw" });
    }
  }
  return ranges;
}

function quoteLine(s) {
  if (!s.u.quotes.length) return "";
  if (!s.resolved) return "As aspas não foram procuradas na fala: a pessoa não foi achada entre quem fala.";
  if (!s.ev) return "As aspas não foram procuradas: não havia frase para comparar.";
  if (s.isQuote && s.quote) {
    const qd = s.quote;
    const n = (qd.qi === 0 ? "as " + qd.np + " primeiras" : qd.np) + " das " + qd.nq + " palavras";
    return "Só o começo das aspas foi procurado na fala: " + n + " aparecem nela, seguidas e iguais (em vermelho). As outras palavras das aspas não foram comparadas.";
  }
  if (s.qsem && s.qsem.np && s.qsem.run >= s.qsem.np) return "O começo das aspas (em vermelho) aparece na fala, mas é curto; a frase foi escolhida pelo sentido.";
  return "As aspas não aparecem iguais na fala; a frase foi escolhida pelo sentido.";
}

function stampHtml(s, M) {
  const sc = s.ev && !s.isQuote ? "<small>nota " + fmtScore(s.ev.score, M.CUT) + "</small>" : "";
  return '<div class="cs-stamp"><div class="wl-ink" data-t="' + s.tier.k + '"><span>' + esc(s.tier.label) + "</span>" + sc + "</div></div>";
}

function claimHtml(s, M) {
  const u = s.u;
  const line = quoteLine(s);
  return (
    '<article class="cs-st"><span class="wl-pin" aria-hidden="true"></span>' +
    '<p class="wl-st-k">Afirmação ' + (s.i + 1) + " de " + M.S.length + " · segundo a matéria,</p>" +
    '<p class="wl-st-who">' + esc(u.actor.name) + "</p>" +
    (u.actor.role ? '<p class="cs-role">' + esc(u.actor.role) + "</p>" : "") +
    '<p class="wl-st-prop cs-prop">' + markText(u.proposition, quoteMarks(s)) + "</p>" +
    (line ? '<p class="cs-qline">' + esc(line) + "</p>" : "") +
    stampHtml(s, M) +
    "</article>"
  );
}

function contextOf(s, M) {
  const ev = s.ev;
  const turn = M.turnsByIdx[ev.speaker_turn];
  const sents = (turn ? turn.sentences : []).filter((x) => x.start !== null && x.end !== null).sort((a, b) => a.start - b.start);
  const before = sents.filter((x) => x.end <= ev.start_char).slice(-CONTEXT_SENTENCES);
  const after = sents.filter((x) => x.start >= ev.end_char).slice(0, CONTEXT_SENTENCES);
  const T = M.T;
  const a = before.length ? before[0].start : ev.start_char;
  const b = after.length ? after[after.length - 1].end : ev.end_char;
  return {
    before: trimStart(T.slice(a, ev.start_char).replace(/\s+/g, " ").trim(), CONTEXT_MAX),
    after: trimEnd(T.slice(ev.end_char, b).replace(/\s+/g, " ").trim(), CONTEXT_MAX),
    main: T.slice(ev.start_char, ev.end_char),
  };
}

function evidenceHtml(s, main) {
  if (s.isQuote && s.quote && main === s.ev.text) {
    const qd = s.quote;
    const ranges = qd.si >= 0 ? [{ s: qd.sw[qd.si].s, e: qd.sw[Math.min(qd.sw.length - 1, qd.si + qd.np - 1)].e, cls: "wl-w" }] : [];
    return markText(main, ranges);
  }
  if (s.qsem && s.qsem.run >= s.qsem.np && s.qsem.si >= 0 && main === s.ev.text) {
    const q = s.qsem;
    return markText(main, [{ s: q.sw[q.si].s, e: q.sw[Math.min(q.sw.length - 1, q.si + q.run - 1)].e, cls: "wl-w" }]);
  }
  return esc(main);
}

function passageHtml(s, M) {
  const ev = s.ev;
  const ctx = contextOf(s, M);
  const g = M.groupOfSt[s.i];
  return (
    '<article class="cs-pa"><span class="wl-pin is-l" aria-hidden="true"></span><span class="wl-pin is-r" aria-hidden="true"></span><div class="cs-pa-in">' +
    '<p class="wl-pa-h"><span>Turno ' + tno(ev.speaker_turn) + " de " + fmtInt(M.NT) + "</span><span>fala de " + esc(M.speakerName(ev.speaker_turn)) + "</span></p>" +
    '<p class="cs-pa-t">' + (ctx.before ? '<span class="cs-ctx">' + esc(ctx.before) + "</span> " : "") + "<mark>" + evidenceHtml(s, ctx.main) + "</mark>" + (ctx.after ? ' <span class="cs-ctx">' + esc(ctx.after) + "</span>" : "") + "</p>" +
    '<p class="cs-pa-f">Em amarelo, a frase escolhida; em cinza, as frases em volta, no mesmo turno. Caracteres ' + fmtInt(ev.start_char) + " a " + fmtInt(ev.end_char) + " de " + fmtInt(M.T.length) + ". " +
    esc(s.u.actor.name) + " fala em " + (g.turns.length === 1 ? "um dos " : g.turns.length + " dos ") + fmtInt(M.NT) + " turnos.</p>" +
    "</div></article>"
  );
}

function englishHtml(s, H) {
  const tr = s.u.signals && s.u.signals.translation;
  if (!tr) return "";
  const model = H.signals.translation;
  return (
    '<aside class="cs-en" aria-label="Cópia em inglês">' +
    '<p class="cs-en-k">Cópia em inglês que o verificador leu</p>' +
    '<p class="cs-en-l">Frase</p><p class="cs-en-t" lang="en">' + esc(tr.premise) + "</p>" +
    '<p class="cs-en-l">Afirmação</p><p class="cs-en-t" lang="en">' + esc(tr.hypothesis) + "</p>" +
    '<p class="cs-en-f">Tradução automática, feita por ' + esc(model.name) + ". O verificador combina respostas sobre o texto em português e sobre esta cópia; se a tradução errar, as respostas em inglês mudam junto.</p>" +
    "</aside>"
  );
}

function whyHtml(s, M) {
  const u = s.u;
  let text;
  if (!s.resolved) {
    const who = isCompany(s.p.role) ? "essa empresa" : "essa pessoa";
    const mt = s.mentions.turns;
    let tailText = "";
    if (mt.length) tailText = " O nome aparece na fala de outras pessoas" + (mt.length <= 3 ? " (" + plural(mt.length, "turno ", "turnos ") + joinPt(mt.map((t) => String(tno(t)))) + ")" : "") + (s.spot.viaNote ? ", e a matéria cita uma nota." : ".");
    else if (s.spot.viaNote) tailText = " A matéria cita uma nota.";
    text = "Procuramos " + who + " entre quem fala nos " + fmtInt(M.NT) + " turnos da transcrição e não achamos. Pode ser que não tenha falado ou que apareça com outro nome." + tailText;
  } else {
    text = u.actor.name + " fala em " + countLabel(s.turns.length, "turno", "turnos") + ", mas nenhuma frase tem pelo menos 4 palavras. Não há frase para comparar com a afirmação.";
  }
  return '<div class="cs-why"><span class="wl-pin" aria-hidden="true"></span><p class="cs-why-h">' + esc(s.tier.label) + '</p><p class="cs-why-t">' + esc(text) + "</p></div>";
}

function rulerHtml(value, cut, label) {
  const c = pct(cut);
  return (
    '<div class="wl-rul cs-rul" role="img" aria-label="' + esc(label) + '"><div class="wl-rul-bar"><span class="wl-rul-zone" style="left:' + c + '"></span><span class="wl-rul-cut" style="left:' + c + '"></span>' +
    (value === null ? "" : '<span class="wl-rul-mk" style="left:' + pct(value) + '"></span>') +
    '</div><div class="wl-rul-ax"><span style="left:0">0</span><span style="left:' + c + '">' + fmtScore(cut) + '</span><span style="left:100%">1</span></div></div>'
  );
}

function measureBlock(title, sub, value, cut, above, valueText) {
  const side = value === null ? "" : above ? "acima do corte" : "abaixo do corte";
  return (
    '<div class="cs-m"><p class="cs-m-h">' + esc(title) + '</p><p class="cs-m-s">' + esc(sub) + "</p>" +
    rulerHtml(value, cut, title + ": " + (value === null ? "sem nota" : "nota " + valueText + ", corte " + fmtScore(cut) + ", " + side)) +
    '<p class="cs-m-v">' + (value === null ? "Sem nota: a frase foi achada pelas aspas." : "Nota <b>" + esc(valueText) + "</b>, " + side + " (" + fmtScore(cut) + ").") + "</p></div>"
  );
}

function agreement(cosAbove, verAbove) {
  if (cosAbove === null) return "Aqui só o verificador dá nota, e ela fica " + (verAbove ? "acima" : "abaixo") + " do corte dele; a frase veio das aspas, não da semelhança.";
  if (cosAbove && verAbove) return "As duas medidas ficam acima dos seus cortes.";
  if (!cosAbove && !verAbove) return "As duas medidas ficam abaixo dos seus cortes.";
  if (cosAbove) return "As medidas discordam: pelo cosseno a frase é parecida com a afirmação, mas o verificador fica abaixo do corte dele.";
  return "As medidas discordam: a semelhança fica abaixo do corte da execução, mas o verificador passa do corte dele.";
}

function questionsHtml(s, H) {
  const sg = s.u.signals;
  const [pt, en] = LAYA_NAMES.map((n) => sg.laya[n]);
  const bar = (x, k) => '<span class="cs-bar" data-k="' + k + '"><i style="width:' + pct(x) + '"></i></span><b>' + fmtScore(x) + "</b>";
  const rows = H.signals.questions
    .map((q, i) => {
      const c = questionCopy(q);
      const split = Math.abs(pt[q.id] - en[q.id]) >= SPLIT_GAP;
      return (
        "<tr" + (split ? ' class="is-split"' : "") + '><th scope="row"><span class="cs-qn">P' + (i + 1) + "</span><span>" + esc(c.text) + (c.value ? "<small>" + esc(c.value) + "</small>" : "") + "</span></th>" +
        "<td>" + bar(pt[q.id], "pt") + "</td><td>" + bar(en[q.id], "en") + "</td></tr>"
      );
    })
    .join("");
  const x = sg.xnli;
  return (
    '<div class="cs-qs"><p class="cs-m-h">Oito perguntas a um modelo de decisão</p>' +
    '<p class="cs-m-s">A mesma frase e a mesma afirmação, oito perguntas diferentes; a nota vai de 0 a 1, a favor da afirmação. PT leu o texto em português, EN a cópia em inglês. Faixa vermelha: as duas leituras diferem em ' + fmtScore(SPLIT_GAP) + " ou mais.</p>" +
    '<table class="cs-qt"><thead><tr><th scope="col">Pergunta</th><th scope="col">PT</th><th scope="col">EN</th></tr></thead><tbody>' + rows + "</tbody></table>" +
    '<p class="cs-xnli">Outro modelo (' + esc(H.signals.scorers.xnli_mdeberta.model.split("/").pop()) + "), no texto em português: implica " + fmtScore(x.entailment) + ", neutra " + fmtScore(x.neutral) + ", contradiz " + fmtScore(x.contradiction) + "." +
    (x.truncated ? " A frase foi cortada para caber nesse modelo." : "") + "</p></div>"
  );
}

function measuresHtml(s, H, M) {
  const sg = s.u.signals;
  const v = sg.verifier;
  const cut = H.signals.verifier.threshold;
  const cos = s.isQuote ? null : s.ev.score;
  const cosAbove = cos === null ? null : s.tier.k !== "w";
  return (
    '<div class="cs-me">' +
    measureBlock("Semelhança de sentido", "Cosseno entre a afirmação e a frase escolhida; o corte é o da execução.", cos, M.CUT, cosAbove, cos === null ? "" : fmtScore(cos, M.CUT)) +
    measureBlock(
      "Verificador",
      "Modelo treinado para dizer se um trecho sustenta uma opinião da matéria. Aprendeu com quatro trechos por opinião e aqui lê uma frase só; o corte foi escolhido no treino.",
      v.probability, cut, v.supported, fmtScore(v.probability, cut),
    ) +
    '<p class="cs-agree">' + esc(agreement(cosAbove, v.supported)) + "</p>" +
    questionsHtml(s, H) +
    "</div>"
  );
}

function candidatesHtml(s, M) {
  const c = s.u.candidates;
  if (!s.ev || !c.length) return "";
  const ev = s.ev;
  const chosenAt = c.findIndex((x) => x.start === ev.start_char && x.end === ev.end_char && x.turn === ev.speaker_turn);
  const items = c
    .map((x, r) => {
      const best = r === chosenAt;
      return (
        "<li" + (best ? ' class="is-best"' : "") + '><span class="cs-rk">' + (r + 1) + '</span><div class="cs-ct"><p class="cs-cx">' + esc(x.text) + '</p><p class="cs-cm">turno ' + tno(x.turn) + (best ? " · <b>a frase escolhida</b>" : "") + "</p></div>" +
        '<div class="cs-cs"><span class="cs-sbar" aria-hidden="true"><i style="width:' + pct(x.score) + '"></i><em style="left:' + pct(M.CUT) + '"></em></span><b>' + fmtScore(x.score, M.CUT) + "</b></div></li>"
      );
    })
    .join("");
  let foot;
  if (s.isQuote) foot = "A frase escolhida veio das aspas" + (chosenAt >= 0 ? " e é a " + (chosenAt + 1) + "ª desta lista pelo sentido." : ", e não está entre estas " + c.length + " frases mais parecidas pelo sentido.");
  else if (c.length > 1) foot = "A segunda frase ficou " + fmtGap(c[0].score - c[1].score) + " abaixo da primeira.";
  else foot = "Só havia esta frase para comparar.";
  return (
    '<section class="cs-row" aria-labelledby="cs-h4"><h2 class="cs-h" id="cs-h4"><span class="cs-hn" aria-hidden="true">4</span>Outras frases parecidas que a pessoa disse</h2>' +
    '<div class="cs-list"><p class="cs-m-s">' + plural(c.length, "A mais parecida", "As " + c.length + " mais parecidas") + " entre as " + countLabel(s.nSent, "frase", "frases") + " de " + esc(s.u.actor.name) + ", pela nota de semelhança; o traço preto na barra é o corte da execução, " + fmtScore(M.CUT) + ".</p>" +
    '<ol class="cs-cands">' + items + '</ol><p class="cs-foot">' + esc(foot) + "</p></div></section>"
  );
}

function chipsHtml(M, id, cur) {
  return (
    '<ol class="cs-chips" aria-label="Afirmações desta matéria">' +
    M.S.map((s) => {
      const n = s.i + 1;
      return (
        '<li><a href="' + caseHash(id, n) + '"' + (n === cur ? ' aria-current="page"' : "") + ' title="' + esc(s.u.actor.name + ": " + s.tier.label) + '">' +
        '<span class="cs-no">' + n + '</span><span class="cs-chip-n">' + esc(firstName(s.u.actor.name)) + '</span><i class="dot" data-k="' + DOT_KEY[s.tier.k] + '" aria-hidden="true"></i><span class="vh">, ' + esc(s.tier.label) + "</span></a></li>"
      );
    }).join("") +
    "</ol>"
  );
}

function headHtml(H, M, s) {
  const id = H.hearing.id;
  const n = s.i + 1;
  const total = M.S.length;
  const date = fmtDate(H.hearing.article_date);
  return (
    '<div class="hv-head"><div class="hv-top"><a class="back" href="#h' + id + '" data-case-back><span aria-hidden="true">&larr;</span> Voltar à audiência</a><div class="hv-nav">' +
    (n > 1 ? '<a class="btn" href="' + caseHash(id, n - 1) + '"><span aria-hidden="true">&larr;</span><span>Anterior</span></a>' : "") +
    (n < total ? '<a class="btn" href="' + caseHash(id, n + 1) + '"><span>Próxima</span><span aria-hidden="true">&rarr;</span></a>' : "") +
    "</div></div>" +
    '<p class="hv-k">' + (date ? "Matéria de " + esc(date) + " · " : "") + "Audiência " + id + "</p>" +
    '<h1 class="hv-title" id="cs-title" tabindex="-1">Pasta da afirmação ' + n + " de " + total + ": " + esc(s.u.actor.name) + "</h1>" +
    '<p class="hv-sum">' + esc(M.article.headline) + "</p>" +
    chipsHtml(M, id, n) +
    '<p class="note is-compact"><b>Sem conferência humana.</b> ' + esc(CASE_NOTE) + "</p></div>"
  );
}

export function renderCase(host, H, n) {
  const M = buildModel(H);
  const s = M.S[n - 1];
  if (!s) return null;
  const scored = !!(H.signals && s.u.signals && s.u.signals.scored);
  const cols = [
    '<section class="cs-col" aria-labelledby="cs-h1"><h2 class="cs-h" id="cs-h1"><span class="cs-hn" aria-hidden="true">1</span>O que a matéria diz</h2>' + claimHtml(s, M) + "</section>",
    '<section class="cs-col" aria-labelledby="cs-h2"><h2 class="cs-h" id="cs-h2"><span class="cs-hn" aria-hidden="true">2</span>O que a pessoa disse</h2>' +
      (s.ev ? passageHtml(s, M) + englishHtml(s, H) : whyHtml(s, M)) +
      (s.ev && !H.signals ? '<p class="cs-slip">' + esc(NO_SECOND_OPINION) + "</p>" : "") +
      "</section>",
  ];
  if (scored) cols.push('<section class="cs-col" aria-labelledby="cs-h3"><h2 class="cs-h" id="cs-h3"><span class="cs-hn" aria-hidden="true">3</span>O que as medidas dizem</h2>' + measuresHtml(s, H, M) + "</section>");
  host.innerHTML =
    headHtml(H, M, s) +
    '<div class="wl cs"><div class="cs-board"><div class="cs-grid" data-cols="' + cols.length + '">' + cols.join("") + "</div>" + candidatesHtml(s, M) + "</div></div>";
  return { title: "Pasta da afirmação " + n + " · Audiência " + H.hearing.id, heading: host.querySelector("#cs-title") };
}
