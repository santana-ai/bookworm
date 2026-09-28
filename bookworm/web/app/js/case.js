import { bandEvidenceLine, bandRange, CASE_NOTE, DISCLOSE_HINTS, findingDetail, GLOSSARY, judgmentLine, MEASURES, NO_SECOND_OPINION, P4_ID, p4Mismatch, p4Note, questionCopy, questionsDisagree, SPLIT_GAP, SUMMARY_JOBS, SUPPORT_NOTE, unitCopy, validationLead, validationLong, validationShort, VERIFIER_NONE } from "./copy.js";
import { bindDisclose, discloseBar, discloseEnd } from "./disclose.js";
import { LAYA_NAMES } from "./data.js";
import { buildModel } from "./model.js";
import { supportAria, supportChip, supportIcon, supportRuler } from "./support.js";
import { clamp, countLabel, esc, firstName, fmtCut, fmtDate, fmtInt, fmtScore, joinPt, markText, plural, seqIndex, shorten, tno, wordsOf } from "./text.js";

const CONTEXT_SENTENCES = 2;
const CONTEXT_MAX = 600;
const RAW_TURNS = 4;
const RAW_TURN_MAX = 320;
const DOT_KEY = { q: "q", h: "h", w: "w", n: "u", u: "u" };
const INTERPRETER = /INT[ÉE]RPRETE/i;

export function caseHash(id, n) {
  return "#h" + id + "-u" + n;
}

export function caseToken(value) {
  const m = String(value || "").match(/^#?h(\d+)-u(\d+)$/);
  return m ? { id: Number(m[1]), n: Number(m[2]) } : null;
}

export function readerHash(hearingId, turn, start, end) {
  return "#h" + hearingId + "-t" + tno(turn) + "-p" + start + "-" + end;
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

function quoteLine(s, M) {
  const by = M.U.window ? "o trecho foi escolhido pela semelhança de sentido." : "a frase foi escolhida pela semelhança de sentido.";
  if (!s.u.quotes.length) return "";
  if (!s.resolved) return "As aspas não foram procuradas na fala: a pessoa não foi achada entre quem fala.";
  if (!s.ev) return "As aspas não foram procuradas: não havia frase para comparar.";
  if (s.isQuote && s.quote) {
    const qd = s.quote;
    const n = (qd.qi === 0 ? "as " + qd.np + " primeiras" : qd.np) + " das " + qd.nq + " palavras";
    const of = s.u.quotes.length > 1 ? " de 1 das " + s.u.quotes.length + " citações" : "";
    return "Só o começo das aspas" + of + " foi procurado na fala: " + n + " aparecem nela, seguidas e iguais (em vermelho). As outras palavras das aspas não foram comparadas.";
  }
  if (s.qsem && s.qsem.np && s.qsem.run >= s.qsem.np) return "O começo das aspas (em vermelho) aparece na fala, mas é curto; " + by;
  return "As aspas não aparecem iguais na fala; " + by;
}

function simText(s, M) {
  return s.ev && !s.isQuote ? fmtScore(s.ev.score, M.CUT) : "";
}

function stampHtml(s, M) {
  const sc = s.ev && !s.isQuote ? "<small>" + MEASURES.similarity + " " + simText(s, M) + "</small>" : "";
  return '<div class="cs-stamp"><div class="wl-ink" data-t="' + s.tier.k + '"><span>' + esc(s.tier.label) + "</span>" + sc + "</div>" + (s.sup ? supportChip(s.sup, true) : "") + "</div>";
}

function profLink(prof) {
  return prof ? '<p class="cs-prof"><a href="' + esc(prof.href) + '">Ver o perfil de ' + esc(prof.name) + ' <span aria-hidden="true">&rarr;</span></a></p>' : "";
}

function claimHtml(s, M, prof) {
  const u = s.u;
  const line = quoteLine(s, M);
  return (
    '<article class="cs-st"><span class="wl-pin" aria-hidden="true"></span>' +
    '<p class="wl-st-k">Afirmação ' + (s.i + 1) + " de " + M.S.length + " · segundo a matéria,</p>" +
    '<p class="wl-st-who">' + esc(u.actor.name) + "</p>" +
    (u.actor.role ? '<p class="cs-role">' + esc(u.actor.role) + "</p>" : "") +
    profLink(prof) +
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
    '<p class="cs-pa-f">Em amarelo, ' + esc(M.U.chosen) + "; em cinza, as frases em volta, no mesmo turno. " +
    esc(s.u.actor.name) + " fala em " + (g.turns.length === 1 ? "um dos " : g.turns.length + " dos ") + fmtInt(M.NT) + " turnos. Posição na transcrição: caracteres " + fmtInt(ev.start_char) + " a " + fmtInt(ev.end_char) + " de " + fmtInt(M.T.length) + ".</p>" +
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
    '<p class="cs-en-l">Trecho</p><p class="cs-en-t" lang="en">' + esc(tr.premise) + "</p>" +
    '<p class="cs-en-l">Afirmação</p><p class="cs-en-t" lang="en">' + esc(tr.hypothesis) + "</p>" +
    '<p class="cs-en-f">Tradução automática, feita por ' + esc(model.name) + ". O verificador combina respostas sobre o texto em português e sobre esta cópia; se a tradução errar, as respostas em inglês mudam junto.</p>" +
    "</aside>"
  );
}

function rawTurnText(M, ti) {
  const t = M.turnsByIdx[ti];
  if (!t) return "";
  const text = M.T.slice(t.start, t.end).replace(/\s+/g, " ").trim();
  return shorten(text, RAW_TURN_MAX);
}

function rawTurnsHtml(s, M) {
  const hid = M.H.hearing.id;
  const rows = [];
  s.turns.slice(0, RAW_TURNS).forEach((ti) => {
    const t = M.turnsByIdx[ti];
    if (!t) return;
    rows.push('<li><p class="cs-raw-k">Turno ' + tno(ti) + ", " + esc(s.u.actor.name) + ' · <a href="' + readerHash(hid, ti, t.start, t.end) + '">ler na transcrição</a></p><p class="cs-raw-t">' + esc(rawTurnText(M, ti)) + "</p></li>");
    const nx = M.turnsByIdx[ti + 1];
    if (nx && INTERPRETER.test(nx.speaker + " " + nx.party)) {
      rows.push('<li class="is-int"><p class="cs-raw-k">Turno ' + tno(ti + 1) + ", logo depois: " + esc(M.speakerName(ti + 1)) + ' · <a href="' + readerHash(hid, ti + 1, nx.start, nx.end) + '">ler na transcrição</a></p><p class="cs-raw-t">' + esc(rawTurnText(M, ti + 1)) + "</p></li>");
    }
  });
  const more = s.turns.length - Math.min(s.turns.length, RAW_TURNS);
  const interp = rows.some((r) => r.indexOf("is-int") >= 0);
  return (
    '<p class="cs-why-t">Os turnos, como estão na transcrição:</p><ol class="cs-raw">' + rows.join("") + "</ol>" +
    (more > 0 ? '<p class="cs-why-f">E mais ' + countLabel(more, "turno", "turnos") + ".</p>" : "") +
    (interp ? '<p class="cs-why-f">O turno seguinte é de um intérprete: a fala pode ter sido registrada na voz dele, e a busca só procura nos turnos da própria pessoa.</p>' : "")
  );
}

function similarHtml(s, M) {
  const list = (s.p && s.p.similar_speakers) || [];
  const hid = M.H.hearing.id;
  if (!list.length) return '<p class="cs-why-f">Nenhum nome parecido aparece entre quem fala.</p>';
  return (
    '<p class="cs-why-t">Nomes parecidos entre quem fala, para conferir. A semelhança é só de grafia: não confirma que é a mesma pessoa.</p><ul class="cs-sim">' +
    list
      .map((x) => {
        const t0 = M.turnsByIdx[x.turns[0]];
        return (
          "<li><b>" + esc(x.name) + "</b> <span>" + esc(countLabel(x.turns.length, "turno", "turnos")) + (x.turns.length <= 3 ? " (" + joinPt(x.turns.map((t) => String(tno(t)))) + ")" : "") + "</span>" +
          (t0 ? ' · <a href="' + readerHash(hid, x.turns[0], t0.start, t0.end) + '">ler o primeiro turno</a>' : "") + "</li>"
        );
      })
      .join("") +
    "</ul>"
  );
}

function whyHtml(s, M, cls) {
  const u = s.u;
  let body;
  if (!s.resolved) {
    const who = isCompany(s.p.role) ? "essa empresa" : "essa pessoa";
    const mt = s.mentions.turns;
    let tailText = "";
    if (mt.length) tailText = " O nome aparece na fala de outras pessoas" + (mt.length <= 3 ? " (" + plural(mt.length, "turno ", "turnos ") + joinPt(mt.map((t) => String(tno(t)))) + ")" : "") + (s.spot.viaNote ? ", e a matéria cita uma nota." : ".");
    else if (s.spot.viaNote) tailText = " A matéria cita uma nota.";
    body = '<p class="cs-why-t">' + esc("A busca procurou " + who + " entre quem fala nos " + fmtInt(M.NT) + " turnos da transcrição e não achou. Pode ser que não tenha falado ou que apareça com outro nome." + tailText) + "</p>" + similarHtml(s, M);
  } else {
    body = '<p class="cs-why-t">' + esc("A transcrição registra " + countLabel(s.turns.length, "turno", "turnos") + " de " + u.actor.name + ", mas sem frase de pelo menos 4 palavras. Não há frase para comparar com a afirmação.") + "</p>" + rawTurnsHtml(s, M);
  }
  const judged = judgmentLine(u.id);
  return '<div class="cs-why' + (cls ? " " + cls : "") + '"><span class="wl-pin" aria-hidden="true"></span><p class="cs-why-h">' + esc(s.tier.label) + "</p>" + body + (judged ? '<p class="cs-judged">' + esc(judged) + "</p>" : "") + "</div>";
}

function rulerHtml(value, cut, label) {
  const c = pct(cut);
  return (
    '<div class="wl-rul cs-rul" role="img" aria-label="' + esc(label) + '"><div class="wl-rul-bar"><span class="wl-rul-zone" style="left:' + c + '"></span><span class="wl-rul-cut" style="left:' + c + '"></span>' +
    (value === null ? "" : '<span class="wl-rul-mk" style="left:' + pct(value) + '"></span>') +
    '</div><div class="wl-rul-ax"><span style="left:0">0</span><span style="left:' + c + '">' + fmtCut(cut) + "</span>" +
    '<span style="left:100%">1</span></div></div>'
  );
}

function similarityBlock(s, M) {
  const cos = s.isQuote ? null : s.ev.score;
  const above = cos !== null && s.tier.k !== "w";
  const side = cos === null ? "" : above ? "acima do corte" : "abaixo do corte";
  const text = cos === null ? "" : fmtScore(cos, M.CUT);
  return (
    '<div class="cs-m"><p class="cs-m-h">Semelhança de sentido</p><p class="cs-m-s">' + esc("Cosseno entre a afirmação e " + M.U.chosen + ", de 0 a 1: quanto o sentido das duas se parece, pelas representações de um modelo de frases. O corte é o da rodada.") + "</p>" +
    rulerHtml(cos, M.CUT, "Semelhança: " + (cos === null ? "não usada" : text + ", corte " + fmtCut(M.CUT) + ", " + side)) +
    '<p class="cs-m-v">' + (cos === null ? "Não usada: o trecho foi achado pelas aspas." : "Semelhança <b>" + esc(text) + "</b>, " + side + " (" + fmtCut(M.CUT) + ").") + "</p></div>"
  );
}

function verifierSub(M) {
  const C = M.C;
  const base = "Modelo treinado para dizer se um trecho sustenta uma opinião da matéria. Aprendeu com quatro trechos recuperados por opinião e aqui lê " + M.U.read + ".";
  if (C.train === null) return base + " O corte foi escolhido no treino.";
  return (
    base + " Dois cortes separam as faixas. O de " + fmtCut(C.low) + " foi calibrado para trechos escolhidos pela busca, em audiências de treino: fica no meio entre o apoio típico de pares certos e a de pares em que o trecho foi sorteado de outra fala (regra " + C.rule + "). O de " +
    fmtCut(C.high) + " é o corte escolhido no treino do verificador, com os quatro trechos."
  );
}

function p4Of(s) {
  const pt = s.u.signals && s.u.signals.laya ? s.u.signals.laya[LAYA_NAMES[0]] : null;
  return pt && typeof pt[P4_ID] === "number" ? pt[P4_ID] : null;
}

function supportBlock(s, M) {
  const sup = s.sup;
  const ev = bandEvidenceLine(sup.b, M.C);
  const p4 = p4Of(s);
  return (
    '<div class="cs-m cs-m-ap"><p class="cs-m-h">Apoio do verificador</p><p class="cs-m-s">' + esc(verifierSub(M)) + "</p>" +
    supportRuler(sup, "Apoio: " + supportAria(sup)) +
    '<p class="cs-m-v">Apoio <b>' + esc(sup.text) + "</b>: " + supportChip(sup, false) + " (" + esc(bandRange(sup.b, M.C)) + "). " + esc(SUPPORT_NOTE) + "</p>" +
    (ev ? '<p class="cs-m-v">' + esc(ev) + "</p>" : "") +
    (p4 !== null && p4Mismatch(sup.b, p4, M.C) ? '<p class="cs-m-v is-alt">' + esc(p4Note(fmtScore(p4))) + "</p>" : "") +
    "</div>"
  );
}

function twoQuestions(s, M) {
  const where = s.tier.label.toLowerCase() + (s.isQuote ? "" : " (semelhança " + simText(s, M) + ")");
  const how = s.sup.b.label + " (" + s.sup.text + ")";
  const extra = questionsDisagree(s.tier.k, s.sup.b);
  return "São duas perguntas diferentes. Onde está o trecho: " + where + ". Quanto ele apoia a afirmação: " + how + "." + (extra ? " " + extra : "");
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
    '<p class="cs-m-s">O mesmo trecho e a mesma afirmação, oito perguntas diferentes; cada resposta vai de 0 a 1, a favor da afirmação, e o apoio acima combina todas elas. PT leu o texto em português, EN a cópia em inglês. Faixa vermelha: as duas leituras diferem em ' + fmtScore(SPLIT_GAP) + " ou mais.</p>" +
    '<table class="cs-qt"><thead><tr><th scope="col">Pergunta</th><th scope="col">PT</th><th scope="col">EN</th></tr></thead><tbody>' + rows + "</tbody></table>" +
    '<p class="cs-xnli">Outro modelo (' + esc(H.signals.scorers.xnli_mdeberta.model.split("/").pop()) + "), no texto em português: implica " + fmtScore(x.entailment) + ", neutra " + fmtScore(x.neutral) + ", contradiz " + fmtScore(x.contradiction) + "." +
    (x.truncated ? " A frase foi cortada para caber nesse modelo." : "") + "</p></div>"
  );
}

function measuresHtml(s, H, M) {
  return '<div class="cs-me">' + similarityBlock(s, M) + supportBlock(s, M) + '<p class="cs-agree">' + esc(twoQuestions(s, M)) + "</p>" + questionsHtml(s, H) + "</div>";
}

function candidatesHtml(s, M) {
  const c = s.u.candidates;
  if (!s.ev || !c.length) return "";
  const ev = s.ev;
  const U = M.U;
  const chosenAt = c.findIndex((x) => x.start === ev.start_char && x.end === ev.end_char && x.turn === ev.speaker_turn);
  const items = c
    .map((x, r) => {
      const best = r === chosenAt;
      return (
        "<li" + (best ? ' class="is-best"' : "") + '><span class="cs-rk">' + (r + 1) + '</span><div class="cs-ct"><p class="cs-cx">' + esc(x.text) + '</p><p class="cs-cm">turno ' + tno(x.turn) + (best ? " · <b>" + esc(U.best) + "</b>" : "") + "</p></div>" +
        '<div class="cs-cs"><span class="cs-sbar" aria-hidden="true"><i style="width:' + pct(x.score) + '"></i><em style="left:' + pct(M.CUT) + '"></em></span><b>' + fmtScore(x.score, M.CUT) + "</b></div></li>"
      );
    })
    .join("");
  let foot;
  if (s.isQuote) foot = (U.window ? "O trecho escolhido veio das aspas" : "A frase escolhida veio das aspas") + (chosenAt >= 0 ? U.rank(chosenAt + 1) : U.notIn(c.length));
  else if (c.length > 1) foot = U.gap(fmtGap(c[0].score - c[1].score));
  else foot = U.only;
  return (
    '<section class="cs-row" aria-labelledby="cs-h4"><h2 class="cs-h" id="cs-h4"><span class="cs-hn" aria-hidden="true">4</span>' + esc(U.heading) + "</h2>" +
    '<div class="cs-list"><p class="cs-m-s">' + esc(U.top(c.length)) + " entre " + (U.window ? "os " : "as ") + esc(U.count(s.nSent)) + " de " + esc(s.u.actor.name) + ", pela semelhança de sentido; o traço preto na barra é o corte da rodada, " + fmtCut(M.CUT) + ".</p>" +
    '<ol class="cs-cands">' + items + '</ol><p class="cs-foot">' + esc(foot) + "</p></div></section>"
  );
}

function chipsHtml(M, id, cur) {
  return (
    '<ol class="cs-chips" aria-label="Afirmações desta matéria">' +
    M.S.map((s) => {
      const n = s.i + 1;
      const supText = s.sup ? ", " + s.sup.b.label : "";
      return (
        '<li><a href="' + caseHash(id, n) + '"' + (n === cur ? ' aria-current="page"' : "") + ' title="' + esc(s.u.actor.name + ": " + s.tier.label + supText) + '">' +
        '<span class="cs-no">' + n + '</span><span class="cs-chip-n">' + esc(firstName(s.u.actor.name)) + '</span><i class="dot" data-k="' + DOT_KEY[s.tier.k] + '" aria-hidden="true"></i>' +
        (s.sup ? '<span class="ap is-icon" data-b="' + s.sup.b.k + '">' + supportIcon(s.sup.b) + "</span>" : "") +
        '<span class="vh">, ' + esc(s.tier.label + supText) + "</span></a></li>"
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
    "</div>"
  );
}

function findGauge(s, M) {
  const t = s.tier.k;
  const cos = s.ev && !s.isQuote ? s.ev.score : null;
  const text = simText(s, M);
  let bar;
  if (!s.ev) bar = '<p class="cs-g-none is-na">não se aplica</p>';
  else if (cos === null) bar = '<p class="cs-g-none">aspas achadas</p>';
  else
    bar =
      '<div class="cs-g-bar" role="img" aria-label="' + esc(MEASURES.similarity + " " + text + " de 0 a 1, corte " + fmtCut(M.CUT)) + '"><i data-t="' + t + '" style="width:' + pct(cos) + '"></i><em style="left:' + pct(M.CUT) + '"></em></div>' +
      '<div class="cs-g-ax" aria-hidden="true"><span style="left:0">0</span><span style="left:' + pct(M.CUT) + '">corte ' + fmtCut(M.CUT) + '</span><span style="left:100%">1</span></div>';
  return (
    '<div class="cs-g"><p class="cs-g-q">' + esc(SUMMARY_JOBS.find) + '</p><p class="cs-g-a" data-t="' + t + '">' + esc(s.tier.label) + (cos === null ? "" : ' <span class="cs-g-m">' + MEASURES.similarity + " <b>" + esc(text) + "</b></span>") + "</p>" + bar +
    (s.ev ? '<p class="cs-g-d">' + esc(findingDetail(s.tierName, text, fmtCut(M.CUT))) + "</p>" : "") + "</div>"
  );
}

function supportGauge(s, H, M) {
  const head = '<div class="cs-g cs-g-ap"><p class="cs-g-q">' + esc(SUMMARY_JOBS.support) + "</p>";
  if (!s.ev) return head + '<p class="cs-g-none is-na">não se aplica</p><p class="cs-g-d">' + esc(VERIFIER_NONE) + "</p></div>";
  if (!s.sup) return head + '<p class="cs-g-d">' + esc(NO_SECOND_OPINION) + "</p></div>";
  const sup = s.sup;
  const ev = bandEvidenceLine(sup.b, M.C);
  const p4 = p4Of(s);
  return (
    head + '<p class="cs-g-a is-ap">' + supportChip(sup, false) + ' <span class="cs-g-m">' + MEASURES.support + " <b>" + esc(sup.text) + "</b></span></p>" +
    supportRuler(sup, "Apoio: " + supportAria(sup)) +
    '<p class="cs-g-d">' + esc("Faixa " + bandRange(sup.b, M.C) + ". " + SUPPORT_NOTE) + "</p>" +
    (ev ? '<p class="cs-g-d is-ev">' + esc(ev) + "</p>" : "") +
    (p4 !== null && p4Mismatch(sup.b, p4, M.C) ? '<p class="cs-g-d">' + esc(p4Note(fmtScore(p4))) + "</p>" : "") +
    "</div>"
  );
}

function summaryHtml(s, H, M, prof) {
  const u = s.u;
  const claim =
    '<article class="cs-st cs-s-claim"><span class="wl-pin" aria-hidden="true"></span>' +
    '<p class="wl-st-k">Afirmação ' + (s.i + 1) + " · segundo a matéria,</p>" +
    '<p class="wl-st-who">' + esc(u.actor.name) + "</p>" +
    (u.actor.role ? '<p class="cs-role">' + esc(u.actor.role) + "</p>" : "") +
    '<p class="wl-st-prop cs-prop">' + markText(u.proposition, quoteMarks(s)) + "</p>" +
    profLink(prof) +
    "</article>";
  const judged = judgmentLine(u.id);
  const passage = s.ev
    ? '<article class="cs-pa cs-s-pa"><span class="wl-pin is-l" aria-hidden="true"></span><span class="wl-pin is-r" aria-hidden="true"></span><div class="cs-pa-in">' +
      '<p class="wl-pa-h"><span>Trecho da audiência</span><span>turno ' + tno(s.ev.speaker_turn) + ", " + esc(M.speakerName(s.ev.speaker_turn)) + "</span></p>" +
      '<p class="cs-pa-t"><mark>' + evidenceHtml(s, s.ev.text) + "</mark></p>" + (judged ? '<p class="cs-judged">' + esc(judged) + "</p>" : "") + "</div></article>"
    : whyHtml(s, M, "cs-s-pa");
  const verdictSup = s.ev ? (s.sup ? supportChip(s.sup, true) : '<span class="cs-verdict-na">' + esc(MEASURES.support + ": não calculado") + "</span>") : "";
  return (
    '<div class="wl cs cs-sum"><div class="cs-board cs-s-board">' +
    '<p class="cs-verdict"><span data-t="' + s.tier.k + '">' + esc(s.tier.label) + "</span>" + (verdictSup ? '<span class="cs-verdict-sep" aria-hidden="true">·</span>' + verdictSup : "") + "</p>" +
    '<div class="cs-s-grid">' + claim + passage + '<div class="cs-me cs-s-g">' + findGauge(s, M) + supportGauge(s, H, M) + '<p class="cs-s-note"><b>' + esc(validationLead()) + "</b> " + esc(validationShort()) + "</p></div></div>" +
    "</div>" + discloseBar("case", "cs-more", DISCLOSE_HINTS.case) + "</div>"
  );
}

function glossaryHtml() {
  return '<dl class="cs-gloss">' + GLOSSARY.map((g) => "<dt>" + esc(g[0]) + "</dt><dd>" + esc(g[1]) + "</dd>").join("") + "</dl>";
}

function notesHtml() {
  return (
    '<div class="note is-compact cs-vnote"><p><b>' + esc(validationLead()) + "</b> " + esc(CASE_NOTE) + "</p>" +
    validationLong().map((p) => "<p>" + esc(p) + "</p>").join("") + glossaryHtml() + "</div>"
  );
}

export function renderCase(host, H, n, profileOf) {
  const M = buildModel(H);
  M.U = unitCopy(H.run);
  const s = M.S[n - 1];
  if (!s) return null;
  const prof = profileOf ? profileOf(s.u.id) : null;
  const cols = [
    '<section class="cs-col" aria-labelledby="cs-h1"><h2 class="cs-h" id="cs-h1"><span class="cs-hn" aria-hidden="true">1</span>O que a matéria diz</h2>' + claimHtml(s, M, prof) + "</section>",
    '<section class="cs-col" aria-labelledby="cs-h2"><h2 class="cs-h" id="cs-h2"><span class="cs-hn" aria-hidden="true">2</span>O que a pessoa disse</h2>' +
      (s.ev ? passageHtml(s, M) + englishHtml(s, H) : whyHtml(s, M)) +
      (s.ev && !H.signals ? '<p class="cs-slip">' + esc(NO_SECOND_OPINION) + "</p>" : "") +
      "</section>",
  ];
  if (s.sup) cols.push('<section class="cs-col" aria-labelledby="cs-h3"><h2 class="cs-h" id="cs-h3"><span class="cs-hn" aria-hidden="true">3</span>O que as medidas dizem</h2>' + measuresHtml(s, H, M) + "</section>");
  host.innerHTML =
    headHtml(H, M, s) +
    summaryHtml(s, H, M, prof) +
    '<div class="cs-more" id="cs-more">' +
    '<div class="cs-more-head">' + chipsHtml(M, H.hearing.id, n) + notesHtml() + "</div>" +
    '<div class="wl cs"><div class="cs-board"><div class="cs-grid" data-cols="' + cols.length + '">' + cols.join("") + "</div>" + candidatesHtml(s, M) + "</div></div>" +
    discloseEnd("case", "cs-more") +
    "</div>";
  bindDisclose(host, "case");
  return { title: "Pasta da afirmação " + n + " · Audiência " + H.hearing.id, heading: host.querySelector("#cs-title") };
}
