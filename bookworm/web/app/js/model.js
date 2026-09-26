import { tierOf } from "./copy.js";
import { fmtDate, firstName, matchBlocks, reEsc, seqIndex, titleCase, wordsOf } from "./text.js";

const DASH_CAP = 440;
const MENTION_CAP = 400;
const SOURCE_CUE = /(?:^|[^\p{L}])(?:em nota|em comunicado|por meio de nota|por nota|segundo nota)(?![\p{L}])/iu;
const DATE_LINE = /^\d{2}\/\d{2}\/\d{4}/;
const CREDIT_LINE = /^(Reportagem|Edição)\s*[–-]/;

function nameVariants(name) {
  const v = [name];
  const m = name.match(/^(.*?)\s*\(([^)]+)\)\s*$/);
  if (m) v.push(m[1], m[2]);
  else {
    const parts = name.split(/\s+/);
    const last = parts[parts.length - 1];
    if (parts.length > 1 && last.length >= 5) v.push(last);
  }
  return v.filter((x, i, a) => x && a.indexOf(x) === i).sort((a, b) => b.length - a.length);
}

function nameRegex(name) {
  const alt = nameVariants(name).map(reEsc).join("|");
  try {
    return new RegExp("(?<![\\p{L}\\p{N}])(?:" + alt + ")(?![\\p{L}\\p{N}])", "gu");
  } catch (error) {
    return new RegExp("(?:" + alt + ")", "g");
  }
}

function findRanges(text, re, cls, d) {
  const out = [];
  let m;
  re.lastIndex = 0;
  while ((m = re.exec(text))) {
    out.push({ s: m.index, e: m.index + m[0].length, cls, d });
    if (!m[0].length) re.lastIndex++;
  }
  return out;
}

function lastAtOrBefore(sorted, value, key) {
  let lo = 0;
  let hi = sorted.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (key(sorted[mid]) <= value) {
      ans = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return ans;
}

function parseArticle(materia) {
  const raw = materia
    .split(/\n+/)
    .map((s) => s.replace(/ /g, " ").trim())
    .filter(Boolean);
  const headline = raw[0] || "";
  let dateAt = -1;
  for (let i = 1; i < Math.min(raw.length, 4); i++) {
    if (DATE_LINE.test(raw[i])) {
      dateAt = i;
      break;
    }
  }
  const subhead = dateAt > 1 ? raw.slice(1, dateAt).join(" ") : dateAt < 0 && raw[1] ? raw[1] : "";
  const bodyStart = dateAt >= 0 ? dateAt + 1 : raw[1] ? 2 : 1;
  const rest = raw.slice(bodyStart);
  const body = rest.filter((p) => !CREDIT_LINE.test(p));
  const credits = rest.filter((p) => CREDIT_LINE.test(p));
  if (!body.length) body.push(headline);
  return { headline, subhead, body, credits, bodyWords: body.map((p) => wordsOf(p)) };
}

function articleSpot(article, u, idx) {
  const { body, bodyWords } = article;
  const needles = u.quotes.concat([u.proposition]);
  const perPara = body.map(() => ({ sc: 0, ranges: [] }));
  needles.forEach((nd) => {
    const nw = wordsOf(nd).map((x) => x.w);
    const n = nw.length;
    if (!n) return;
    let best = null;
    bodyWords.forEach((pw, pi) => {
      const bl = matchBlocks(nw, pw.map((x) => x.w), Math.min(4, n));
      const sc = bl.reduce((a, b) => a + b.n, 0);
      if (!best || sc > best.sc) best = { pi, bl, sc };
    });
    if (!best || best.sc < Math.max(Math.min(5, n), Math.ceil(0.3 * n))) return;
    const pw = bodyWords[best.pi];
    perPara[best.pi].sc = Math.max(perPara[best.pi].sc, best.sc);
    best.bl.forEach((b) => perPara[best.pi].ranges.push({ s: pw[b.j].s, e: pw[b.j + b.n - 1].e, cls: "wl-u", d: idx }));
  });
  let pi = -1;
  let top = 0;
  perPara.forEach((p, i) => {
    if (p.sc > top) {
      top = p.sc;
      pi = i;
    }
  });
  const re = nameRegex(u.actor.name);
  const matched = pi >= 0;
  if (pi < 0) {
    for (let i = 0; i < body.length; i++) {
      re.lastIndex = 0;
      if (re.test(body[i])) {
        pi = i;
        break;
      }
    }
    if (pi < 0) pi = 0;
  }
  const p = body[pi] || "";
  const ranges = (matched ? perPara[pi].ranges : []).concat(findRanges(p, re, "wl-nm", idx));
  let first = ranges.filter((r) => r.cls === "wl-u").reduce((a, r) => Math.min(a, r.s), Infinity);
  if (!isFinite(first)) first = ranges.length ? ranges[0].s : 0;
  return { pi, text: p, ranges, pos: pi + (p.length ? first / p.length : 0), viaNote: SOURCE_CUE.test(p) };
}

export function buildModel(H) {
  const T = H.transcript;
  const RUN = H.run;
  const CUT = RUN.threshold;
  const turns = H.turns.slice().sort((a, b) => a.index - b.index);
  const NT = turns.length;
  const turnsByIdx = {};
  turns.forEach((t) => {
    turnsByIdx[t.index] = t;
  });
  const turnsByStart = turns.slice().sort((a, b) => a.start - b.start);
  const peopleByName = {};
  const personOfTurn = {};
  H.people.forEach((p) => {
    peopleByName[p.name] = p;
    if (p.resolved) p.turns.forEach((ti) => {
      personOfTurn[ti] = p;
    });
  });

  function speakerName(ti) {
    const p = personOfTurn[ti];
    if (p) return p.name;
    const t = turnsByIdx[ti];
    if (!t) return "";
    const m = t.party.match(/^([^.]+)\./);
    return m ? m[1].trim() : titleCase(t.speaker);
  }
  function turnAt(pos) {
    const j = lastAtOrBefore(turnsByStart, pos, (t) => t.start);
    if (j < 0) return null;
    const t = turnsByStart[j];
    return pos < t.end ? t : null;
  }

  const allSents = [];
  turns.forEach((t) => {
    t.sentences.forEach((s) => {
      if (s.start !== null && s.end !== null) allSents.push({ turn: t.index, text: s.text, start: s.start, end: s.end });
    });
  });
  allSents.sort((a, b) => a.start - b.start);
  allSents.forEach((s, i) => {
    s.i = i;
  });
  function sentAt(pos) {
    const j = lastAtOrBefore(allSents, pos, (s) => s.start);
    return j;
  }

  const article = parseArticle(H.hearing.materia);
  const DATE = fmtDate(H.hearing.article_date);

  const PAS = [];
  const pasByKey = {};
  const S = H.udvs.map((u, i) => {
    const p = peopleByName[u.actor.name];
    const resolved = p.resolved;
    const sTurns = resolved ? p.turns.slice().sort((a, b) => a - b) : [];
    const ev = u.evidence;
    const isQuote = !!(ev && ev.support_type === "direct_quote");
    const scenes = !resolved ? ["article", "who", "result"] : ev ? ["article", "who", "search", "passage", "result"] : ["article", "who", "search", "result"];
    const s = {
      u,
      i,
      p,
      pname: p.name,
      resolved,
      turns: sTurns,
      ev,
      isQuote,
      scenes,
      tierName: u.tier,
      tier: tierOf(u.tier),
      nSent: u.n_candidates,
      spot: articleSpot(article, u, i),
      mentions: { turns: [], people: [] },
      quote: null,
      qsem: null,
      pa: null,
    };
    if (!resolved) {
      const re = nameRegex(u.actor.name);
      const seenT = {};
      const seenP = {};
      let m;
      let n = 0;
      re.lastIndex = 0;
      while ((m = re.exec(T)) && n++ < MENTION_CAP) {
        const t = turnAt(m.index);
        if (t) {
          if (!seenT[t.index]) {
            seenT[t.index] = 1;
            s.mentions.turns.push(t.index);
          }
          const nm = speakerName(t.index);
          if (nm && !seenP[nm]) {
            seenP[nm] = 1;
            s.mentions.people.push(nm);
          }
        }
        if (!m[0].length) re.lastIndex++;
      }
    }
    if (ev) {
      const key = ev.start_char + ":" + ev.end_char;
      if (!pasByKey[key]) {
        pasByKey[key] = { key, k: PAS.length, start: ev.start_char, end: ev.end_char, turn: ev.speaker_turn, text: ev.text, sts: [] };
        PAS.push(pasByKey[key]);
      }
      pasByKey[key].sts.push(i);
      s.pa = pasByKey[key].k;
    }
    if (isQuote) {
      const pw = wordsOf(ev.quote_prefix || "");
      const quote = u.quotes.filter((qq) => seqIndex(wordsOf(qq), pw) === 0)[0] || u.quotes[0] || u.proposition;
      const qw = wordsOf(quote);
      const sw = wordsOf(ev.text);
      s.quote = { text: quote, qw, sw, np: pw.length, nq: qw.length, qi: seqIndex(qw, pw), si: seqIndex(sw, pw) };
    } else if (ev && u.quotes.length) {
      const sw2 = wordsOf(ev.text);
      const pw2 = wordsOf(ev.quote_prefix || "");
      let best = null;
      if (pw2.length) {
        u.quotes.forEach((qq) => {
          if (!best && seqIndex(wordsOf(qq), pw2) === 0) best = { text: qq, qw: wordsOf(qq) };
        });
      }
      if (!best) best = { text: u.quotes[0], qw: wordsOf(u.quotes[0]) };
      let run = 0;
      if (pw2.length) {
        const qi2 = seqIndex(best.qw, pw2);
        const si2 = seqIndex(sw2, pw2);
        if (qi2 >= 0 && si2 >= 0) while (qi2 + run < best.qw.length && si2 + run < sw2.length && best.qw[qi2 + run].w === sw2[si2 + run].w) run++;
      }
      s.qsem = { text: best.text, np: pw2.length, run, prefix: ev.quote_prefix || "", sw: sw2, si: pw2.length ? seqIndex(sw2, pw2) : -1 };
    }
    return s;
  });

  const evidenceBySpan = {};
  S.forEach((s) => {
    if (!s.ev) return;
    const key = s.ev.start_char + ":" + s.ev.end_char;
    (evidenceBySpan[key] = evidenceBySpan[key] || []).push(s.i);
  });

  const GROUPS = [];
  const byName = {};
  S.forEach((s) => {
    let g = byName[s.pname];
    if (!g) {
      g = byName[s.pname] = { name: s.pname, p: s.p, sts: [], pas: [], turns: [], resolved: s.resolved };
      GROUPS.push(g);
    }
    g.sts.push(s.i);
    if (s.pa != null && g.pas.indexOf(s.pa) < 0 && !PAS[s.pa].group) g.pas.push(s.pa);
    if (s.pa != null && !PAS[s.pa].group) PAS[s.pa].group = g;
    if (s.resolved) g.resolved = true;
  });
  GROUPS.forEach((g, gi) => {
    g.gi = gi;
    g.turns = g.resolved ? g.p.turns.slice().sort((a, b) => a - b) : [];
    g.pos = g.sts.reduce((a, i) => a + S[i].spot.pos, 0) / g.sts.length;
    g.first = Math.min(...g.sts);
    const own = {};
    g.turns.forEach((t) => {
      own[t] = 1;
    });
    g.sents = g.resolved ? allSents.filter((x) => own[x.turn]) : [];
    g.bin = Math.max(1, Math.ceil(g.sents.length / DASH_CAP));
    g.nDash = Math.ceil(g.sents.length / g.bin);
    g.dashTurn = [];
    for (let d = 0; d < g.nDash; d++) g.dashTurn.push(g.sents[d * g.bin].turn);
    g.dashOf = (pos) => {
      const j = lastAtOrBefore(g.sents, pos, (x) => x.start);
      return j < 0 ? (g.sents.length ? 0 : -1) : Math.floor(j / g.bin);
    };
  });
  GROUPS.sort((a, b) => a.pos - b.pos || a.first - b.first);
  GROUPS.forEach((g, j) => {
    g.order = j;
  });
  const groupOfSt = {};
  GROUPS.forEach((g) => g.sts.forEach((i) => {
    groupOfSt[i] = g;
  }));
  PAS.forEach((pa) => {
    if (!pa.group) pa.group = groupOfSt[pa.sts[0]];
  });
  const ownedTurns = {};
  GROUPS.forEach((g) => g.turns.forEach((t) => {
    ownedTurns[t] = g;
  }));
  const otherTurns = turns.map((t) => t.index).filter((t) => !ownedTurns[t]);
  const otherNames = [];
  otherTurns.forEach((t) => {
    const n = speakerName(t);
    if (n && otherNames.indexOf(n) < 0) otherNames.push(n);
  });

  return {
    H,
    T,
    RUN,
    CUT,
    NT,
    turns,
    turnsByIdx,
    peopleByName,
    personOfTurn,
    speakerName,
    turnAt,
    firstName,
    allSents,
    sentAt,
    article,
    DATE,
    S,
    PAS,
    GROUPS,
    groupOfSt,
    ownedTurns,
    otherTurns,
    otherNames,
    evidenceBySpan,
  };
}
