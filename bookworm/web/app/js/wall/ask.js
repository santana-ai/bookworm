import { esc, firstName, fmtInt, foldText, joinPt, markText, shorten, tail, tno } from "../text.js";
import { cadId } from "./objects.js";

const STOP = (
  "a o e as os um uma uns umas de do da dos das em no na nos nas num numa por pelo pela pelos pelas para pra pro com sem sob sobre ate " +
  "que se nao sim ja ha foi era sao ser sera estar esta estao este esse essa esses essas isso isto aquele aquela aquilo ele ela eles elas eu nos " +
  "voce voces me te lhe lhes seu sua seus suas meu minha nosso nossa ao aos como mas ou mais muito tambem so entao aqui la quando onde qual quais " +
  "quem porque tem ter dito disse falou falaram sobre foram forma entre apos ainda cada todo toda todos todas outro outra outros outras"
).split(" ");
const STOPSET = new Set(STOP);
const GENERIC = new Set(
  "brasil pais brasileiro brasileira brasileiros brasileiras governo federal nacional deputado deputada deputados camara comissao audiencia publica publicas publico presidente senhor senhora sobre".split(" "),
);
const MAX_HITS = 5;
const READ_SPAN = 1400;
const READ_STEP = 1500;

function rawTokens(s) {
  return foldText(s).match(/[a-z0-9]+/g) || [];
}
function keepTok(t) {
  return t.length > 1 && !STOPSET.has(t);
}

export function installAsk(w) {
  const M = w.M;
  const { T, allSents } = M;
  const els = w.els;
  const docs = allSents.map((s) => rawTokens(s.text).filter(keepTok));
  const ND = docs.length;
  const avgdl = docs.reduce((a, d) => a + d.length, 0) / Math.max(1, ND);
  const df = new Map();
  docs.forEach((d) => {
    new Set(d).forEach((t) => df.set(t, (df.get(t) || 0) + 1));
  });
  const idf = (t) => Math.log(1 + (ND - (df.get(t) || 0) + 0.5) / ((df.get(t) || 0) + 0.5));
  const ASK = { hits: [] };
  w.ASK = ASK;

  function bm25(terms) {
    const k1 = 1.2;
    const b = 0.75;
    const out = [];
    docs.forEach((d, i) => {
      if (!d.length) return;
      let sc = 0;
      terms.forEach((t) => {
        let f = 0;
        for (let j = 0; j < d.length; j++) if (d[j] === t) f++;
        if (!f) return;
        sc += (idf(t) * (f * (k1 + 1))) / (f + k1 * (1 - b + (b * d.length) / avgdl));
      });
      if (sc > 0) out.push({ i, s: sc });
    });
    return out.sort((a, b2) => b2.s - a.s || a.i - b2.i).slice(0, MAX_HITS);
  }

  function suggestions() {
    const words = M.H.hearing.assunto.match(/[\p{L}\p{N}]+/gu) || [];
    const seen = new Set();
    const out = [];
    words.forEach((orig) => {
      const toks = rawTokens(orig);
      if (toks.length !== 1) return;
      const t = toks[0];
      if (t.length < 4 || STOPSET.has(t) || GENERIC.has(t) || seen.has(t)) return;
      const n = df.get(t) || 0;
      if (!n || n > ND * 0.35) return;
      seen.add(t);
      out.push({ t, orig: /^[A-ZÁÉÍÓÚÂÊÔÃÕÇ]{2,}$/.test(orig) ? orig : orig.toLowerCase(), n });
    });
    out.sort((a, b) => b.n - a.n);
    return out.slice(0, 4);
  }

  function termHtml(text, qset) {
    let out = "";
    let last = 0;
    let m;
    const re = /[\p{L}\p{N}]+/gu;
    while ((m = re.exec(text))) {
      out += esc(text.slice(last, m.index));
      out += qset.has(foldText(m[0])) ? '<span class="wl-term">' + esc(m[0]) + "</span>" : esc(m[0]);
      last = m.index + m[0].length;
    }
    return out + esc(text.slice(last));
  }

  function placeStrip(hits) {
    const row = els.astripRow;
    row.innerHTML = "";
    let prev = -100;
    let lift = 0;
    hits.forEach((h, r) => {
      const s = allSents[h.i];
      const left = (s.start / Math.max(1, T.length)) * 100;
      lift = Math.abs(left - prev) < 3 ? (lift === 0 ? -14 : lift < 0 ? 14 : 0) : 0;
      prev = left;
      const pin = document.createElement("span");
      pin.className = "wl-apin";
      pin.textContent = String(r + 1);
      pin.style.left = left.toFixed(2) + "%";
      if (lift) pin.style.marginTop = -12 + lift + "px";
      row.appendChild(pin);
    });
  }

  function runAsk(animate) {
    const qv = els.q.value || "";
    const shown = {};
    (qv.match(/[\p{L}\p{N}]+/gu) || []).forEach((orig) => {
      const toks = rawTokens(orig);
      toks.forEach((t) => {
        if (!(t in shown)) shown[t] = toks.length === 1 ? orig.toLowerCase() : t;
      });
    });
    const terms = [];
    const ignored = [];
    const qset = new Set();
    rawTokens(qv).forEach((t) => {
      if (keepTok(t)) {
        if (!qset.has(t)) {
          qset.add(t);
          terms.push(t);
        }
      } else if (ignored.indexOf(t) < 0) ignored.push(t);
    });
    const disp = (t) => shown[t] || t;
    const sug = suggestions();
    els.astrip.hidden = true;
    clearHitDash();
    if (!terms.length) {
      els.aterms.innerHTML = "Nenhuma palavra para procurar. Escreva ao menos uma palavra menos comum" + (sug.length ? ", como " + joinPt(sug.slice(0, 2).map((x) => "<b>" + esc(x.orig) + "</b>")) : "") + ".";
      els.apages.innerHTML = "";
      ASK.hits = [];
      return;
    }
    const hits = bm25(terms);
    ASK.hits = hits;
    els.aterms.innerHTML =
      "Palavras procuradas, sem diferenciar acentos: " + joinPt(terms.map((t) => "<b>" + esc(disp(t)) + "</b>")) + ". " +
      (ignored.length ? "Palavras muito comuns (" + esc(ignored.map(disp).join(", ")) + ") ficam de fora. " : "") +
      (hits.length ? (hits.length === 1 ? "Só uma frase" : "Estas são as " + hits.length + " frases") + ", entre as " + fmtInt(ND) + " da audiência, que mais combinam com essas palavras." : "Nenhuma frase da audiência tem essas palavras.");
    if (!hits.length) {
      els.apages.innerHTML = '<li><p class="wl-ask-empty">Não encontramos essas palavras nesta audiência.</p></li>';
      return;
    }
    placeStrip(hits);
    els.astrip.hidden = false;
    els.astripCap.textContent =
      "A barra é a transcrição inteira, do começo ao fim. Os alfinetes numerados mostram onde está cada frase: " +
      joinPt(hits.map((h, r) => "frase " + (r + 1) + " no turno " + tno(allSents[h.i].turn) + " (" + M.speakerName(allSents[h.i].turn) + ")")) + ".";
    els.apages.innerHTML = hits
      .map((h, r) => {
        const s = allSents[h.i];
        const prev = h.i > 0 && allSents[h.i - 1].turn === s.turn ? allSents[h.i - 1] : null;
        const nxt = h.i + 1 < allSents.length && allSents[h.i + 1].turn === s.turn ? allSents[h.i + 1] : null;
        const has = terms.filter((t) => docs[h.i].indexOf(t) >= 0).map(disp);
        const evs = M.evidenceBySpan[s.start + ":" + s.end] || [];
        const owner = M.ownedTurns[s.turn];
        return (
          '<li class="wl-apage"><span class="wl-apage-no" aria-hidden="true">' + (r + 1) + "</span>" +
          '<p class="wl-apage-h">Frase ' + (r + 1) + " · turno " + tno(s.turn) + " · " + esc(M.speakerName(s.turn)) + "</p>" +
          (prev ? '<p class="wl-apage-ctx" aria-hidden="true">' + esc(tail(prev.text, 140)) + "</p>" : "") +
          '<p class="wl-apage-main"><mark>' + termHtml(s.text, qset) + "</mark></p>" +
          (nxt ? '<p class="wl-apage-ctx" aria-hidden="true">' + esc(shorten(nxt.text, 140)) + "</p>" : "") +
          '<p class="wl-apage-f">Caracteres ' + fmtInt(s.start) + " a " + fmtInt(s.end) + ". Palavras da pergunta nesta frase: " + esc(joinPt(has)) + ".</p>" +
          '<p class="wl-apage-ev"><button type="button" class="wl-evlink" data-readat="' + h.i + '">Ler em volta</button>' +
          (owner && w.OBJ[cadId(owner)] ? '<button type="button" class="wl-evlink" data-cad="' + owner.gi + '" data-sent="' + h.i + '">Ver no caderno de ' + esc(firstName(owner.name)) + "</button>" : "") +
          evs.map((j) => '<button type="button" class="wl-evlink" data-open="' + j + '">Esta frase é o trecho da afirmação ' + (j + 1) + " (" + esc(M.S[j].u.actor.name) + ")</button>").join("") +
          '<span class="wl-evmsg" aria-live="polite"></span></p></li>'
        );
      })
      .join("");
    if (animate && !w.reduced()) {
      w.G.fromTo(els.apages.querySelectorAll(".wl-apage"), { y: -30, rotation: -2, autoAlpha: 0 }, { y: 0, rotation: 0, autoAlpha: 1, duration: 0.5, ease: "power3.out", stagger: 0.1, clearProps: "transform,opacity,visibility" });
      w.G.fromTo(els.astripRow.querySelectorAll(".wl-apin"), { scale: 0 }, { scale: 1, duration: 0.35, ease: "back.out(3)", stagger: 0.06, clearProps: "transform" });
    }
  }

  function clearHitDash() {
    els.objs.querySelectorAll(".wl-cad-d i.is-hit").forEach((d) => d.classList.remove("is-hit"));
  }

  function askMsg(btn, text) {
    els.apages.querySelectorAll(".wl-evmsg").forEach((m) => {
      m.textContent = "";
    });
    const msg = btn && btn.parentNode ? btn.parentNode.querySelector(".wl-evmsg") : null;
    if (msg) msg.textContent = text;
  }

  function showInCad(gi, si, btn) {
    const g = M.GROUPS.find((x) => x.gi === gi);
    if (!g) return;
    w.finishAll();
    if (w.st.auto) w.setAuto(false);
    closeReader(true);
    if (w.st.mode !== "net") {
      w.goNetwork();
      w.finishAll();
    }
    w.st.hist = [];
    w.doPull(cadId(g), { text: "A frase da busca está marcada em vermelho no caderno." }, true);
    clearHitDash();
    const d = g.dashOf(allSents[si].start);
    const el = w.OBJ[cadId(g)].el.querySelector('.wl-cad-d i[data-d="' + d + '"]');
    if (el) el.classList.add("is-hit");
    askMsg(btn, "Aberta na parede, logo acima.");
    scrollWallIntoView();
  }

  function scrollWallIntoView() {
    const r = els.wrap.getBoundingClientRect();
    if (r.top < 0 || r.bottom > window.innerHeight) els.wrap.scrollIntoView({ block: "center", behavior: w.reduced() ? "auto" : "smooth" });
  }

  const rd = { a: 0, b: 0, mark: null, back: null };

  function snapBack(pos) {
    const j = M.sentAt(pos);
    return j >= 0 ? allSents[j].start : 0;
  }
  function snapFwd(pos) {
    for (let i = Math.max(0, M.sentAt(pos)); i < allSents.length; i++) if (allSents[i].end >= pos) return allSents[i].end;
    return T.length;
  }

  function renderReader(keep) {
    const slice = T.slice(rd.a, rd.b);
    const ranges = [{ s: rd.mark.start - rd.a, e: rd.mark.end - rd.a, tag: "mark" }];
    M.turns.forEach((t) => {
      if (t.start <= rd.a || t.start > rd.b) return;
      const hs = T.lastIndexOf("\n", t.start - 1) + 1;
      if (hs < rd.b) ranges.push({ s: Math.max(0, hs - rd.a), e: Math.min(slice.length, t.start - rd.a), cls: "wl-th" });
    });
    const body = els.readerBody;
    const prevH = body.scrollHeight;
    const prevTop = body.scrollTop;
    body.innerHTML = (rd.a > 0 ? "[…]\n" : "") + markText(slice, ranges) + (rd.b < T.length ? "\n[…]" : "");
    els.readerPos.textContent =
      "Caracteres " + fmtInt(rd.a) + " a " + fmtInt(rd.b) + " de " + fmtInt(T.length) + ". Em amarelo, " + rd.mark.label + " (turno " + tno(rd.mark.turn) + ", " + M.speakerName(rd.mark.turn) + "). Em negrito, quem pega a palavra.";
    els.rdBefore.disabled = rd.a <= 0;
    els.rdAfter.disabled = rd.b >= T.length;
    if (keep === "before") body.scrollTop = prevTop + (body.scrollHeight - prevH);
    else if (keep === "after") body.scrollTop = prevTop;
    else {
      const mk = body.querySelector("mark");
      if (mk) body.scrollTop = Math.max(0, mk.getBoundingClientRect().top - body.getBoundingClientRect().top + body.scrollTop - body.clientHeight / 4);
    }
  }

  function openReaderAt(mark, title, from) {
    if (w.st.auto) w.setAuto(false);
    rd.mark = mark;
    rd.back = from || document.activeElement;
    rd.a = snapBack(Math.max(0, mark.start - READ_SPAN));
    rd.b = snapFwd(Math.min(T.length, mark.end + READ_SPAN));
    els.readerH.textContent = title;
    els.reader.hidden = false;
    renderReader();
    try {
      els.rdClose.focus({ preventScroll: true });
    } catch (error) {
      els.rdClose.focus();
    }
  }

  function openReader(k, from) {
    const pa = M.PAS[k];
    if (!pa) return;
    openReaderAt({ start: pa.start, end: pa.end, turn: pa.turn, label: "o trecho escolhido" }, "A transcrição em volta do trecho", from);
  }

  function openReaderSentence(i, from) {
    const s = allSents[i];
    if (!s) return;
    scrollWallIntoView();
    openReaderAt({ start: s.start, end: s.end, turn: s.turn, label: "a frase da busca" }, "A transcrição em volta da frase", from);
  }

  function closeReader(silent) {
    if (els.reader.hidden) return;
    els.reader.hidden = true;
    if (!silent) {
      const b = rd.back && rd.back.isConnected ? rd.back : els.wrap;
      try {
        b.focus({ preventScroll: true });
      } catch (error) {
        b.focus();
      }
    }
  }

  function readMore(dir) {
    if (dir < 0) rd.a = snapBack(Math.max(0, rd.a - READ_STEP));
    else rd.b = snapFwd(Math.min(T.length, rd.b + READ_STEP));
    renderReader(dir < 0 ? "before" : "after");
  }

  function initAsk() {
    const sug = suggestions();
    els.askAud.textContent = "Audiência " + M.H.hearing.id + (M.H.hearing.assunto ? ": " + M.H.hearing.assunto : "") + ".";
    els.askSug.innerHTML = sug.length ? "<span>Experimente:</span>" + sug.map((x) => '<button type="button" data-sug="' + esc(x.orig) + '">' + esc(x.orig) + "</button>").join("") : "";
    els.q.value = sug.length ? "O que foi dito sobre " + joinPt(sug.slice(0, 2).map((x) => x.orig)) + "?" : "";
    runAsk(false);
  }

  function onAskClick(e) {
    const t = e.target;
    const sg = t.closest("[data-sug]");
    if (sg) {
      els.q.value = sg.dataset.sug;
      runAsk(true);
      return true;
    }
    const ra = t.closest("[data-readat]");
    if (ra) {
      openReaderSentence(Number(ra.dataset.readat), ra);
      askMsg(ra, "Aberta na parede, logo acima.");
      return true;
    }
    const cb = t.closest("[data-cad]");
    if (cb) {
      showInCad(Number(cb.dataset.cad), Number(cb.dataset.sent), cb);
      return true;
    }
    const ob = t.closest("[data-open]");
    if (ob) {
      w.openStatementAt(Number(ob.dataset.open), "passage");
      askMsg(ob, "Aberta na parede, logo acima, no trecho escolhido.");
      scrollWallIntoView();
      return true;
    }
    return false;
  }

  Object.assign(w, { runAsk, initAsk, onAskClick, openReader, closeReader, readMore });
}
