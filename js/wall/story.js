import { SUPPORT_NOTE, unitCopy } from "../copy.js";
import { clamp, countLabel, esc, fmtInt, fmtScore, joinPt, plural, tno } from "../text.js";
import { cadId, paId, peId, stId } from "./objects.js";

const SCENE_NAME = { article: "a matéria", who: "quem disse", search: "a busca", passage: "o trecho", result: "o resultado" };
const REST_PHRASE = {
  q: "achamos o começo das aspas na fala",
  h: "achamos na audiência um trecho parecido",
  w: "só achamos algo pouco parecido",
  u: "não achamos essa pessoa entre quem fala",
  n: "não havia frase para comparar",
};

export function installStory(w) {
  const M = w.M;
  const { S, GROUPS, PAS } = M;
  const G = w.G;
  const els = w.els;
  const st = { mode: "rest", i: 0, k: 0, done: S.map(() => false), auto: false, timer: 0, tl: null, last: null, pull: null, follow: null, hist: [], visited: {}, frame: null };
  w.st = st;
  let netMsg = "";

  const cutText = () => fmtScore(M.CUT);
  const scoreOf = (s) => (s.ev && !s.isQuote ? s.ev.score : null);
  const groupById = (gi) => GROUPS.find((g) => g.gi === gi);
  const isCompany = (g) => /empresa|companhia/i.test(g.p.role);
  const eachDash = (g) => (g.bin > 1 ? "cada tracinho reúne até " + g.bin + " frases" : "cada tracinho é uma frase");

  function viewNow() {
    return { mode: st.mode, i: st.i, k: st.k, done: st.done, pull: st.pull, follow: st.follow, all: false };
  }
  function storyView(i, k) {
    return { mode: st.mode === "rest" ? "rest" : "story", i, k, done: st.done, pull: null, follow: null, all: false };
  }

  function sheetPlace(i) {
    const g = M.groupOfSt[i];
    const pe = w.OBJ[peId(g)];
    const sh = w.OBJ.sheet;
    w.fillSheet(S[i]);
    sh.el.style.width = w.L.sheetW + "px";
    sh.el.style.height = "0px";
    const hh = w.measurable() ? Math.max(120, els.sheetIn.scrollHeight + 6) : 300;
    const wd = w.L.sheetW;
    let cx;
    if (w.L.name === "wide") {
      const so = w.OBJ[stId(i)];
      cx = g.side === "R" ? Math.max(so.x + 30 + wd / 2, pe.x + pe.w / 2) : Math.min(so.x + so.w - 30 - wd / 2, pe.x + pe.w / 2);
      if (g.side === "R") cx = Math.min(cx, so.x + so.w + w.L.c2 + w.L.paW - wd / 2 + 20);
      else cx = Math.max(cx, so.x - w.L.c2 - w.L.paW + wd / 2 - 20);
    } else cx = pe.x + pe.w / 2;
    const cy = g.y + g.h / 2;
    const y = clamp(cy - hh / 2, 10, Math.max(10, w.WORLD.H - hh - 10));
    w.setBox(sh, cx - wd / 2, y, wd, hh, -0.8);
  }

  function sceneBeats(s, id) {
    const i = s.i;
    const g = M.groupOfSt[i];
    const peI = peId(g);
    const cad = w.OBJ[cadId(g)] ? cadId(g) : null;
    const B = [];
    if (id === "article") {
      const so = soFarIds(i);
      if (so.length) {
        B.push({
          ids: so,
          ctx: true,
          pre: (tl) => tl.call(() => els.world.classList.add("is-sofar")),
          act: (tl) => tl.to({}, { duration: 1.25 }).call(() => els.world.classList.remove("is-sofar")),
        });
      }
      B.push({ ids: ["artp" + i], act: (tl) => tl.call(() => markSpans(i, true)).to({}, { duration: 0.5 }) });
      B.push({ ids: [stId(i)], act: (tl) => {
        dropIn(tl, w.OBJ[stId(i)]);
        growStr(tl, w.STR_BY["pub" + i], 0.8, "-=0.15");
      } });
    } else if (id === "who") {
      B.push({ ids: [stId(i), peI], fb: [peI], act: (tl) => {
        if (!w.OBJ[peI].el.classList.contains("is-on")) dropIn(tl, w.OBJ[peI]);
        else pluck(tl, w.OBJ[peI]);
        growStr(tl, w.STR_BY["att" + i], 0.75, "-=0.1");
        if (!s.resolved) for (let j = 0; j < 3; j++) growStr(tl, w.STR_BY["loose" + g.gi + "_" + j], 0.5, j ? "-=0.35" : "+=0.1");
      } });
      if (s.resolved && cad) {
        B.push({ ids: [peI, cad], fb: [cad], act: (tl) => {
          const x = w.STR_BY["fala" + g.gi];
          if (x && !x.on) growStr(tl, x, 0.7, "+=0.05");
          else tl.call(() => w.kick(x, 6), null, "+=0.05").to({}, { duration: 0.1 });
          tl.call(() => {
            w.OBJ[cad].el.dataset.v = "cur";
          }).fromTo(w.OBJ[cad].inn, { rotation: (w.OBJ[cad].r || 0) - 3 }, { rotation: w.OBJ[cad].r || 0, duration: 0.5, ease: "elastic.out(1.4, 0.35)", clearProps: "transform" }).to({}, { duration: 0.2 });
        } });
      }
    } else if (id === "search") {
      if (cad) B.push({ ids: [cad], act: (tl) => searchCad(tl, s) });
      B.push({ ids: ["sheet"], act: (tl) => sheetIn(tl, s) });
    } else if (id === "passage") {
      const pa = paId(s.pa);
      if (cad) B.push({ ids: [cad], act: (tl) => sheetOut(tl) });
      else B.push({ ids: [pa], act: (tl) => sheetOut(tl) });
      B.push({ ids: cad ? [pa, cad] : [pa], fb: [pa], act: (tl) => {
        passageIn(tl, s);
        if (w.STR_BY["esta" + s.pa]) growStr(tl, w.STR_BY["esta" + s.pa], 0.6, "-=0.1");
      } });
      B.push({ ids: [stId(i), pa], fb: [pa], act: (tl) => growStr(tl, w.STR_BY["sus" + i], 0.8, "+=0.05") });
    } else if (id === "result") {
      const ids = [stId(i)];
      if (s.pa != null) ids.push(paId(s.pa));
      if (!s.resolved) ids.push(peI);
      B.push({ ids: w.fits(ids) ? ids : [stId(i)], act: (tl) => stampIn(tl, s) });
    }
    return B;
  }

  function mergeBeats(B) {
    const out = [];
    B.forEach((b0) => {
      let b = b0;
      if (b.ctx) {
        out.push({ ids: b.ids.slice(), beats: [b], ctx: true });
        return;
      }
      if (b.fb && !w.fits(b.ids)) b = { ids: b.fb, act: b.act, pre: b.pre };
      const last = out[out.length - 1];
      if (last && !last.ctx && w.fits(last.ids.concat(b.ids))) {
        last.ids = last.ids.concat(b.ids);
        last.beats.push(b);
      } else out.push({ ids: b.ids.slice(), beats: [b] });
    });
    return out;
  }

  function soFarIds(i) {
    const ids = [];
    S.forEach((s, j) => {
      if (j === i || !st.done[j]) return;
      const g = M.groupOfSt[j];
      ids.push("artp" + j, stId(j), peId(g));
      if (s.pa != null) ids.push(paId(s.pa));
      if (w.OBJ[cadId(g)]) ids.push(cadId(g));
    });
    if (ids.length) ids.push("artp" + i);
    return ids;
  }

  function keyIds(s, id) {
    if (id === "who") return [peId(M.groupOfSt[s.i])];
    if (id === "search") return ["sheet"];
    if (id === "passage") return [paId(s.pa)];
    return [stId(s.i)];
  }

  function prepScene(i, k) {
    const s = S[i];
    const id = s.scenes[k];
    if (id === "search" || id === "passage") sheetPlace(i);
    const B = sceneBeats(s, id);
    const gr = mergeBeats(B);
    const key = keyIds(s, id);
    const keyFits = w.fits(key);
    if (id === "article" || id === "who") {
      const all = [];
      const last = gr[gr.length - 1];
      B.forEach((b) => {
        if (!b.ctx) b.ids.forEach((x) => {
          if (all.indexOf(x) < 0) all.push(x);
        });
      });
      const covered = all.every((x) => last.ids.indexOf(x) >= 0);
      if (!covered && (w.L.name === "tall" || !w.fits(last.ids)) && w.frameIds(all).k >= 0.3) {
        if (keyFits) gr[gr.length - 1] = { ids: all, beats: last.beats, ctx: true };
        else gr.push({ ids: all, beats: [], ctx: true });
      }
    }
    const fin = gr[gr.length - 1];
    if (keyFits && (!key.every((x) => fin.ids.indexOf(x) >= 0) || !w.fits(fin.ids))) gr.push({ ids: key, beats: [] });
    else if (!keyFits && !w.fits(fin.ids)) gr.push({ ids: [(id === "result" ? "end:" : "top:") + key[0]], beats: [] });
    return gr;
  }

  function sceneFinalIds(i, k) {
    const gr = prepScene(i, k);
    return gr[gr.length - 1].ids;
  }

  function dropIn(tl, o) {
    tl.call(() => {
      o.el.classList.add("is-on");
      o.el.dataset.v = "cur";
    }).fromTo(o.inn, { y: -46, scale: 1.1, autoAlpha: 0 }, { y: 0, scale: 1, autoAlpha: 1, duration: 0.55, ease: "back.out(1.6)", clearProps: "transform,opacity,visibility" });
    const pins = Array.from(o.el.querySelectorAll(".wl-pin"));
    if (pins.length) tl.fromTo(pins, { y: -26, scale: 1.8, autoAlpha: 0 }, { y: 0, scale: 1, autoAlpha: 1, duration: 0.4, ease: "bounce.out", stagger: 0.08, clearProps: "transform,opacity,visibility" }, "-=0.18");
  }

  function pluck(tl, o) {
    tl.fromTo(o.inn, { rotation: (o.r || 0) - 2.5 }, { rotation: o.r || 0, duration: 0.5, ease: "elastic.out(1.4, 0.35)", clearProps: "transform" }).call(
      () => (w.STR_OF[o.id] || []).forEach((x) => {
        if (x.on) w.kick(x, 7);
      }),
      null,
      "<",
    );
  }

  function growStr(tl, x, dur, pos) {
    if (!x) return;
    tl.call(
      () => {
        x.on = true;
        x.g.classList.add("is-on");
        x.g.setAttribute("data-v", "cur");
        const tv = w.tagSet(viewNow())[x.id];
        if (tv) w.setTag(x, tv);
        x.prog = Math.min(x.prog, 0.001);
        w.drawStr(x);
      },
      null,
      pos,
    )
      .to(x, { prog: 1, duration: dur, ease: "power2.out", onUpdate: () => w.drawStr(x) })
      .call(() => w.kick(x, 6));
  }

  function markSpans(i, on) {
    w.OBJ.art.el.querySelectorAll(".wl-u, .wl-nm").forEach((sp) => {
      if ((" " + sp.dataset.s + " ").indexOf(" " + i + " ") >= 0) {
        sp.classList.toggle("is-on", on);
        sp.classList.toggle("is-cur", on);
      }
    });
  }

  function candidateDashes(s) {
    const g = M.groupOfSt[s.i];
    const out = {};
    if (!g.resolved) return out;
    if (s.isQuote && s.ev) out[g.dashOf(s.ev.start_char)] = 1;
    else s.u.candidates.slice(0, 8).forEach((c) => {
      if (c.start !== null) out[g.dashOf(c.start)] = 1;
    });
    return out;
  }

  function searchCad(tl, s) {
    const g = M.groupOfSt[s.i];
    const o = w.OBJ[cadId(g)];
    const box = o.el.querySelector(".wl-cad-d");
    const dashes = Array.from(box.querySelectorAll("i"));
    const cand = candidateDashes(s);
    const hits = dashes.filter((d) => cand[d.dataset.d]);
    if (dashes.length) tl.fromTo(dashes, { backgroundColor: "#A9A59B" }, { backgroundColor: "#1D2F5C", duration: 0.12, stagger: { amount: 0.6 }, yoyo: true, repeat: 1, clearProps: "backgroundColor" });
    tl.call(() => {
      hits.forEach((d) => d.classList.add("is-c"));
      box.classList.add("is-fade");
    });
    if (hits.length) tl.fromTo(hits, { scale: 3.4 }, { scale: 1.9, duration: 0.35, stagger: 0.05, clearProps: "transform" });
    tl.to({}, { duration: 0.3 });
  }

  function clearSearchMarks() {
    els.objs.querySelectorAll(".wl-cad-d i.is-c").forEach((d) => d.classList.remove("is-c"));
    els.objs.querySelectorAll(".wl-cad-d.is-fade").forEach((d) => d.classList.remove("is-fade"));
  }

  function searchMarksNow(s) {
    const g = M.groupOfSt[s.i];
    const o = w.OBJ[cadId(g)];
    if (o) {
      const box = o.el.querySelector(".wl-cad-d");
      const cand = candidateDashes(s);
      box.classList.add("is-fade");
      box.querySelectorAll("i").forEach((x) => {
        if (cand[x.dataset.d]) x.classList.add("is-c");
      });
    }
    const list = w.OBJ.sheet.el.querySelector(".wl-slips");
    if (list) list.classList.add("is-fade");
  }

  function sheetIn(tl, s) {
    const o = w.OBJ.sheet;
    tl.call(() => {
      o.el.classList.add("is-on");
      o.el.dataset.v = "cur";
    }).fromTo(o.inn, { y: -60, rotation: -4, autoAlpha: 0 }, { y: 0, rotation: -0.8, autoAlpha: 1, duration: 0.6, ease: "power3.out", clearProps: "transform,opacity,visibility" });
    const box = els.sheetIn;
    if (s.isQuote) {
      const qws = Array.from(box.querySelectorAll(".wl-qm-art .wl-qw"));
      const sws = Array.from(box.querySelectorAll(".wl-qm-sent .wl-qw"));
      const n = Math.min(qws.length, sws.length);
      const cnt = box.querySelector(".wl-qm-n");
      if (cnt) tl.set(cnt, { autoAlpha: 0 });
      if (qws.length + sws.length) tl.set(qws.concat(sws), { backgroundSize: "0% 100%" });
      for (let j = 0; j < n; j++) tl.to([qws[j], sws[j]], { backgroundSize: "100% 100%", duration: 0.13, ease: "none" }, j === 0 ? "+=0.2" : ">-0.02");
      if (cnt) tl.to(cnt, { autoAlpha: 1, duration: 0.3, clearProps: "opacity,visibility" });
      tl.call(() => qws.concat(sws).forEach((el) => {
        el.style.backgroundSize = "";
      }));
    } else {
      const list = box.querySelector(".wl-slips");
      const li = Array.from(box.querySelectorAll(".wl-slips li"));
      if (li.length) {
        tl.fromTo(li, { autoAlpha: 0, x: -14 }, { autoAlpha: 1, x: 0, duration: 0.26, stagger: 0.07, clearProps: "transform,opacity,visibility" }, "-=0.1")
          .call(() => list.classList.add("is-fade"), null, "+=0.35")
          .fromTo(box.querySelector("li.is-best") || li[0], { scale: 1 }, { scale: 1.03, duration: 0.25, yoyo: true, repeat: 1, clearProps: "transform" });
      }
    }
  }

  function sheetOut(tl) {
    const o = w.OBJ.sheet;
    tl.to(o.inn, { y: -70, rotation: 3, autoAlpha: 0, duration: 0.45, ease: "power2.in" }).call(() => {
      o.el.classList.remove("is-on");
      G.set(o.inn, { clearProps: "transform,opacity,visibility" });
    });
  }

  function passageIn(tl, s) {
    const o = w.OBJ[paId(s.pa)];
    const g = PAS[s.pa].group;
    const cad = g && w.OBJ[cadId(g)];
    if (!o.el.classList.contains("is-on")) {
      const fx = cad ? cad.x + cad.w / 2 : o.x + o.w / 2;
      const fy = cad ? cad.y + (w.CAD_Y[s.pa] || cad.h / 2) : o.y;
      const dx = fx - (o.x + o.w / 2);
      const dy = fy - (o.y + o.ah / 2);
      tl.call(() => {
        o.el.classList.add("is-on");
        o.el.dataset.v = "cur";
      }).fromTo(o.inn, { x: dx, y: dy, scale: 0.25, autoAlpha: 0 }, { x: 0, y: 0, scale: 1, autoAlpha: 1, duration: 0.85, ease: "power3.inOut", clearProps: "transform,opacity,visibility" });
    } else pluck(tl, o);
    tl.call(() => o.el.classList.add("is-readable")).to({}, { duration: 0.35 });
  }

  function stampIn(tl, s) {
    const o = w.OBJ[stId(s.i)];
    const ink = o.el.querySelector(".wl-ink");
    const res = o.el.querySelector(".wl-st-res");
    const mk = o.el.querySelector(".wl-rul-mk");
    tl.call(() => o.el.classList.add("is-inked"));
    if (res) tl.fromTo(res, { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.2, clearProps: "opacity,visibility" });
    if (ink) tl.fromTo(ink, { scale: 1.6, rotation: -14, autoAlpha: 0 }, { scale: 1, rotation: -5, autoAlpha: 1, duration: 0.28, ease: "power4.in", clearProps: "transform,opacity,visibility" }, "<");
    tl.fromTo(o.inn, { y: 0 }, { y: 3, duration: 0.07, yoyo: true, repeat: 1, clearProps: "transform" });
    if (mk && scoreOf(s) != null) tl.fromTo(mk, { left: "0%" }, { left: (clamp(s.ev.score, 0, 1) * 100).toFixed(1) + "%", duration: 0.8, ease: "power2.out" }, "-=0.1");
    if (s.pa != null) tl.call(() => w.kick(w.STR_BY["sus" + s.i], 5));
    tl.addLabel("shown");
    tl.to({}, { duration: 0.2 });
  }

  function captionFor(s, id) {
    const u = s.u;
    const name = u.actor.name;
    const ev = s.ev;
    const g = M.groupOfSt[s.i];
    const dateBit = M.DATE ? " de " + M.DATE : "";
    if (id === "article") {
      const others = S.filter((o) => o !== s && o.u.proposition === u.proposition).map((o) => o.u.actor.name);
      const nDone = st.done.filter((d, j) => d && j !== s.i).length;
      return (
        (nDone ? "A parede já tem " + countLabel(nDone, "afirmação ligada", "afirmações ligadas") + " por fios. " : "") +
        "A matéria" + dateBit + " publica esta afirmação e a atribui a " + name + (others.length ? "; a mesma frase também é atribuída a " + joinPt(others) : "") + "." +
        (nDone ? "" : " O fio claro e fino liga a matéria ao cartão.")
      );
    }
    if (id === "who") {
      const earlier = g.sts.filter((j) => j !== s.i && st.done[j]);
      const pre = earlier.length ? name + " já estava na parede; o cartão ganha mais um fio vermelho. " : "O fio vermelho liga o cartão a " + name + ". ";
      if (!s.resolved) {
        const who = isCompany(g) ? "essa empresa" : "essa pessoa";
        const mt = s.mentions.turns;
        let tailText = "";
        if (mt.length) tailText = " O nome aparece na fala de outras pessoas" + (mt.length <= 3 ? " (" + plural(mt.length, "turno ", "turnos ") + joinPt(mt.map((t) => String(tno(t)))) + ")" : "") + (s.spot.viaNote ? ", e a matéria cita uma nota." : ".");
        else if (s.spot.viaNote) tailText = " A matéria cita uma nota.";
        return pre + "Procuramos " + who + " entre quem fala nos " + fmtInt(M.NT) + " turnos da transcrição e não achamos, então os fios ficam soltos. Pode ser que não tenha falado ou que apareça com outro nome." + tailText;
      }
      const n = s.turns.length;
      if (earlier.length) return pre + "O caderno de " + name + " já está ligado.";
      return pre + name + " fala em " + (n === 1 ? "um dos " : n + " dos ") + fmtInt(M.NT) + " turnos da audiência. O caderno ao lado junta essas falas, na ordem em que aconteceram, e " + eachDash(g) + "; o fio pontilhado liga os dois.";
    }
    if (id === "search") {
      const U = unitCopy(M.RUN);
      const N = U.count(s.nSent);
      const each = U.window ? " com cada um dos " : " com cada uma das ";
      if (!ev) return name + " fala em " + countLabel(s.turns.length, "turno", "turnos") + ", mas nenhuma frase tem pelo menos 4 palavras. Não há frase para comparar com o cartão.";
      if (s.isQuote) return "A matéria põe a frase entre aspas. Então procuramos o começo das aspas, palavra por palavra, " + (U.window ? "nos " : "nas ") + N + " de " + name + ".";
      if (s.qsem) {
        const qs = s.qsem;
        return (
          (qs.np && qs.run >= qs.np
            ? "O começo das aspas, “" + qs.prefix + "”, aparece na fala, mas é curto; então comparamos pelo sentido"
            : "A matéria põe a frase entre aspas, mas o começo das aspas não aparece igual na fala; então comparamos pelo sentido") +
          each + N + " de " + name + "."
        );
      }
      const top = Math.min(8, u.candidates.length);
      return "Comparamos o sentido do cartão" + each + N + " de " + name + ". " + U.top(top) + plural(top, " fica", " ficam") + " em vermelho no caderno; " + (U.window ? "os outros apagam." : "as outras apagam.");
    }
    if (id === "passage") {
      const t = tno(ev.speaker_turn);
      const shared = PAS[s.pa].sts.filter((j) => j !== s.i && st.done[j]);
      const line = s.tier.k === "q" ? "O fio verde, grosso e duplo," : s.tier.k === "h" ? "O fio azul" : "O fio tracejado, mais fraco,";
      return (
        (shared.length ? "É o mesmo trecho da " + plural(shared.length, "afirmação ", "afirmações ") + joinPt(shared.map((j) => String(j + 1))) + ". " : "Este é o trecho escolhido, no turno " + t + ", fala de " + M.speakerName(ev.speaker_turn) + ". ") +
        line + " liga o trecho ao cartão; o preto, de traço e ponto, mostra onde ele está no caderno de " + name + "."
      );
    }
    const k = s.tier.k;
    const sup = supportTail(s);
    if (k === "q") return "Resultado: " + w.quoteCount(s.quote).toLowerCase() + " entre aspas aparecem iguais na fala de " + name + ", no turno " + tno(ev.speaker_turn) + "; aqui não se mede semelhança." + sup + " Compare o cartão com o trecho em amarelo.";
    if (k === "h") return "Resultado: achamos um trecho parecido, com semelhança " + fmtScore(ev.score, M.CUT) + ", de 0 a 1." + sup + " Compare o cartão com o trecho em amarelo.";
    if (k === "w") return "Resultado: só achamos algo pouco parecido. A semelhança, " + fmtScore(ev.score, M.CUT) + ", ficou abaixo de " + cutText() + ", o ponto a partir do qual chamamos de parecido." + sup;
    if (k === "u") return "Resultado: não achamos " + (isCompany(g) ? "essa empresa" : "essa pessoa") + " entre quem fala, então não há fala para comparar." + (s.spot.viaNote ? " A matéria relata uma nota." : "");
    return "Resultado: não havia frase de " + name + " para comparar.";
  }

  function supportTail(s) {
    return s.sup ? " O verificador dá " + s.sup.b.label + " ao trecho (" + s.sup.text + "); " + SUPPORT_NOTE.charAt(0).toLowerCase() + SUPPORT_NOTE.slice(1) : "";
  }

  function restCaption() {
    const s = S[0];
    const sc = scoreOf(s) != null ? ", com semelhança " + fmtScore(s.ev.score, M.CUT) + " (de 0 a 1; não é a chance de a afirmação estar certa)" : "";
    return "A matéria atribui uma afirmação a " + s.u.actor.name + ", e " + REST_PHRASE[s.tier.k] + sc + ". Aperte Começar para ver os fios sendo amarrados.";
  }

  function weakTail(s) {
    return "com semelhança " + fmtScore(s.ev.score, M.CUT) + ", abaixo de " + cutText() + ": só achamos algo pouco parecido";
  }

  function summary(id) {
    if (!id) return "";
    if (id === "art") return "A matéria" + (M.DATE ? " de " + M.DATE : "") + " publica " + countLabel(S.length, "afirmação", "afirmações") + ", atribuídas a " + countLabel(GROUPS.length, "pessoa ou entidade", "pessoas ou entidades") + ".";
    const o = w.OBJ[id];
    if (!o) return "";
    if (o.type === "st") {
      const s = S[Number(id.slice(2))];
      const r = "Afirmação " + (s.i + 1) + ", atribuída a " + s.u.actor.name + ". ";
      if (!s.resolved) return r + "Não achamos " + (isCompany(M.groupOfSt[s.i]) ? "essa empresa" : "essa pessoa") + " entre quem fala.";
      if (!s.ev) return r + "Não havia frase para comparar.";
      if (s.tier.k === "w") return r + "Ligada ao trecho mais parecido do turno " + tno(s.ev.speaker_turn) + ", " + weakTail(s) + ".";
      return r + "Ligada a um trecho do turno " + tno(s.ev.speaker_turn) + ": " + s.tier.label.toLowerCase() + (scoreOf(s) != null ? ", semelhança " + fmtScore(s.ev.score, M.CUT) : "") + "." + supportTail(s);
    }
    if (o.type === "pe") {
      const g = groupById(Number(id.slice(2)));
      const n = g.sts.length;
      if (!g.resolved) return g.name + ": " + countLabel(n, "afirmação atribuída", "afirmações atribuídas") + ". Não achamos a fala na transcrição, então os fios ficam soltos.";
      const withPa = g.sts.filter((j) => S[j].pa != null).length;
      return g.name + ": " + countLabel(n, "afirmação atribuída", "afirmações atribuídas") + ", fala em " + countLabel(g.turns.length, "turno", "turnos") + ", " + countLabel(g.pas.length, "trecho ligado", "trechos ligados") + " às afirmações" + (g.pas.length < withPa ? "; um trecho está ligado a mais de uma afirmação." : ".");
    }
    if (o.type === "cad") {
      const g = groupById(Number(id.slice(3)));
      return "Falas de " + g.name + ": " + countLabel(g.turns.length, "turno", "turnos") + " e " + countLabel(g.sents.length, "frase", "frases") + ", na ordem da audiência; " + eachDash(g) + ". " + (g.pas.length ? countLabel(g.pas.length, "trecho escolhido está", "trechos escolhidos estão") + " marcados em amarelo." : "Nenhum trecho escolhido está aqui.");
    }
    if (o.type === "pa") {
      const pa = PAS[Number(id.slice(2))];
      const strong = pa.sts.filter((j) => S[j].tier.k !== "w");
      const weak = pa.sts.filter((j) => S[j].tier.k === "w");
      const num = (j) => String(j + 1);
      const parts = [];
      if (strong.length) parts.push("é o trecho escolhido para " + plural(strong.length, "a afirmação ", "as afirmações ") + joinPt(strong.map(num)));
      if (weak.length) parts.push("é o mais parecido que achamos para " + plural(weak.length, "a afirmação ", "as afirmações ") + joinPt(weak.map(num)) + ", mas a semelhança ficou abaixo de " + cutText());
      return "Trecho do turno " + tno(pa.turn) + ", fala de " + M.speakerName(pa.turn) + ": " + parts.join("; ") + ".";
    }
    return "";
  }

  function summaryShort(id) {
    const o = w.OBJ[id];
    if (o && o.type === "st") {
      const s = S[Number(id.slice(2))];
      return "Afirmação " + (s.i + 1) + ", atribuída a " + s.u.actor.name + ".";
    }
    if (o && o.type === "pa") {
      const pa = PAS[Number(id.slice(2))];
      return "Trecho do turno " + tno(pa.turn) + ", fala de " + M.speakerName(pa.turn) + ".";
    }
    return summary(id);
  }

  function nameOfObj(id) {
    if (id === "art") return "a matéria";
    const o = w.OBJ[id];
    if (!o) return "";
    if (o.type === "st") return "a afirmação " + (Number(id.slice(2)) + 1);
    if (o.type === "pe") return groupById(Number(id.slice(2))).name;
    if (o.type === "cad") return "o caderno de " + groupById(Number(id.slice(3))).name;
    if (o.type === "pa") return "o trecho do turno " + tno(PAS[Number(id.slice(2))].turn);
    return "";
  }

  function followSentence(x) {
    if (x.type === "pub") return "Fio “a matéria publica”: a matéria publica a afirmação " + (x.st + 1) + ".";
    if (x.type === "att") return "Fio “atribuída a”: a matéria atribui a afirmação " + (x.st + 1) + " a " + S[x.st].u.actor.name + ".";
    if (x.type === "fala") return "Fio “" + x.label + "”: as falas de " + groupById(x.gi).name + " estão neste caderno.";
    if (x.type === "sus") {
      const s = S[x.st];
      if (s.tier.k === "w") return "Fio tracejado: liga a afirmação " + (x.st + 1) + " ao trecho mais parecido que achamos, " + weakTail(s) + ".";
      return "Fio “" + x.label + "”: liga a afirmação " + (x.st + 1) + " a este trecho; " + s.tier.label.toLowerCase() + (scoreOf(s) != null ? ", semelhança " + fmtScore(s.ev.score, M.CUT) : "") + ".";
    }
    if (x.type === "esta") {
      const pa = PAS[x.pa];
      return "Fio “está no turno " + tno(pa.turn) + "”: este trecho está no turno " + tno(pa.turn) + ", fala de " + M.speakerName(pa.turn) + ".";
    }
    if (x.type === "loose") return "Fio solto: não achamos a fala de " + groupById(x.gi).name + " na transcrição, então este fio não chega a nenhum caderno.";
    return "";
  }

  function netIntro() {
    return "<b>Puxe um fio:</b> toque em um cartão para ver só ele e suas ligações. <b>Siga o barbante:</b> toque em um fio ou aperte Próximo." + (w.overviewPartial ? " Arraste a parede para ver o resto da rede." : "");
  }

  function renderUi() {
    const s = S[st.i];
    if (st.mode === "net") {
      els.capScene.innerHTML =
        'A rede inteira<span class="wl-cs-w">: ' + esc(countLabel(S.length, "afirmação", "afirmações") + ", " + countLabel(GROUPS.length, "pessoa ou entidade", "pessoas ou entidades") + ", " + countLabel(PAS.length, "trecho", "trechos")) +
        '</span><span class="wl-cs-n"> · ' + esc(countLabel(S.length, "afirmação", "afirmações")) + "</span>";
      els.capText.innerHTML = st.pull ? esc(netMsg) : netIntro();
    } else if (st.mode === "rest") {
      els.capScene.textContent = "Um exemplo pronto";
      els.capText.textContent = restCaption();
    } else if (st.k >= 0) {
      const id = s.scenes[st.k];
      els.capScene.innerHTML = "Afirmação " + (st.i + 1) + " de " + S.length + ' · <span class="wl-cs-w">Cena ' + (st.k + 1) + " de " + s.scenes.length + ": </span>" + SCENE_NAME[id];
      els.capText.textContent = captionFor(s, id);
    }
    let label;
    if (st.mode === "rest") label = "Começar";
    else if (st.mode === "net") label = "Próximo";
    else if (st.k < w.lastK(s)) label = "Próximo";
    else if (st.i < S.length - 1) label = "Próxima afirmação";
    else label = "Ver a rede inteira";
    els.next.textContent = label;
    els.back.disabled = st.mode === "rest" || (st.mode === "story" && st.i === 0 && st.k <= 0);
    els.all.textContent = st.mode === "net" ? "Voltar à história" : "Ver a rede inteira";
    els.auto.innerHTML = st.auto ? "Pausar" : 'Tocar <span class="wl-a2">sozinho</span>';
    els.auto.setAttribute("aria-pressed", st.auto ? "true" : "false");
    els.pick.value = st.mode === "net" ? "" : String(st.i);
    renderCaseLink();
    fitCaption();
  }

  function caseClaim() {
    if (st.mode === "net") return st.pull && /^st\d+$/.test(st.pull) ? Number(st.pull.slice(2)) : -1;
    if (st.mode !== "net" && st.k >= 0 && S[st.i].scenes[st.k] === "result") return st.i;
    return -1;
  }

  function renderCaseLink() {
    const j = caseClaim();
    const link = els.kase;
    link.hidden = j < 0;
    if (j < 0) {
      link.removeAttribute("href");
      return;
    }
    link.href = "#h" + M.H.hearing.id + "-u" + (j + 1);
    link.setAttribute("aria-label", "Abrir a pasta da afirmação " + (j + 1));
  }

  function fitCaption() {
    els.cap.removeAttribute("data-fit");
    if (!w.measurable()) return;
    for (let f = 1; f <= 4 && els.capText.scrollHeight > els.capText.clientHeight + 1; f++) els.cap.setAttribute("data-fit", String(f));
  }

  function run(tl, done) {
    st.tl = tl;
    w.root.dataset.busy = "1";
    tl.eventCallback("onComplete", () => {
      if (st.tl !== tl) return;
      st.tl = null;
      delete w.root.dataset.busy;
      done();
    });
  }
  function busy() {
    if (!st.tl) delete w.root.dataset.busy;
    return !!st.tl;
  }
  function finishAll() {
    let n = 0;
    while (st.tl && n++ < 8) {
      const t = st.tl;
      t.progress(1);
      if (st.tl === t) {
        st.tl = null;
        t.kill();
      }
    }
    if (!st.tl) delete w.root.dataset.busy;
    if (w.camTween) {
      w.camTween.progress(1);
      w.killCamTween();
    }
  }

  function applyState(i, k, keepCam) {
    if (st.tl) {
      const old = st.tl;
      st.tl = null;
      old.kill();
    }
    delete w.root.dataset.busy;
    st.i = i;
    st.k = k;
    els.world.classList.remove("is-sofar");
    if (G && G.set) els.objs.querySelectorAll(".wl-o-in, .wl-pin, .wl-ink, .wl-st-res").forEach((el) => G.set(el, { clearProps: "transform,opacity,visibility" }));
    clearSearchMarks();
    const s = S[i];
    if (k >= 0) {
      const id = s.scenes[k];
      if (id === "search" || id === "passage") sheetPlace(i);
      if (id === "search") searchMarksNow(s);
    }
    w.applyWall(storyView(i, k), true);
    if (!keepCam && k >= 0) {
      const ids = st.mode === "rest" ? restIds() : sceneFinalIds(i, k);
      st.frame = ids;
      w.setCam(w.frameIds(ids));
    }
    w.redrawAll();
    renderUi();
  }

  function restIds() {
    const s = S[0];
    const st0 = stId(0);
    const g = M.groupOfSt[0];
    const pe = peId(g);
    const pa = s.pa != null ? paId(s.pa) : null;
    const cad = w.OBJ[cadId(g)] && s.pa != null ? cadId(g) : null;
    const tries = [["artp0", st0, pe, pa, cad], [st0, pe, pa, cad], [st0, pa, cad], [st0, pe, pa], [st0, pa], [st0, pe], [st0]];
    for (let j = 0; j < tries.length; j++) {
      const ids = tries[j].filter(Boolean);
      if (w.fits(ids)) return ids;
    }
    return ["end:" + st0];
  }

  function buildScene(i, k) {
    const gr = prepScene(i, k);
    const tl = G.timeline();
    const ref = { v: w.camView(w.cam) };
    tl.call(() => clearSearchMarks());
    gr.forEach((g) => {
      g.beats.forEach((b) => {
        if (b.pre) b.pre(tl);
      });
      w.camStep(tl, g.ids, ref);
      g.beats.forEach((b) => b.act(tl));
    });
    st.frame = gr[gr.length - 1].ids;
    if (tl.labels.shown == null) tl.addLabel("shown", tl.duration());
    return tl;
  }

  function playScene(i, k) {
    clearTimeout(st.timer);
    st.mode = "story";
    st.i = i;
    st.k = k;
    if (w.reduced()) {
      applyState(i, k);
      afterScene();
      return;
    }
    const prevK = k - 1;
    w.applyWall(storyView(i, prevK), true);
    if (prevK >= 0 && S[i].scenes[prevK] === "search") {
      searchMarksNow(S[i]);
      w.OBJ.sheet.el.classList.add("is-on");
    }
    st.k = k;
    renderUi();
    run(buildScene(i, k), () => {
      applyState(i, k, true);
      afterScene();
    });
  }

  function complete() {
    return st.mode === "rest" || (st.mode === "story" && st.k === w.lastK(S[st.i]));
  }

  function goStatement(j, jump) {
    finishAll();
    clearTimeout(st.timer);
    w.closeReader(true);
    if (st.mode === "net") {
      st.pull = null;
      st.follow = null;
    } else if (complete() && st.i !== j) st.done[st.i] = st.mode !== "rest";
    if (st.mode === "rest") st.done = S.map(() => false);
    if (jump) for (let x = 0; x < j; x++) st.done[x] = true;
    st.done[j] = false;
    st.mode = "story";
    if (w.reduced()) {
      applyState(j, 0);
      afterScene();
      return;
    }
    w.applyWall(storyView(j, -1), true);
    playScene(j, 0);
  }

  function cutTo(i, k) {
    st.mode = "story";
    applyState(i, k);
    afterScene();
  }

  function contentShown(t) {
    const sh = t.labels && t.labels.shown != null ? t.labels.shown : t.duration();
    return t.time() >= sh - 0.05 || t.duration() - t.time() < 0.35;
  }

  function next() {
    clearTimeout(st.timer);
    if (busy()) {
      const shown = st.mode === "story" && contentShown(st.tl);
      finishAll();
      clearTimeout(st.timer);
      if (!shown) return;
    }
    w.closeReader(true);
    if (st.mode === "net") {
      netNext();
      return;
    }
    if (st.mode === "rest") {
      st.done = S.map(() => false);
      goStatement(0);
      return;
    }
    const s = S[st.i];
    if (st.k < w.lastK(s)) {
      playScene(st.i, st.k + 1);
      return;
    }
    if (st.i < S.length - 1) {
      goStatement(st.i + 1);
      return;
    }
    st.done[st.i] = true;
    goNetwork();
  }

  function back() {
    clearTimeout(st.timer);
    finishAll();
    w.closeReader(true);
    if (st.mode === "net") {
      netBack();
      return;
    }
    if (st.mode !== "story") return;
    if (st.k > 0) {
      cutTo(st.i, st.k - 1);
      return;
    }
    if (st.i > 0) {
      st.done[st.i - 1] = false;
      cutTo(st.i - 1, w.lastK(S[st.i - 1]));
    }
  }

  function holdMs() {
    return clamp(2600 + (els.capText.textContent || "").length * 36, 3600, 9000);
  }

  function afterScene() {
    clearTimeout(st.timer);
    if (!st.auto || w.destroyed) return;
    st.timer = setTimeout(() => {
      if (!st.auto || w.destroyed) return;
      next();
    }, holdMs());
  }

  function setAuto(on) {
    st.auto = on;
    clearTimeout(st.timer);
    renderUi();
    if (!on) return;
    w.closeReader(true);
    if (st.mode === "rest") {
      st.done = S.map(() => false);
      goStatement(0);
    } else if (!busy()) next();
  }

  function goNetwork() {
    finishAll();
    clearTimeout(st.timer);
    w.closeReader(true);
    if (st.mode !== "net") st.last = { mode: st.mode, i: st.i, k: Math.max(0, st.k), done: st.done.slice() };
    st.mode = "net";
    st.pull = null;
    st.follow = null;
    st.hist = [];
    st.visited = {};
    const before = {};
    w.STR.forEach((x) => {
      before[x.id] = x.on;
    });
    w.applyWall(viewNow(), w.reduced());
    st.frame = ["all"];
    const t = w.frameIds(["all"]);
    if (w.overviewPartial) w.showHint("Arraste a parede para ver o resto da rede");
    if (w.reduced()) {
      w.setCam(t);
      renderUi();
      afterScene();
      return;
    }
    const tl = G.timeline();
    const ref = { v: w.camView(w.cam) };
    w.camStep(tl, ["all"], ref);
    const fresh = w.STR.filter((x) => x.on && !before[x.id]);
    fresh.forEach((x) => {
      x.prog = 0;
    });
    if (fresh.length) tl.to(fresh, { prog: 1, duration: 0.5, ease: "power1.out", stagger: { amount: 0.9 }, onUpdate: () => fresh.forEach(w.drawStr) }, "-=0.4");
    renderUi();
    run(tl, () => {
      fresh.forEach((x) => {
        x.prog = 1;
        w.drawStr(x);
      });
      afterScene();
    });
  }

  function leaveNetwork() {
    finishAll();
    const l = st.last || { mode: "rest", i: 0, k: w.lastK(S[0]), done: S.map(() => false) };
    st.pull = null;
    st.follow = null;
    st.done = l.done.slice();
    st.mode = l.mode === "rest" ? "rest" : "story";
    w.hideHint();
    if (st.mode === "rest") {
      applyState(0, w.lastK(S[0]));
      return;
    }
    cutTo(l.i, l.k);
  }

  function neighborsIds(id) {
    const ids = [id];
    (w.STR_OF[id] || []).forEach((x) => {
      if (!x.on) return;
      [x.a, x.b].forEach((e) => {
        if (!e) return;
        const v = e === "art" && x.type === "pub" ? "artp" + x.st : e;
        if (ids.indexOf(v) < 0) ids.push(v);
      });
    });
    return ids;
  }

  function pullFrame(id) {
    if (id === "art") return ["art"];
    if (w.OBJ[id]) w.OBJ[id].ls = 1.045;
    let ids = neighborsIds(id);
    if (w.fitBox(w.unionBox(ids, id), 1.3).k < (w.L.name === "tall" ? 0.6 : 0.3)) ids = [id];
    return ids;
  }

  function pullCam(id, ids) {
    const o = w.OBJ[id];
    if (!o) return w.fitBox(w.unionBox(ids), 1.3);
    const lsMax = w.L.name === "tall" ? 2.05 : 3.8;
    o.ls = 1.045;
    let t = w.fitBox(w.unionBox(ids, id), 1.3);
    if (id !== "art" && t.k < w.TH - 0.004) {
      for (let it = 0; it < 6; it++) {
        o.ls = clamp((w.TH + 0.01) / t.k, 1.045, lsMax);
        t = w.fitBox(w.unionBox(ids, id), 1.3);
      }
      if (t.k * o.ls < w.TH - 0.004) {
        o.ls = 1.045;
        t = w.fitBox(w.unionBox(["top:" + id]), 1.3);
      }
    }
    o.el.style.setProperty("--ls", o.ls.toFixed(3));
    return t;
  }

  function doPull(id, sentence, instant) {
    st.pull = id;
    st.follow = sentence && sentence.sid ? sentence.sid : null;
    netMsg = (sentence && sentence.text ? sentence.text + " " : "") + (sentence && sentence.short ? summaryShort(id) : summary(id));
    const ids = pullFrame(id);
    st.frame = ids;
    const t = pullCam(id, ids);
    w.applyWall(viewNow(), false);
    (w.STR_OF[id] || []).forEach((x) => {
      if (x.on) w.kick(x, 4);
    });
    if (instant || w.reduced()) w.setCam(t);
    else w.camTo(t);
    renderUi();
  }

  function stringsOf(id) {
    const order = { pub: 0, att: 1, sus: 2, fala: 3, esta: 4, loose: 5 };
    return (w.STR_OF[id] || [])
      .filter((x) => x.on)
      .slice()
      .sort((a, b) => order[a.type] - order[b.type] || (a.st || 0) - (b.st || 0) || (a.pa || 0) - (b.pa || 0) || (a.j || 0) - (b.j || 0));
  }

  function otherEnd(x, from) {
    return x.a === from ? x.b : x.a;
  }

  function netFollow(x, fromId) {
    finishAll();
    let from = fromId || st.pull;
    if (!from || (x.a !== from && x.b !== from)) from = x.a;
    const to = otherEnd(x, from);
    st.visited[x.id] = true;
    st.hist.push(from);
    if (!to) {
      doPull(from, { text: followSentence(x), sid: x.id });
      return;
    }
    st.pull = from;
    st.follow = x.id;
    w.applyWall(viewNow(), false);
    w.kick(x, 8);
    const short = x.type === "sus" || x.type === "esta";
    if (w.reduced()) {
      doPull(to, { text: followSentence(x), sid: x.id, short }, true);
      return;
    }
    const tl = G.timeline();
    const ref = { v: w.camView(w.cam) };
    const both = [from, to];
    w.camStep(tl, w.fitBox(w.unionBox(both), 1.2).k >= 0.3 ? both : [to], ref);
    tl.to({}, { duration: 0.35 });
    run(tl, () => {
      doPull(to, { text: followSentence(x), sid: x.id, short });
      afterScene();
    });
    netMsg = followSentence(x);
    els.capText.textContent = netMsg;
    fitCaption();
  }

  function netNext() {
    if (!st.pull) {
      st.hist = [];
      st.visited = {};
      doPull("art", { text: "Começamos pela matéria, no centro da parede." });
      afterScene();
      return;
    }
    const cand = stringsOf(st.pull).filter((x) => !st.visited[x.id]);
    if (cand.length) {
      netFollow(cand[0], st.pull);
      return;
    }
    while (st.hist.length) {
      const prev = st.hist.pop();
      if (stringsOf(prev).some((x) => !st.visited[x.id])) {
        doPull(prev, { text: "Voltamos para " + nameOfObj(prev) + ", que ainda tem fios para seguir." });
        afterScene();
        return;
      }
    }
    st.visited = {};
    st.hist = [];
    doPull("art", { text: "Você passou por todos os fios da rede. Recomeçamos pela matéria." });
    afterScene();
  }

  function unpull() {
    st.pull = null;
    st.follow = null;
    st.hist = [];
    w.applyWall(viewNow(), false);
    st.frame = ["all"];
    w.camTo(w.frameIds(["all"]));
    renderUi();
  }

  function netBack() {
    if (st.hist.length) {
      const prev = st.hist.pop();
      doPull(prev, { text: "De volta a " + nameOfObj(prev) + "." });
      return;
    }
    if (st.pull) {
      unpull();
      return;
    }
    leaveNetwork();
  }

  function pickStatement(j) {
    if (st.mode === "net") {
      finishAll();
      st.done = (st.last ? st.last.done : st.done).slice();
      st.mode = "story";
      st.pull = null;
    }
    goStatement(j, true);
  }

  function openStatementAt(j, sceneId) {
    finishAll();
    clearTimeout(st.timer);
    if (st.auto) setAuto(false);
    w.closeReader(true);
    if (st.mode === "net") {
      st.pull = null;
      st.follow = null;
      st.done = (st.last ? st.last.done : st.done).slice();
    } else if (complete() && st.i !== j && st.mode !== "rest") st.done[st.i] = true;
    if (st.mode === "rest") st.done = S.map(() => false);
    for (let x = 0; x < j; x++) st.done[x] = true;
    st.done[j] = false;
    const s = S[j];
    const k = s.scenes.indexOf(sceneId);
    cutTo(j, k < 0 ? w.lastK(s) : k);
  }

  function pullFromUser(id) {
    finishAll();
    clearTimeout(st.timer);
    if (st.auto) setAuto(false);
    if (st.pull && st.pull !== id) st.hist.push(st.pull);
    doPull(id, null);
  }

  function followFromUser(sid) {
    if (!w.STR_BY[sid]) return;
    clearTimeout(st.timer);
    if (st.auto) setAuto(false);
    netFollow(w.STR_BY[sid], st.pull);
  }

  function reapply() {
    if (st.mode === "net") {
      const t0 = st.pull ? pullCam(st.pull, st.frame) : null;
      w.applyWall(viewNow(), true);
      w.setCam(t0 || w.frameIds(st.frame || ["all"]));
      renderUi();
      return;
    }
    applyState(st.i, st.k);
  }

  function refit() {
    const t = st.mode === "net" && st.pull ? pullCam(st.pull, st.frame) : w.frameIds(st.frame || ["all"]);
    w.setCam(t);
    fitCaption();
  }

  function start() {
    st.mode = "rest";
    st.i = 0;
    st.k = w.lastK(S[0]);
  }

  function stopStory() {
    clearTimeout(st.timer);
    st.auto = false;
    if (st.tl) {
      st.tl.kill();
      st.tl = null;
    }
  }

  Object.assign(w, {
    renderUi, fitCaption, busy, finishAll, next, back, setAuto, goNetwork, leaveNetwork, pickStatement, openStatementAt,
    pullFromUser, followFromUser, unpull, doPull, reapply, refit, start, stopStory, applyState, summary,
  });
}
