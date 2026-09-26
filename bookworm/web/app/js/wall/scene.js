import { cadId, paId, peId, stId } from "./objects.js";

export function installScene(w) {
  const M = w.M;
  const { S } = M;
  w.VIEW = null;

  function sceneIdx(s, id) {
    const i = s.scenes.indexOf(id);
    return i < 0 ? Infinity : i;
  }
  function lastK(s) {
    return s.scenes.length - 1;
  }
  function lvOf(view, j) {
    if (view.mode === "net" || view.all || view.done[j]) return Infinity;
    if (j === view.i) return view.k;
    return -1;
  }
  function scLv(view, j, id) {
    return lvOf(view, j) >= sceneIdx(S[j], id);
  }

  function groupStrings(g, s) {
    s["fala" + g.gi] = 1;
    for (let j = 0; j < 3; j++) s["loose" + g.gi + "_" + j] = 1;
  }

  function curSets(view) {
    const o = {};
    const s = {};
    if (view.mode === "net") {
      if (!view.pull) return null;
      o[view.pull] = 1;
      (w.STR_OF[view.pull] || []).forEach((x) => {
        if (!x.on) return;
        s[x.id] = 1;
        if (x.a) o[x.a] = 1;
        if (x.b) o[x.b] = 1;
      });
      return { o, s };
    }
    const i = view.i;
    const st = S[i];
    const g = M.groupOfSt[i];
    o.art = 1;
    o[stId(i)] = 1;
    o[peId(g)] = 1;
    if (w.OBJ[cadId(g)]) o[cadId(g)] = 1;
    groupStrings(g, s);
    s["pub" + i] = 1;
    s["att" + i] = 1;
    if (st.pa != null) {
      o[paId(st.pa)] = 1;
      s["sus" + i] = 1;
      s["esta" + st.pa] = 1;
    }
    if (view.k >= sceneIdx(st, "search")) o.sheet = 1;
    return { o, s };
  }

  function tagSet(view) {
    const t = {};
    if (view.mode === "net") {
      if (view.pull) {
        const byType = {};
        (w.STR_OF[view.pull] || []).forEach((x) => {
          if (x.on && x.label) (byType[x.type] = byType[x.type] || []).push(x);
        });
        Object.keys(byType).forEach((ty) => {
          const arr = byType[ty];
          if (arr.length <= 4) arr.forEach((x) => {
            t[x.id] = 1;
          });
          else t[arr[Math.floor(arr.length / 2)].id] = ty === "esta" ? arr.length + " trechos neste caderno" : ty === "pub" ? "a matéria publica " + arr.length + " afirmações" : arr[0].label;
        });
      }
      if (view.follow) t[view.follow] = 1;
      return t;
    }
    const i = view.i;
    const st = S[i];
    const g = M.groupOfSt[i];
    let id = st.scenes[Math.max(0, view.k)];
    if (view.mode === "rest") id = "rest";
    if (id === "article" || id === "rest") t["pub" + i] = 1;
    if (id === "who" || id === "rest") {
      t["att" + i] = 1;
      t["fala" + g.gi] = 1;
      t["loose" + g.gi + "_1"] = 1;
    }
    if (id === "passage" || id === "result" || id === "rest") {
      t["sus" + i] = 1;
      if (st.pa != null) t["esta" + st.pa] = 1;
    }
    if (id === "result") t["loose" + g.gi + "_1"] = 1;
    return t;
  }

  function setTag(x, tv) {
    x.tagOn = !!tv;
    if (!x.tag) return;
    const txt = typeof tv === "string" ? tv : x.label;
    x.tagP = typeof tv === "string" ? 0.45 : null;
    if (x.tag.textContent !== txt) {
      x.tag.textContent = txt;
      x.tag.setAttribute("aria-label", "Seguir o fio: " + txt);
      x.tw = 0;
    }
  }

  function applyWall(view, instant) {
    w.VIEW = view;
    const i = view.i;
    const curS = S[i];
    const onO = { art: 1, note: 1 };
    M.GROUPS.forEach((g) => {
      if (w.OBJ[cadId(g)]) onO[cadId(g)] = 1;
    });
    S.forEach((s, j) => {
      if (!(lvOf(view, j) >= 0)) return;
      onO[stId(j)] = 1;
      if (scLv(view, j, "who")) onO[peId(M.groupOfSt[j])] = 1;
      if (s.pa != null && scLv(view, j, "passage")) onO[paId(s.pa)] = 1;
    });
    if (view.mode !== "net" && view.k >= 0 && curS.scenes[view.k] === "search") onO.sheet = 1;
    const onS = {};
    w.STR.forEach((x) => {
      let on = false;
      if (x.type === "pub") on = !!onO[stId(x.st)];
      else if (x.type === "att") on = scLv(view, x.st, "who");
      else if (x.type === "sus") on = scLv(view, x.st, "passage");
      else if (x.type === "fala" || x.type === "loose") on = !!onO["pe" + x.gi];
      else if (x.type === "esta") on = !!onO[paId(x.pa)];
      onS[x.id] = on;
    });
    const linked = {};
    w.STR.forEach((x) => {
      if (!onS[x.id]) return;
      if (x.a) linked[x.a] = 1;
      if (x.b) linked[x.b] = 1;
    });
    const cur = view.all ? null : curSets(view);
    const tags = view.all ? {} : tagSet(view);
    w.els.world.classList.toggle("is-net", view.mode === "net");
    w.els.world.classList.toggle("is-pull", view.mode === "net" && !!view.pull);
    w.els.wrap.dataset.mode = view.mode;
    const reduced = w.reduced();
    w.OBJ_LIST.forEach((o) => {
      const on = !!onO[o.id];
      const el = o.el;
      el.classList.toggle("is-on", on);
      let v = "cur";
      if (view.mode === "net") v = cur ? (cur.o[o.id] ? "cur" : "faint") : "cur";
      else v = !cur || cur.o[o.id] ? "cur" : "dim";
      if (o.type === "cad" && !linked[o.id]) v = v === "faint" ? "faint" : "pale";
      el.dataset.v = v;
      const hb = el.querySelector(".wl-hit");
      if (hb) hb.tabIndex = view.mode === "net" && on ? 0 : -1;
      if (o.type === "st") el.classList.toggle("is-inked", scLv(view, Number(o.id.slice(2)), "result"));
      if (o.type === "pa") {
        const k = Number(o.id.slice(2));
        let readable = false;
        if (view.mode === "net") readable = view.pull === o.id;
        else readable = curS.pa === k && view.k >= sceneIdx(curS, "passage");
        el.classList.toggle("is-readable", readable);
      }
      const pulled = view.mode === "net" && view.pull === o.id;
      o.pulled = pulled;
      if (pulled) el.style.setProperty("--ls", (o.ls || 1.045).toFixed(3));
      el.classList.toggle("is-pulled", pulled);
      if (instant || reduced) o.liftT = o.lift = pulled ? 1 : 0;
      else w.setLift(o, pulled ? 1 : 0);
    });
    w.STR.forEach((x) => {
      const on = onS[x.id];
      x.on = on;
      x.g.classList.toggle("is-on", !!on);
      if (instant || reduced) x.prog = on ? 1 : 0;
      else if (!on) x.prog = 0;
      const v = view.mode === "net" ? (cur ? (cur.s[x.id] ? "cur" : "faint") : "cur") : !cur || cur.s[x.id] ? "cur" : "dim";
      x.g.setAttribute("data-v", v);
      const lit = view.mode === "net" && cur && cur.s[x.id];
      const layer = lit ? w.els.svHi : w.els.svLo;
      if (x.g.parentNode !== layer) layer.appendChild(x.g);
      x.mT = lit ? 0.22 : 1;
      if (instant || reduced) x.m = x.mT;
      else if (Math.abs(x.m - x.mT) > 0.01) w.markLive(x);
      setTag(x, tags[x.id]);
      if (x.tag) {
        x.tag.tabIndex = view.mode === "net" && x.tagOn ? 0 : -1;
        x.tag.classList.toggle("is-follow", view.follow === x.id);
      }
      x.g.classList.toggle("is-follow", view.follow === x.id);
    });
    w.els.art.querySelectorAll(".wl-u, .wl-nm").forEach((sp) => {
      const ids = sp.dataset.s.split(" ").map(Number);
      let on = false;
      let isCur = false;
      ids.forEach((j) => {
        if (onO[stId(j)]) on = true;
        if (view.mode !== "net" && j === i && view.k >= 0) isCur = true;
        if (view.mode === "net" && view.pull === stId(j)) isCur = true;
      });
      sp.classList.toggle("is-on", on);
      sp.classList.toggle("is-cur", isCur);
    });
    const legendHidden = !((view.mode === "net" && !view.pull) || view.all);
    if (w.els.legend.hidden !== legendHidden) {
      w.els.legend.hidden = legendHidden;
      w.resetZoomBounds();
    }
    w.redrawAll();
    w.startLoop();
  }

  Object.assign(w, { sceneIdx, lastK, lvOf, scLv, curSets, tagSet, setTag, applyWall });
}
