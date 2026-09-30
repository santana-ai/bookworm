import { clamp, countLabel, tno } from "../text.js";
import { cadId, paId, peId, stId } from "./objects.js";

const SVGNS = "http://www.w3.org/2000/svg";
const TAG_ORDER = { att: 0, sus: 1, esta: 2, fala: 3, pub: 4, loose: 5 };

function f1(v) {
  return v.toFixed(1);
}

function bez(g, t) {
  const u = 1 - t;
  return { x: u * u * g.A.x + 2 * u * t * g.C.x + t * t * g.B.x, y: u * u * g.A.y + 2 * u * t * g.C.y + t * t * g.B.y };
}

export function installStrings(w) {
  const M = w.M;
  const { S, GROUPS, PAS } = M;
  w.STR = [];
  w.STR_BY = {};
  w.STR_OF = {};
  let dcNeed = false;
  let dcBatch = false;

  function mkStr(id, type, a, b, extra) {
    const s = { id, type, a, b, prog: 0, m: 1, mT: 1, wob: 0, ph: (w.STR.length * 1.7) % 6.28, on: false, tagOn: false, k: null };
    if (extra) Object.assign(s, extra);
    const g = document.createElementNS(SVGNS, "g");
    g.setAttribute("class", "wl-s");
    g.setAttribute("data-t", type);
    if (s.k) g.setAttribute("data-k", s.k);
    g.setAttribute("data-sid", id);
    const parts = type === "loose" ? ["sh", "ln", "fray", "hit"] : ["sh", "ln", "l2", "hit"];
    s.paths = {};
    parts.forEach((p) => {
      const el = document.createElementNS(SVGNS, "path");
      el.setAttribute("class", p === "fray" ? "wl-fray" : p);
      g.appendChild(el);
      s.paths[p] = el;
    });
    ["ta", "tb"].forEach((t) => {
      if (type === "loose" && t === "tb") return;
      const c = document.createElementNS(SVGNS, "circle");
      c.setAttribute("class", "wl-tack");
      c.setAttribute("r", "3.2");
      g.appendChild(c);
      s[t] = c;
    });
    g.style.display = "none";
    w.els.svLo.appendChild(g);
    s.g = g;
    if (s.label) {
      const tag = document.createElement("button");
      tag.type = "button";
      tag.className = "wl-tag";
      tag.dataset.t = type;
      if (s.k) tag.dataset.k = s.k;
      tag.dataset.sid = id;
      tag.tabIndex = -1;
      tag.textContent = s.label;
      tag.setAttribute("aria-label", "Seguir o fio: " + s.label);
      w.els.tags.appendChild(tag);
      s.tag = tag;
    }
    w.STR.push(s);
    w.STR_BY[id] = s;
    return s;
  }

  S.forEach((s) => {
    const g = M.groupOfSt[s.i];
    mkStr("pub" + s.i, "pub", "art", stId(s.i), { st: s.i, label: "a matéria publica" });
    mkStr("att" + s.i, "att", stId(s.i), peId(g), { st: s.i, label: "atribuída a" });
    if (s.pa != null) mkStr("sus" + s.i, "sus", stId(s.i), paId(s.pa), { st: s.i, k: s.tier.k, label: s.tier.tag || "trecho" });
  });
  GROUPS.forEach((g) => {
    if (g.resolved && w.OBJ[cadId(g)]) mkStr("fala" + g.gi, "fala", peId(g), cadId(g), { gi: g.gi, label: "fala em " + countLabel(g.turns.length, "turno", "turnos"), tagT: 0.55 });
    if (!g.resolved) for (let j = 0; j < 3; j++) mkStr("loose" + g.gi + "_" + j, "loose", peId(g), null, { gi: g.gi, j, label: j === 1 ? "fala não achada" : null, tagT: 0.62 });
  });
  PAS.forEach((pa) => {
    if (pa.group && w.OBJ[cadId(pa.group)]) mkStr("esta" + pa.k, "esta", paId(pa.k), cadId(pa.group), { pa: pa.k, label: "está no turno " + tno(pa.turn), tagT: 0.5 });
  });
  w.STR.forEach((s) => {
    if (!s.tagT) s.tagT = s.type === "pub" ? 0.72 : 0.5;
  });
  w.STR.forEach((s) => {
    [s.a, s.b].forEach((id) => {
      if (id) (w.STR_OF[id] = w.STR_OF[id] || []).push(s);
    });
  });

  function groupOfObj(objId) {
    const o = w.OBJ[objId];
    if (!o) return null;
    const n = Number(objId.replace(/^\D+/, ""));
    if (o.type === "st") return M.groupOfSt[n];
    if (o.type === "pe" || o.type === "cad") return GROUPS.find((x) => x.gi === n) || null;
    if (o.type === "pa") return PAS[n].group;
    return null;
  }

  function sideOf(objId, role) {
    const g = groupOfObj(objId);
    const sd = g && g.side ? g.side : "R";
    if (role === "in") return sd === "R" ? "L" : "R";
    return sd;
  }

  function refY(objId, s) {
    if (objId === "art") return w.ART.pins[s.st] ? w.ART.pins[s.st].y : w.OBJ.art.y;
    const o = w.OBJ[objId];
    return o.y + (o.ah || o.h) / 2;
  }

  function buildAnchors() {
    const slots = {};
    w.STR.forEach((s) => {
      [["a", s.a, s.b], ["b", s.b, s.a]].forEach((e) => {
        const id = e[1];
        if (!id || id === "art") return;
        let role = e[0] === "a" ? "out" : "in";
        if (s.type === "pub") role = "in";
        const side = sideOf(id, role);
        s[e[0] + "Side"] = side;
        if (w.OBJ[id].type === "cad") return;
        const key = id + ":" + side;
        (slots[key] = slots[key] || []).push({ s, end: e[0], other: e[2] });
      });
    });
    Object.keys(slots).forEach((key) => {
      const list = slots[key];
      list.forEach((x) => {
        x.oy = x.s.type === "loose" ? w.OBJ[x.s.a].y + 40 + x.s.j * 30 : refY(x.other, x.s);
      });
      list.sort((a, b) => a.oy - b.oy || (a.s.id < b.s.id ? -1 : 1));
      const n = list.length;
      const lo = n > 3 ? 0.14 : 0.24;
      const hi = 1 - lo;
      list.forEach((x, j) => {
        x.s[x.end + "F"] = n === 1 ? 0.5 : lo + ((hi - lo) * j) / (n - 1);
      });
    });
  }

  function objPoint(o, px, py) {
    const cx = o.x + o.w / 2;
    const cy = o.y + o.h / 2;
    const a = ((o.r || 0) * Math.PI) / 180;
    const dx = px - cx;
    const dy = py - cy;
    let rx = cx + dx * Math.cos(a) - dy * Math.sin(a);
    let ry = cy + dx * Math.sin(a) + dy * Math.cos(a);
    if (o.lift) {
      const ox = o.x + o.w / 2;
      const oy = o.y + o.h * 0.4;
      const sc = 1 + ((o.ls || 1.045) - 1) * o.lift;
      rx = ox + (rx - ox) * sc;
      ry = oy + (ry - oy) * sc - 7 * o.lift;
    }
    return { x: rx, y: ry };
  }

  function endPt(s, end) {
    const id = end === "a" ? s.a : s.b;
    if (id === "art") {
      const pin = w.ART.pins[s.st] || { x: w.OBJ.art.x + w.OBJ.art.w, y: w.OBJ.art.y + 40 };
      return { x: pin.x, y: pin.y };
    }
    const o = w.OBJ[id];
    const side = s[end + "Side"];
    const hh = o.ah || o.h;
    let y;
    if (o.type === "cad") y = s.type === "esta" ? o.y + clamp(w.CAD_Y[s.pa] || hh * 0.6, 8, hh - 8) : o.y + Math.min(24, hh * 0.3);
    else y = o.y + hh * (s[end + "F"] == null ? 0.5 : s[end + "F"]);
    return objPoint(o, side === "L" ? o.x - 3 : o.x + o.w + 3, y);
  }

  function strGeom(s) {
    const A = endPt(s, "a");
    if (s.type === "loose") {
      const dir = s.aSide === "L" ? -1 : 1;
      const B = { x: A.x + dir * (44 + s.j * 16), y: A.y + 60 + s.j * 22 };
      const C = { x: A.x + dir * (16 + s.j * 6) + s.wob * Math.sin(s.ph), y: B.y + 4 };
      return { A, B, C };
    }
    const B = endPt(s, "b");
    const d = Math.hypot(B.x - A.x, B.y - A.y);
    const sag = Math.min(60, 9 + d * 0.1) * s.m;
    const C = { x: (A.x + B.x) / 2 + s.wob * Math.sin(s.ph), y: (A.y + B.y) / 2 + 2 * sag + s.wob * 0.4 * Math.cos(s.ph * 1.3) };
    return { A, B, C };
  }

  function drawStr(s) {
    if (!s.on && s.prog <= 0.001) {
      if (s.g.style.display !== "none") s.g.style.display = "none";
      if (s.tag) s.tag.classList.remove("is-on");
      return;
    }
    s.g.style.display = "";
    const g = strGeom(s);
    const t = clamp(s.prog, 0, 1);
    let d;
    if (t >= 0.999) d = "M" + f1(g.A.x) + " " + f1(g.A.y) + " Q" + f1(g.C.x) + " " + f1(g.C.y) + " " + f1(g.B.x) + " " + f1(g.B.y);
    else {
      const p1 = { x: g.A.x + (g.C.x - g.A.x) * t, y: g.A.y + (g.C.y - g.A.y) * t };
      const p2 = bez(g, t);
      d = "M" + f1(g.A.x) + " " + f1(g.A.y) + " Q" + f1(p1.x) + " " + f1(p1.y) + " " + f1(p2.x) + " " + f1(p2.y);
    }
    s.paths.sh.setAttribute("d", d);
    s.paths.ln.setAttribute("d", d);
    if (s.paths.l2) s.paths.l2.setAttribute("d", s.type === "att" || (s.type === "sus" && s.k === "q") ? d : "");
    s.paths.hit.setAttribute("d", t >= 0.999 ? d : "");
    if (s.paths.fray) {
      const e = bez(g, t);
      s.paths.fray.setAttribute("d", t >= 0.999 ? "M" + f1(e.x) + " " + f1(e.y) + " l3 7 M" + f1(e.x) + " " + f1(e.y) + " l-3 6 M" + f1(e.x) + " " + f1(e.y) + " l1 8" : "");
    }
    s.ta.setAttribute("cx", f1(g.A.x));
    s.ta.setAttribute("cy", f1(g.A.y));
    if (s.tb) {
      s.tb.setAttribute("cx", f1(g.B.x));
      s.tb.setAttribute("cy", f1(g.B.y));
      s.tb.style.display = t >= 0.999 ? "" : "none";
    }
    if (s.tag) {
      const tp = s.tagP != null ? s.tagP : s.tagT;
      if (s.tagOn && t >= Math.min(tp, 0.999)) {
        const tt = s.tagPick != null && s.tagPick <= t ? s.tagPick : tp;
        const p = bez(g, tt);
        s.tag.style.left = f1(p.x) + "px";
        s.tag.style.top = f1(p.y) + "px";
        if (!s.tag.classList.contains("is-on")) {
          s.tag.classList.add("is-on", "is-clip");
          s.tw = 0;
          s.tagPick = null;
          dcNeed = true;
        }
      } else if (s.tag.classList.contains("is-on")) {
        s.tag.classList.remove("is-on");
        s.tagPick = null;
        dcNeed = true;
      }
    }
    if (dcNeed && !dcBatch) {
      dcNeed = false;
      declutter();
    }
  }

  function obstacles(wr) {
    const out = [];
    const nearZ = w.cam.k >= w.TH - 0.004;
    const push = (r, ins) => {
      if (r.width > 0) out.push({ l: r.left - wr.left + ins, t: r.top - wr.top + ins, r: r.right - wr.left - ins, b: r.bottom - wr.top - ins });
    };
    w.OBJ_LIST.forEach((o) => {
      const el = o.el;
      if (!el.classList.contains("is-on") || el.dataset.v === "faint") return;
      if (o.type === "sheet") {
        push(el.getBoundingClientRect(), 0);
        return;
      }
      if (nearZ || o.big) {
        if (o.type !== "art") push(o.inn.getBoundingClientRect(), Math.min(26, 16 * w.cam.k * (o.big ? o.ls : 1)));
        return;
      }
      if (o.type === "art" && !w.els.world.classList.contains("z-tiny")) return;
      const far = o.far || (o.far = Array.from(el.querySelectorAll(".wl-far > *")));
      far.forEach((x) => {
        if (x.offsetWidth) push(x.getBoundingClientRect(), 1);
      });
    });
    return out;
  }

  function hitAny(r, list) {
    for (let i = 0; i < list.length; i++) {
      const o = list[i];
      if (r.l < o.r && r.r > o.l && r.t < o.b && r.b > o.t) return true;
    }
    return false;
  }

  function declutter() {
    const tags = w.STR.filter((x) => x.tag && x.tag.classList.contains("is-on"));
    if (!tags.length || !w.SW) return;
    const wr = w.els.wrap.getBoundingClientRect();
    const obs = obstacles(wr);
    const placed = [];
    const fol = w.VIEW && w.VIEW.follow;
    tags.sort((a, b) => (b.id === fol) - (a.id === fol) || TAG_ORDER[a.type] - TAG_ORDER[b.type] || (a.id < b.id ? -1 : 1));
    tags.forEach((x) => {
      const br = x.tag.getBoundingClientRect();
      x.tw = br.width || 110;
      x.th = br.height || 23;
      const g = strGeom(x);
      const base = x.tagP != null ? x.tagP : x.tagT;
      const cands = [base, 0.5, 0.38, 0.62, 0.28, 0.72, 0.2, 0.8, 0.88, 0.12];
      let hit = null;
      for (let c = 0; c < cands.length && !hit; c++) {
        const t = cands[c];
        if (t > x.prog + 1e-6) continue;
        const p = bez(g, t);
        const sx = p.x * w.cam.k + w.cam.x;
        const sy = p.y * w.cam.k + w.cam.y;
        const r = { l: sx - x.tw / 2 - 2, t: sy + 4, r: sx + x.tw / 2 + 2, b: sy + 5 + x.th };
        if (r.l < 2 || r.r > w.SW - 2 || r.t < 2 || r.b > w.SH - 2) continue;
        if (hitAny(r, obs) || hitAny(r, placed)) continue;
        hit = { p, r, t };
      }
      if (hit) {
        x.tagPick = hit.t;
        x.tag.style.left = f1(hit.p.x) + "px";
        x.tag.style.top = f1(hit.p.y) + "px";
        x.tag.classList.remove("is-clip");
        placed.push(hit.r);
      } else x.tag.classList.add("is-clip");
    });
  }

  function batchDraw(fn) {
    dcBatch = true;
    try {
      fn();
    } finally {
      dcBatch = false;
    }
    declutter();
    dcNeed = false;
  }

  function redrawAll() {
    batchDraw(() => w.STR.forEach(drawStr));
  }

  function setBatch(on) {
    dcBatch = on;
    if (!on) dcNeed = false;
  }

  Object.assign(w, { buildAnchors, strGeom, drawStr, declutter, batchDraw, redrawAll, setBatch, groupOfObj });
}
