import { clamp, esc, tilt } from "../text.js";
import { cadId, paId, peId, stId } from "./objects.js";

const LP = {
  wide: {
    name: "wide", wings: 2, stW: 304, stMin: 214, stGap: 16, peW: 262, peMin: 118, paW: 300, cadW: 150, noteW: 240,
    c1: 100, c2: 62, c3: 88, gGap: 66, midGap: 20, top: 78, bottom: 56, side: 110, artColW: 312, artPad: 22, sheetW: 540,
    th: 1.09, maxK: 1.45, pad: 18, overK: 0.12,
  },
  tall: {
    name: "tall", wings: 1, stW: 304, stMin: 214, stGap: 16, peW: 262, peMin: 118, paW: 300, cadW: 150, noteW: 240,
    c1: 84, c2: 52, c3: 72, gGap: 60, midGap: 18, top: 70, bottom: 40, side: 24, artColW: 300, artPad: 18, sheetW: 300,
    th: 1.09, maxK: 1.35, pad: 10, overK: 0.2,
  },
};

function perms(a) {
  if (a.length <= 1) return [a.slice()];
  const out = [];
  a.forEach((x, i) => {
    const rest = a.slice(0, i).concat(a.slice(i + 1));
    perms(rest).forEach((p) => out.push([x].concat(p)));
  });
  return out;
}

function crossCount(segs) {
  let c = 0;
  for (let i = 0; i < segs.length; i++) {
    for (let j = i + 1; j < segs.length; j++) {
      const a = segs[i];
      const b = segs[j];
      if (a[0] !== b[0] && a[1] !== b[1] && (a[0] - b[0]) * (a[1] - b[1]) < 0) c++;
    }
  }
  return c;
}

function planGroup(M, g) {
  const { S } = M;
  const mids = g.pas.map((k) => "pa" + k);
  mids.push("pe");
  if (g.sts.length > 4 || mids.length > 5) {
    const st = g.sts.slice().sort((a, b) => S[a].spot.pos - S[b].spot.pos || a - b);
    const rank = {};
    st.forEach((i, r) => {
      rank[i] = r;
    });
    const bary = (id) => {
      if (id === "pe") return (st.length - 1) / 2;
      const k = Number(id.slice(2));
      const users = M.PAS[k].sts.filter((i) => rank[i] != null);
      return users.reduce((a, i) => a + rank[i], 0) / Math.max(1, users.length);
    };
    const mo = mids.slice().sort((a, b) => bary(a) - bary(b) || (a === "pe" ? -1 : b === "pe" ? 1 : 0));
    g.stOrder = st;
    g.midOrder = mo;
    return;
  }
  let best = null;
  perms(g.sts).forEach((sp) => {
    perms(mids).forEach((mp) => {
      const ns = sp.length;
      const nm = mp.length;
      const yS = (i) => (sp.indexOf(i) + 0.5) / ns;
      const yM = (id) => (mp.indexOf(id) + 0.5) / nm;
      const segA = [];
      const segB = [];
      sp.forEach((i) => segA.push([S[i].spot.pos, yS(i)]));
      sp.forEach((i) => {
        segB.push([yS(i), yM("pe")]);
        if (S[i].pa != null && mp.indexOf("pa" + S[i].pa) >= 0) segB.push([yS(i), yM("pa" + S[i].pa)]);
      });
      const cost = crossCount(segA) * 1.2 + crossCount(segB) + Math.abs(yM("pe") - 0.5) * 0.3;
      if (!best || cost < best.cost - 1e-9) best = { cost, sp, mp };
    });
  });
  g.stOrder = best ? best.sp : g.sts.slice();
  g.midOrder = best ? best.mp : mids;
}

export function installLayout(w) {
  const M = w.M;
  const { S, GROUPS, PAS, article } = M;
  GROUPS.forEach((g) => planGroup(M, g));
  w.L = null;
  w.TH = 1.09;
  w.WORLD = { W: 1000, H: 800 };
  w.ART = { blocks: {}, pins: {} };
  w.CAD_Y = {};
  w.OTH = null;

  function measurable() {
    return !!w.root.isConnected && w.els.wrap.offsetWidth > 0;
  }

  function naturalHeight(o, width, fallback) {
    if (!measurable()) return fallback;
    const near = o.el.querySelector(".wl-near");
    o.el.style.width = width + "px";
    o.el.style.height = "10px";
    return Math.ceil(near.scrollHeight) + 2;
  }

  function setBox(o, x, y, wd, h, r) {
    o.x = x;
    o.y = y;
    o.w = wd;
    o.h = h;
    o.r = r || 0;
    o.ah = h;
    o.el.style.left = x.toFixed(1) + "px";
    o.el.style.top = y.toFixed(1) + "px";
    o.el.style.width = wd.toFixed(1) + "px";
    o.el.style.height = h.toFixed(1) + "px";
    o.el.style.setProperty("--r", o.r + "deg");
  }

  function measureAll() {
    const P = w.L;
    const hs = {};
    S.forEach((s) => {
      hs[stId(s.i)] = Math.max(P.stMin, naturalHeight(w.OBJ[stId(s.i)], P.stW, P.stMin + Math.ceil(s.u.proposition.length / 36) * 19));
    });
    GROUPS.forEach((g) => {
      hs[peId(g)] = Math.max(P.peMin, naturalHeight(w.OBJ[peId(g)], P.peW, P.peMin));
      if (w.OBJ[cadId(g)]) hs[cadId(g)] = Math.max(90, naturalHeight(w.OBJ[cadId(g)], P.cadW, 70 + Math.ceil(g.nDash / 22) * 4.5));
    });
    PAS.forEach((pa) => {
      hs[paId(pa.k)] = Math.max(90, naturalHeight(w.OBJ[paId(pa.k)], P.paW, 88 + Math.ceil(Math.min(pa.text.length, 560) / 36) * 19));
    });
    if (w.OBJ.note) hs.note = Math.max(80, naturalHeight(w.OBJ.note, P.noteW, 104));
    return hs;
  }

  function columns() {
    const P = w.L;
    const u1 = P.c1;
    const u2 = P.c1 + P.stW + P.c2;
    const u3 = u2 + P.paW + P.c3;
    return { u1, u2, u3, width: u3 + P.cadW };
  }

  function groupHeight(g, hs) {
    const P = w.L;
    const H1 = g.stOrder.reduce((a, i) => a + hs[stId(i)], 0) + (g.stOrder.length - 1) * P.stGap;
    const mid = g.midOrder.map((id) => (id === "pe" ? hs[peId(g)] : hs[id]));
    const H2 = mid.reduce((a, b) => a + b, 0) + Math.max(0, mid.length - 1) * P.midGap;
    const H3 = w.OBJ[cadId(g)] ? hs[cadId(g)] : 0;
    return { H1, H2, H3, mid, Hg: Math.max(H1, H2, H3, 90) };
  }

  function layoutGroup(g, gy, side, edge, hs) {
    const P = w.L;
    const { u1, u2, u3 } = columns();
    const X = (u, wd) => (side === "R" ? edge + u : edge - u - wd);
    const gh = groupHeight(g, hs);
    let y = gy + (gh.Hg - gh.H1) / 2;
    g.stOrder.forEach((i) => {
      setBox(w.OBJ[stId(i)], X(u1, P.stW), y, P.stW, hs[stId(i)], tilt(i + 1) * 0.6);
      y += hs[stId(i)] + P.stGap;
    });
    y = gy + (gh.Hg - gh.H2) / 2;
    let peCy = gy + gh.Hg / 2;
    g.midOrder.forEach((id, j) => {
      if (id === "pe") {
        setBox(w.OBJ[peId(g)], X(u2 + (P.paW - P.peW) / 2, P.peW), y, P.peW, gh.mid[j], tilt(g.gi + 20) * 0.7);
        peCy = y + gh.mid[j] / 2;
      } else {
        setBox(w.OBJ[id], X(u2, P.paW), y, P.paW, gh.mid[j], tilt(j + g.gi * 3 + 40) * 0.35);
      }
      y += gh.mid[j] + P.midGap;
    });
    const cad = w.OBJ[cadId(g)];
    if (cad) {
      const y3 = clamp(peCy - gh.H3 / 2, gy, gy + gh.Hg - gh.H3);
      setBox(cad, X(u3, P.cadW), y3, P.cadW, gh.H3, tilt(g.gi + 60) * 0.5);
    }
    g.side = side;
    g.y = gy;
    g.h = gh.Hg;
    return gh.Hg;
  }

  function layoutNote(gy, side, edge, hs) {
    const P = w.L;
    const o = w.OBJ.note;
    if (!o) return 0;
    const { u2 } = columns();
    const x = side === "R" ? edge + u2 + (P.paW - P.noteW) / 2 : edge - u2 - P.paW + (P.paW - P.noteW) / 2;
    setBox(o, x, gy, P.noteW, hs.note, -1);
    w.OTH = { side };
    return hs.note;
  }

  function spotOffset(blockEl, i, bh, s) {
    const sp = Array.from(blockEl.querySelectorAll("[data-s]")).filter((x) => (" " + x.dataset.s + " ").indexOf(" " + i + " ") >= 0);
    const u = sp.filter((x) => x.classList.contains("wl-u"));
    const el = (u.length ? u : sp)[0];
    if (el && measurable() && el.offsetHeight) return el.offsetTop + el.offsetHeight / 2;
    const p = article.body[s.spot.pi] || "";
    const first = s.spot.ranges.length ? s.spot.ranges[0].s : 0;
    return clamp((first / Math.max(1, p.length)) * bh, 10, bh - 8);
  }

  function layoutArticle(ax, ay) {
    const P = w.L;
    const o = w.OBJ.art;
    const artW = P.artColW + 2 * P.artPad;
    o.el.style.width = artW + "px";
    const mast = o.el.querySelector(".wl-art-mast");
    const mastH = measurable() ? mast.offsetHeight : 120;
    let cursor = 20 + mastH + 16;
    const blocks = Array.from(o.el.querySelectorAll(".wl-art-p"));
    w.ART.blocks = {};
    blocks.forEach((b) => {
      const pi = b.dataset.pi;
      b.style.left = P.artPad + "px";
      b.style.width = P.artColW + "px";
      const bh = measurable() ? b.offsetHeight : Math.ceil((pi === "cred" ? 40 : article.body[Number(pi)].length) / 44) * 19;
      let target = cursor;
      if (pi !== "cred") {
        const n = Number(pi);
        let best = Infinity;
        S.forEach((s) => {
          if (s.spot.pi !== n) return;
          const so = w.OBJ[stId(s.i)];
          const t = so.y + Math.min(so.h * 0.5, 110) - ay - spotOffset(b, s.i, bh, s);
          if (t < best) best = t;
        });
        if (isFinite(best)) target = Math.max(cursor, best);
      }
      b.style.top = target.toFixed(1) + "px";
      w.ART.blocks[pi] = { el: b, x: P.artPad, y: target, w: P.artColW, h: bh };
      cursor = target + bh + (pi === "cred" ? 0 : 12);
    });
    const artH = cursor + 26;
    setBox(o, ax, ay, artW, artH, 0);
    w.ART.pins = {};
    S.forEach((s) => {
      const bl = w.ART.blocks[String(s.spot.pi)];
      if (!bl) return;
      w.ART.pins[s.i] = { y: ay + bl.y + spotOffset(bl.el, s.i, bl.h, s), blk: bl };
    });
    return artH;
  }

  function measureCadAnchors() {
    w.CAD_Y = {};
    PAS.forEach((pa) => {
      const g = pa.group;
      const o = g && w.OBJ[cadId(g)];
      if (!o) return;
      const d = g.dashOf(pa.start);
      const el = d >= 0 ? o.el.querySelector('.wl-cad-d i[data-d="' + d + '"]') : null;
      pa.dash = d;
      if (el) el.classList.add("is-ev");
      w.CAD_Y[pa.k] = el && measurable() ? el.offsetTop + el.offsetHeight / 2 : o.h * 0.6;
    });
  }

  function buildLayout(kind) {
    const P = LP[kind];
    w.L = P;
    w.TH = P.th;
    w.els.world.dataset.layout = kind;
    w.els.wrap.dataset.layout = kind;
    const sheet = w.OBJ.sheet;
    sheet.el.style.width = P.sheetW + "px";
    const hs = measureAll();
    const WW = columns().width;
    const gs = GROUPS.slice().sort((a, b) => a.order - b.order);
    const left = [];
    const right = [];
    let hl = 0;
    let hr = 0;
    gs.forEach((g, j) => {
      const h = groupHeight(g, hs).Hg + P.gGap;
      if (P.wings === 1 || j === 0 || hr <= hl) {
        right.push(g);
        hr += h;
      } else {
        left.push(g);
        hl += h;
      }
    });
    const artW = P.artColW + 2 * P.artPad;
    const artX = P.side + (left.length ? WW : 0);
    const artR = artX + artW;
    let yL = P.top;
    let yR = P.top;
    left.forEach((g) => {
      yL += layoutGroup(g, yL, "L", artX, hs) + P.gGap;
    });
    right.forEach((g) => {
      yR += layoutGroup(g, yR, "R", artR, hs) + P.gGap;
    });
    w.OTH = null;
    if (w.OBJ.note) {
      if (left.length && yL <= yR) yL += layoutNote(yL, "L", artX, hs) + P.gGap;
      else yR += layoutNote(yR, "R", artR, hs) + P.gGap;
    }
    const artH = layoutArticle(artX, P.top - 20);
    w.ART.x = artX;
    w.ART.r = artR;
    S.forEach((s) => {
      const g = M.groupOfSt[s.i];
      if (w.ART.pins[s.i]) w.ART.pins[s.i].x = g.side === "L" ? artX - 1 : artR + 1;
    });
    w.WORLD.W = artR + WW + P.side;
    w.WORLD.H = Math.max(yL - P.gGap, yR - P.gGap, P.top - 20 + artH) + P.bottom;
    const cork = w.els.cork;
    cork.style.left = "-14px";
    cork.style.top = "-14px";
    cork.style.width = w.WORLD.W + 28 + "px";
    cork.style.height = w.WORLD.H + 28 + "px";
    [w.els.svLo, w.els.svHi].forEach((sv) => {
      sv.setAttribute("width", w.WORLD.W);
      sv.setAttribute("height", w.WORLD.H);
      sv.setAttribute("viewBox", "0 0 " + w.WORLD.W + " " + w.WORLD.H);
    });
    const gl = [];
    const heads = (side, edge) => {
      const { u1, u2, u3 } = columns();
      [
        [u1, P.stW, "Afirmações da matéria"],
        [u2, P.paW, "Quem disse e trechos"],
        [u3, P.cadW, "Falas"],
      ].forEach((c) => {
        const x = side === "R" ? edge + c[0] + c[1] / 2 : edge - c[0] - c[1] / 2;
        gl.push('<p class="wl-glab" style="left:' + x.toFixed(0) + "px;top:" + (P.top - 44) + 'px;transform:translateX(-50%)">' + esc(c[2]) + "</p>");
      });
    };
    if (left.length) heads("L", artX);
    heads("R", artR);
    w.els.glabs.innerHTML = gl.join("");
    measureCadAnchors();
    w.buildAnchors();
  }

  Object.assign(w, { measurable, setBox, buildLayout, columns });
}
