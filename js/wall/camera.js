import { clamp } from "../text.js";

const DRAG_THRESHOLD = 6;

export function installCamera(w) {
  w.cam = { x: 0, y: 0, k: 0.5 };
  w.SW = 0;
  w.SH = 0;
  w.camTween = null;
  w.overviewPartial = false;
  let panRaf = 0;

  function applyCam() {
    const cam = w.cam;
    w.els.world.style.transform = "translate(" + cam.x.toFixed(2) + "px," + cam.y.toFixed(2) + "px) scale(" + cam.k.toFixed(4) + ")";
    w.els.world.style.setProperty("--inv", (1 / cam.k).toFixed(4));
    w.els.world.style.setProperty("--sw", Math.max(Math.pow(cam.k, -0.62), 0.85 / cam.k).toFixed(3));
    w.els.world.classList.toggle("n-on", cam.k >= w.TH - 0.004);
    w.els.world.classList.toggle("z-lo", cam.k < 0.6);
    w.els.world.classList.toggle("z-tiny", cam.k < 0.34);
    bigPulled();
    w.declutter();
    updateZoomButtons();
  }

  function bigPulled() {
    w.OBJ_LIST.forEach((o) => {
      const big = !!o.pulled && w.cam.k < w.TH - 0.004 && w.cam.k * (o.ls || 1) >= w.TH - 0.004;
      if (o.big !== big) {
        o.big = big;
        o.el.classList.toggle("is-big", big);
      }
    });
  }

  function clampCam(t, extra) {
    if (!t) return t;
    const x0 = extra ? Math.min(-14, extra.x) : -14;
    const y0 = extra ? Math.min(-14, extra.y) : -14;
    const x1 = extra ? Math.max(w.WORLD.W + 14, extra.x + extra.w) : w.WORLD.W + 14;
    const y1 = extra ? Math.max(w.WORLD.H + 14, extra.y + extra.h) : w.WORLD.H + 14;
    if ((x1 - x0) * t.k >= w.SW) t.x = clamp(t.x, w.SW - x1 * t.k, -x0 * t.k);
    else t.x = clamp(t.x, -x0 * t.k, w.SW - x1 * t.k);
    if ((y1 - y0) * t.k >= w.SH) t.y = clamp(t.y, w.SH - y1 * t.k, -y0 * t.k);
    else t.y = clamp(t.y, -y0 * t.k, w.SH - y1 * t.k);
    return t;
  }

  function boxOf(id) {
    if (id === "all") return { x: -8, y: -8, w: w.WORLD.W + 16, h: w.WORLD.H + 16 };
    if (id.indexOf("artp") === 0) {
      const pin = w.ART.pins[Number(id.slice(4))];
      if (!pin) return boxOf("art");
      const bl = pin.blk;
      return { x: w.OBJ.art.x + bl.x - 8, y: w.OBJ.art.y + bl.y - 10, w: bl.w + 16, h: bl.h + 20 };
    }
    if (id.indexOf("top:") === 0 || id.indexOf("end:") === 0) {
      const b = boxOf(id.slice(4));
      const R = withControls(null);
      const hMax = Math.max(80, (w.SH - R.t - R.b - 2 * w.L.pad) / w.TH);
      if (!b || b.h <= hMax) return b;
      return { x: b.x, y: id[0] === "t" ? b.y : b.y + b.h - hMax, w: b.w, h: hMax };
    }
    const o = w.OBJ[id];
    if (!o) return null;
    return { x: o.x - 4, y: o.y - 12, w: o.w + 8, h: o.h + 16 };
  }

  function liftBox(o) {
    const sc = o.ls || 1.045;
    const ox = o.x + o.w / 2;
    const oy = o.y + o.h * 0.4;
    return { x: ox + (o.x - ox) * sc - 4, y: oy + (o.y - oy) * sc - 7 - 12, w: o.w * sc + 8, h: o.h * sc + 16 };
  }

  function unionBox(ids, liftId) {
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    ids.forEach((id) => {
      const b = id === liftId && w.OBJ[id] ? liftBox(w.OBJ[id]) : boxOf(id);
      if (!b) return;
      x0 = Math.min(x0, b.x);
      y0 = Math.min(y0, b.y);
      x1 = Math.max(x1, b.x + b.w);
      y1 = Math.max(y1, b.y + b.h);
    });
    if (!isFinite(x0)) return { x: 0, y: 0, w: w.WORLD.W, h: w.WORLD.H };
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  }

  let ctrl = null;
  function controlsResv() {
    if (ctrl) return ctrl;
    const z = w.els.zoom;
    ctrl = { l: 0, r: 0, t: 0, b: 0 };
    if (!z.offsetWidth) return ctrl;
    const zr = z.getBoundingClientRect();
    const wr = w.els.wrap.getBoundingClientRect();
    if (w.els.wrap.dataset.layout === "tall") ctrl.b = Math.max(0, wr.bottom - zr.top + 4);
    else ctrl.r = Math.max(0, wr.right - zr.left + 4);
    return ctrl;
  }

  function withControls(resv) {
    const C = controlsResv();
    const R = resv || { l: 0, r: 0, t: 0, b: 0 };
    return { l: Math.max(R.l, C.l), r: Math.max(R.r, C.r), t: Math.max(R.t, C.t), b: Math.max(R.b, C.b) };
  }

  function fitBox(b, maxK, resv) {
    const R = withControls(resv);
    const pad = w.L.pad;
    const aw = Math.max(40, w.SW - R.l - R.r - 2 * pad);
    const ah = Math.max(40, w.SH - R.t - R.b - 2 * pad);
    const k = Math.min(aw / Math.max(1, b.w), ah / Math.max(1, b.h), maxK || w.L.maxK);
    const cx = R.l + (w.SW - R.l - R.r) / 2;
    const cy = R.t + (w.SH - R.t - R.b) / 2;
    return clampCam({ k, x: cx - (b.x + b.w / 2) * k, y: cy - (b.y + b.h / 2) * k }, b);
  }

  function fullK() {
    return fitBox(boxOf("all"), null, legendResv()).k;
  }

  function overviewCam() {
    const R = legendResv();
    const t = fitBox(boxOf("all"), null, R);
    w.overviewPartial = t.k < w.L.overK;
    if (!w.overviewPartial) return t;
    const r = withControls(R);
    const pad = w.L.pad;
    const k = Math.max(w.L.overK, Math.min((w.SW - r.l - r.r - 2 * pad) / (w.WORLD.W + 16), w.L.maxK));
    return clampCam({ k, x: r.l + (w.SW - r.l - r.r) / 2 - (w.WORLD.W / 2) * k, y: pad + 30 * k });
  }

  function fits(ids) {
    return fitBox(unionBox(ids)).k >= w.TH - 0.004;
  }

  function frameIds(ids) {
    if (ids.length === 1 && ids[0] === "all") return overviewCam();
    return fitBox(unionBox(ids));
  }

  function legendResv() {
    const lg = w.els.legend;
    if (lg.hidden || !lg.offsetWidth) return null;
    const lr = lg.getBoundingClientRect();
    const wr = w.els.wrap.getBoundingClientRect();
    if (w.els.wrap.dataset.layout === "wide") return { l: Math.max(0, lr.right - wr.left + 6), r: 0, t: 0, b: 0 };
    return { l: 0, r: 0, t: 0, b: Math.max(0, wr.bottom - lr.top + 2) };
  }

  function camView(c) {
    return [(w.SW / 2 - c.x) / c.k, (w.SH / 2 - c.y) / c.k, w.SW / c.k];
  }

  function setCamFromView(v) {
    w.cam.k = w.SW / v[2];
    w.cam.x = w.SW / 2 - v[0] * w.cam.k;
    w.cam.y = w.SH / 2 - v[1] * w.cam.k;
  }

  function zoomPath(p0, p1) {
    try {
      return w.D3 && w.D3.interpolateZoom ? w.D3.interpolateZoom.rho(1.1)(p0, p1) : null;
    } catch (error) {
      return null;
    }
  }

  function setCam(t) {
    w.cam.x = t.x;
    w.cam.y = t.y;
    w.cam.k = t.k;
    applyCam();
  }

  function camTo(t, dur) {
    killCamTween();
    if (w.reduced() || !w.SW) {
      setCam(t);
      return null;
    }
    const p0 = camView(w.cam);
    const p1 = camView(t);
    const iz = zoomPath(p0, p1);
    const d = dur != null ? dur : iz ? clamp((iz.duration / 1000) * 0.62, 0.55, 1.7) : 0.9;
    const o = { t: 0 };
    w.camTween = w.G.to(o, {
      t: 1,
      duration: d,
      ease: "power2.inOut",
      onUpdate: () => {
        if (iz) setCamFromView(iz(o.t));
        else setCamFromView([p0[0] + (p1[0] - p0[0]) * o.t, p0[1] + (p1[1] - p0[1]) * o.t, p0[2] + (p1[2] - p0[2]) * o.t]);
        applyCam();
      },
    });
    return w.camTween;
  }

  function camStep(tl, ids, ref) {
    const t = frameIds(ids);
    const p0 = ref.v;
    const p1 = camView(t);
    const iz = zoomPath(p0, p1);
    const same = Math.abs(p0[0] - p1[0]) < 2 && Math.abs(p0[1] - p1[1]) < 2 && Math.abs(p0[2] - p1[2]) / p1[2] < 0.01;
    const d = same ? 0.01 : iz ? clamp((iz.duration / 1000) * 0.62, 0.5, 1.6) : 0.85;
    const o = { t: 0 };
    tl.to(o, {
      t: 1,
      duration: d,
      ease: "power2.inOut",
      onUpdate: () => {
        if (iz) setCamFromView(iz(o.t));
        else setCamFromView([p0[0] + (p1[0] - p0[0]) * o.t, p0[1] + (p1[1] - p0[1]) * o.t, p0[2] + (p1[2] - p0[2]) * o.t]);
        applyCam();
      },
    });
    ref.v = p1;
  }

  function killCamTween() {
    if (w.camTween) {
      w.camTween.kill();
      w.camTween = null;
    }
  }

  let kLo = null;
  function kBounds() {
    if (kLo == null) kLo = Math.min(fullK(), w.L.overK) * 0.9;
    return { lo: kLo, hi: Math.max(w.L.maxK * 1.5, 2) };
  }
  function resetZoomBounds() {
    kLo = null;
    ctrl = null;
  }

  function zoomAt(f, cx, cy) {
    const b = kBounds();
    const k1 = clamp(w.cam.k * f, b.lo, b.hi);
    const wx = (cx - w.cam.x) / w.cam.k;
    const wy = (cy - w.cam.y) / w.cam.k;
    w.cam.k = k1;
    w.cam.x = cx - wx * k1;
    w.cam.y = cy - wy * k1;
    clampCam(w.cam);
    applyCam();
  }

  function zoomBy(f) {
    userMoved();
    if (w.reduced()) {
      zoomAt(f, w.SW / 2, w.SH / 2);
      return;
    }
    const b = kBounds();
    const k1 = clamp(w.cam.k * f, b.lo, b.hi);
    const cx = w.SW / 2;
    const cy = w.SH / 2;
    const wx = (cx - w.cam.x) / w.cam.k;
    const wy = (cy - w.cam.y) / w.cam.k;
    camTo(clampCam({ k: k1, x: cx - wx * k1, y: cy - wy * k1 }), 0.35);
  }

  function updateZoomButtons() {
    if (!w.els.zin || !w.L) return;
    const b = kBounds();
    w.els.zin.disabled = w.cam.k >= b.hi - 1e-3;
    w.els.zout.disabled = w.cam.k <= b.lo + 1e-3;
  }

  function schedulePanDraw() {
    if (panRaf) return;
    panRaf = requestAnimationFrame(() => {
      panRaf = 0;
      applyCam();
    });
  }

  function userMoved() {
    if (w.busy && w.busy()) w.finishAll();
    killCamTween();
    hideHint();
  }

  let hintTimer = 0;
  function showHint(text) {
    const h = w.els.hint;
    h.textContent = text;
    h.classList.remove("is-off");
    clearTimeout(hintTimer);
    hintTimer = setTimeout(hideHint, 6000);
  }
  function hideHint() {
    clearTimeout(hintTimer);
    w.els.hint.classList.add("is-off");
  }

  const pointers = new Map();
  let drag = null;
  let suppressClick = false;

  function wrapPoint(e) {
    const r = w.els.wrap.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  function ignoreTarget(t) {
    return !!(t && t.closest && t.closest(".wl-zoom, .wl-legend, .wl-reader, .wl-read"));
  }

  function onPointerDown(e) {
    if (!w.L || !w.els.reader.hidden || ignoreTarget(e.target)) return;
    if (e.pointerType === "mouse" && e.button !== 0) return;
    if (e.pointerType !== "mouse" && w.st.mode !== "net") return;
    pointers.set(e.pointerId, wrapPoint(e));
    if (pointers.size === 1) drag = { id: e.pointerId, start: wrapPoint(e), last: wrapPoint(e), moved: false, pinch: null };
    else if (pointers.size === 2 && drag) {
      const pts = Array.from(pointers.values());
      drag.pinch = { d: Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y) || 1 };
      drag.moved = true;
      startPan();
    }
  }

  function startPan() {
    if (drag.capturing) return;
    drag.capturing = true;
    userMoved();
    w.els.wrap.classList.add("is-panning");
    pointers.forEach((p, id) => {
      try {
        w.els.wrap.setPointerCapture(id);
      } catch (error) {
        return;
      }
    });
  }

  function onPointerMove(e) {
    if (!drag || !pointers.has(e.pointerId)) return;
    const p = wrapPoint(e);
    pointers.set(e.pointerId, p);
    if (pointers.size >= 2 && drag.pinch) {
      const pts = Array.from(pointers.values());
      const d = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y) || 1;
      const mid = { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2 };
      zoomAt(d / drag.pinch.d, mid.x, mid.y);
      drag.pinch.d = d;
      e.preventDefault();
      return;
    }
    if (e.pointerId !== drag.id) return;
    if (!drag.moved && Math.hypot(p.x - drag.start.x, p.y - drag.start.y) < DRAG_THRESHOLD) return;
    if (!drag.moved) {
      drag.moved = true;
      startPan();
    }
    w.cam.x += p.x - drag.last.x;
    w.cam.y += p.y - drag.last.y;
    drag.last = p;
    clampCam(w.cam);
    schedulePanDraw();
    e.preventDefault();
  }

  function onPointerUp(e) {
    if (!pointers.has(e.pointerId)) return;
    pointers.delete(e.pointerId);
    if (!drag) return;
    if (pointers.size === 0) {
      if (drag.moved) suppressClick = true;
      drag = null;
      w.els.wrap.classList.remove("is-panning");
    } else if (drag.pinch) {
      const rest = Array.from(pointers.entries())[0];
      drag = { id: rest[0], start: rest[1], last: rest[1], moved: true, pinch: null, capturing: true };
    }
  }

  function onClickCapture(e) {
    if (!suppressClick) return;
    suppressClick = false;
    e.stopPropagation();
    e.preventDefault();
  }

  function onWheel(e) {
    if (!w.L || !(e.ctrlKey || e.metaKey) || !w.els.reader.hidden) return;
    e.preventDefault();
    userMoved();
    const p = wrapPoint(e);
    zoomAt(Math.exp(-e.deltaY * 0.0025), p.x, p.y);
  }

  function onZoomClick(e) {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.classList.contains("wl-zin")) zoomBy(1.35);
    else if (b.classList.contains("wl-zout")) zoomBy(1 / 1.35);
    else if (b.classList.contains("wl-zfit")) {
      userMoved();
      camTo(frameIds(["all"]));
    }
  }

  const wrap = w.els.wrap;
  wrap.addEventListener("pointerdown", onPointerDown);
  wrap.addEventListener("pointermove", onPointerMove);
  wrap.addEventListener("pointerup", onPointerUp);
  wrap.addEventListener("pointercancel", onPointerUp);
  wrap.addEventListener("click", onClickCapture, true);
  wrap.addEventListener("wheel", onWheel, { passive: false });
  w.els.zoom.addEventListener("click", onZoomClick);

  function destroyCamera() {
    killCamTween();
    cancelAnimationFrame(panRaf);
    clearTimeout(hintTimer);
  }

  Object.assign(w, {
    applyCam, clampCam, boxOf, unionBox, fitBox, fits, frameIds, legendResv, camView, setCamFromView, camTo, camStep,
    killCamTween, zoomBy, setCam, showHint, hideHint, destroyCamera, overviewCam, resetZoomBounds,
  });
}
