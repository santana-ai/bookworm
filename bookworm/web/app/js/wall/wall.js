import { caseHash } from "../case.js";
import { bindDisclose } from "../disclose.js";
import { buildModel } from "../model.js";
import { clamp, esc, shorten } from "../text.js";
import { installAsk } from "./ask.js";
import { installCamera } from "./camera.js";
import { installLayout } from "./layout.js";
import { installLegend } from "./legend.js";
import { wallMarkup } from "./markup.js";
import { installMotion } from "./motion.js";
import { installObjects } from "./objects.js";
import { installScene } from "./scene.js";
import { installStory } from "./story.js";
import { installStrings } from "./strings.js";

function collectElements(root) {
  const q = (sel) => root.querySelector(sel);
  return {
    wrap: q(".wl-wrap"), world: q(".wl-world"), cork: q(".wl-cork"), objs: q(".wl-objs"), glabs: q(".wl-glabs"),
    svLo: q(".wl-sv-lo"), svHi: q(".wl-sv-hi"), tags: q(".wl-tags"),
    reader: q(".wl-reader"), readerBody: q(".wl-reader-body"), readerPos: q(".wl-reader-pos"), readerH: q(".wl-reader-h"),
    rdBefore: q(".wl-rd-before"), rdAfter: q(".wl-rd-after"), rdClose: q(".wl-rd-close"), rdMore: q(".wl-rd-more"), list: q(".wl-list-l"),
    next: q(".wl-next"), back: q(".wl-back"), auto: q(".wl-auto"), kase: q(".wl-case"), all: q(".wl-all"), pick: q(".wl-pick-sel"),
    capScene: q(".wl-cap-scene"), capText: q(".wl-cap-text"), cap: q(".wl-cap"), legend: q(".wl-legend"), key: q(".wl-key"), disc: q(".wl-disc"),
    zoom: q(".wl-zoom"), zin: q(".wl-zin"), zout: q(".wl-zout"), hint: q(".wl-hint"),
    q: q(".wl-q"), askform: q(".wl-askform"), askAud: q(".wl-ask-aud"), askSug: q(".wl-ask-sug"), aterms: q(".wl-ask-terms"),
    astrip: q(".wl-astrip"), astripRow: q(".wl-astrip-row"), astripCap: q(".wl-astrip-cap"), apages: q(".wl-apages"),
  };
}

export function createWall(host, H) {
  const M = buildModel(H);
  const root = document.createElement("section");
  root.className = "wl";
  root.setAttribute("aria-label", "Rede de barbantes da audiência " + H.hearing.id);
  root.innerHTML = wallMarkup();
  host.innerHTML = "";
  host.appendChild(root);
  const mq = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
  const w = { M, root, G: window.gsap || null, D3: window.d3 || null, destroyed: false };
  w.reduced = () => !w.G || !!(mq && mq.matches);
  w.els = collectElements(root);

  installObjects(w);
  w.els.art = w.OBJ.art.el;
  installLayout(w);
  installStrings(w);
  installCamera(w);
  installMotion(w);
  installScene(w);
  installLegend(w);
  installStory(w);
  installAsk(w);

  const els = w.els;
  els.pick.innerHTML =
    '<option value="" disabled>Escolha uma afirmação</option>' +
    M.S.map((s, j) => '<option value="' + j + '">' + (j + 1) + ". " + esc(s.u.actor.name) + ": " + esc(shorten(s.u.proposition.replace(/^["“]/, ""), 58)) + "</option>").join("");
  els.disc.textContent = w.disclaimer;
  els.list.innerHTML = M.S.map(
    (s, j) =>
      '<li><a href="' + caseHash(H.hearing.id, j + 1) + '"><span class="wl-list-n">' + (j + 1) + '</span><span class="wl-list-b"><span class="wl-list-w">' + esc(s.u.actor.name) + '</span><span class="wl-list-p">' + esc(shorten(s.u.proposition, 170)) + '</span></span><span class="wl-list-t" data-t="' + s.tier.k + '">' + esc(s.tier.short) + "</span></a></li>",
  ).join("");

  function stageSize() {
    const vh = window.innerHeight || 800;
    const top = els.wrap.getBoundingClientRect().top - root.getBoundingClientRect().top;
    const capH = els.cap.offsetHeight || 120;
    const cw = els.wrap.clientWidth;
    let h = vh - top - capH - 10 - 14;
    const kind = cw >= 760 && cw / Math.max(300, h) >= 1.2 ? "wide" : "tall";
    h = clamp(h, 320, kind === "wide" ? 820 : 720);
    return { w: cw, h: Math.round(h), kind };
  }

  function setTH() {
    w.TH = w.L.name === "tall" ? clamp((w.SW - 2 * w.L.pad) / (w.L.stW + 8), 0.95, w.L.th) : w.L.th;
  }

  function relayout(force) {
    if (w.destroyed || !w.measurable()) return;
    const z = stageSize();
    if (!z.w) return;
    const changed = force || !w.L || w.L.name !== z.kind;
    els.wrap.style.height = z.h + "px";
    const resized = Math.abs(w.SW - z.w) > 0.5 || Math.abs(w.SH - z.h) > 0.5;
    w.SW = z.w;
    w.SH = z.h;
    w.resetZoomBounds();
    if (changed) {
      w.finishAll();
      w.buildLayout(z.kind);
      w.renderLegend();
      setTH();
      w.autoLegend();
      w.reapply();
    } else if (resized) {
      setTH();
      w.autoLegend();
      w.finishAll();
      w.refit();
    }
  }

  function storyOnScreen() {
    const r = els.wrap.getBoundingClientRect();
    const vh = window.innerHeight || 0;
    return r.bottom > 40 && r.top < vh - 40;
  }

  function onKey(e) {
    if (w.destroyed || e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
    const key = e.key;
    const zoomKey = key === "+" || key === "=" || key === "-" || key === "_";
    if (key !== "ArrowRight" && key !== "ArrowLeft" && key !== "Escape" && !zoomKey) return;
    const tg = e.target;
    const inside = !!(tg && tg.nodeType === 1 && root.contains(tg));
    const loose = !tg || tg === document.body || tg === document.documentElement || tg === document;
    if (!inside && !loose) return;
    if (!els.reader.hidden) {
      if (key === "Escape") {
        e.preventDefault();
        w.closeReader(false);
      }
      return;
    }
    if (key === "Escape") {
      if (w.st.mode === "net" && w.st.pull) {
        e.preventDefault();
        w.back();
      }
      return;
    }
    if (inside) {
      const tag = tg.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || tg.isContentEditable) return;
      if (tg.closest && tg.closest(".wl-ask")) return;
    } else if (!storyOnScreen()) return;
    e.preventDefault();
    if (zoomKey) w.zoomBy(key === "+" || key === "=" ? 1.35 : 1 / 1.35);
    else if (key === "ArrowRight") w.next();
    else w.back();
  }

  function onRootClick(e) {
    const t = e.target;
    if (!t || !t.closest) return;
    const rb = t.closest("[data-read]");
    if (rb && root.contains(rb)) {
      w.openReader(Number(rb.dataset.read), rb);
      return;
    }
    if (t.closest(".wl-lg-h")) {
      w.toggleLegend();
      return;
    }
    if (t.closest(".wl-ask")) {
      w.onAskClick(e);
      return;
    }
    if (w.st.mode !== "net" || t.closest(".wl-zoom, .wl-reader, .wl-legend")) return;
    const pb = t.closest("[data-pull]");
    if (pb && root.contains(pb)) {
      w.pullFromUser(pb.dataset.pull);
      return;
    }
    const tg = t.closest(".wl-tag");
    const sg = !tg && t.closest("g.wl-s");
    const sid = tg ? tg.dataset.sid : sg ? sg.getAttribute("data-sid") : null;
    if (sid) {
      w.followFromUser(sid);
      return;
    }
    if (els.wrap.contains(t) && els.reader.hidden && w.st.pull) {
      w.finishAll();
      w.unpull();
    }
  }

  function onWorldFocus(e) {
    const el = e.target;
    if (!el || !el.getBoundingClientRect || !w.SW) return;
    if (els.wrap.scrollLeft || els.wrap.scrollTop) {
      els.wrap.scrollLeft = 0;
      els.wrap.scrollTop = 0;
    }
    const r = el.getBoundingClientRect();
    const wr = els.wrap.getBoundingClientRect();
    const m = 8;
    if (r.left >= wr.left + m && r.right <= wr.right - m && r.top >= wr.top + m && r.bottom <= wr.bottom - m) return;
    if (w.busy()) w.finishAll();
    w.killCamTween();
    w.cam.x += w.SW / 2 - ((r.left + r.right) / 2 - wr.left);
    w.cam.y += w.SH / 2 - ((r.top + r.bottom) / 2 - wr.top);
    w.clampCam(w.cam);
    w.applyCam();
  }

  function onWrapScroll() {
    if (els.wrap.scrollLeft || els.wrap.scrollTop) {
      els.wrap.scrollLeft = 0;
      els.wrap.scrollTop = 0;
    }
  }

  const onNext = () => w.next();
  const onBack = () => w.back();
  const onAll = () => {
    w.closeReader(true);
    if (w.st.mode === "net") w.leaveNetwork();
    else w.goNetwork();
  };
  const onAuto = () => w.setAuto(!w.st.auto);
  const onPick = () => {
    const j = Number(els.pick.value);
    if (els.pick.value === "" || !isFinite(j)) return;
    w.pickStatement(j);
  };
  const onAskSubmit = (e) => {
    e.preventDefault();
    w.runAsk(true);
  };
  const onAskKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      w.runAsk(true);
    }
  };
  let relayoutRaf = 0;
  const onResize = () => {
    if (relayoutRaf) return;
    relayoutRaf = requestAnimationFrame(() => {
      relayoutRaf = 0;
      relayout(false);
    });
  };

  els.next.addEventListener("click", onNext);
  els.back.addEventListener("click", onBack);
  els.all.addEventListener("click", onAll);
  els.auto.addEventListener("click", onAuto);
  els.pick.addEventListener("change", onPick);
  els.rdClose.addEventListener("click", () => w.closeReader(false));
  els.rdMore.addEventListener("click", () => w.toggleReader());
  els.rdBefore.addEventListener("click", () => w.readMore(-1));
  els.rdAfter.addEventListener("click", () => w.readMore(1));
  els.askform.addEventListener("submit", onAskSubmit);
  els.q.addEventListener("keydown", onAskKey);
  root.addEventListener("click", onRootClick);
  els.wrap.addEventListener("scroll", onWrapScroll);
  els.world.addEventListener("focusin", onWorldFocus);
  document.addEventListener("keydown", onKey);
  window.addEventListener("resize", onResize);
  const ro = window.ResizeObserver ? new ResizeObserver(onResize) : null;
  if (ro) ro.observe(root);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => relayout(true));

  bindDisclose(root, "wall");
  w.start();
  w.initAsk();
  relayout(true);

  function destroy() {
    if (w.destroyed) return;
    w.destroyed = true;
    cancelAnimationFrame(relayoutRaf);
    w.stopStory();
    w.destroyCamera();
    w.stopLoop();
    document.removeEventListener("keydown", onKey);
    window.removeEventListener("resize", onResize);
    if (ro) ro.disconnect();
    if (w.G && w.G.killTweensOf) w.G.killTweensOf(root.querySelectorAll(".wl-o-in, .wl-apage, .wl-apin, .wl-cad-d i, .wl-slips li, .wl-qw"));
    root.remove();
  }

  function openStatement(j) {
    if (w.destroyed || !M.S[j]) return;
    w.openStatementAt(j, "result");
  }

  function openPassage(p) {
    if (w.destroyed) return false;
    return w.openReaderPassage(p);
  }

  return { destroy, openStatement, openPassage };
}
