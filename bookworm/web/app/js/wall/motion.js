export function installMotion(w) {
  const live = {};
  const liveObj = {};
  let raf = 0;
  let lastTs = 0;

  function startLoop() {
    if (raf || w.destroyed) return;
    if (w.reduced()) {
      Object.keys(live).forEach((id) => {
        const s = live[id];
        s.wob = 0;
        s.m = s.mT;
        w.drawStr(s);
        delete live[id];
      });
      Object.keys(liveObj).forEach((id) => {
        const o = liveObj[id];
        o.lift = o.liftT;
        redrawObj(o);
        delete liveObj[id];
      });
      return;
    }
    lastTs = 0;
    raf = requestAnimationFrame(tick);
  }

  function tick(ts) {
    raf = 0;
    if (w.destroyed) return;
    const dt = lastTs ? Math.min(0.05, (ts - lastTs) / 1000) : 0.016;
    lastTs = ts;
    Object.keys(liveObj).forEach((id) => {
      const o = liveObj[id];
      o.lift += (o.liftT - o.lift) * (1 - Math.exp(-dt / 0.09));
      if (Math.abs(o.lift - o.liftT) < 0.01) {
        o.lift = o.liftT;
        delete liveObj[id];
      }
      (w.STR_OF[o.id] || []).forEach((s) => {
        if (s.on) live[s.id] = s;
      });
    });
    w.setBatch(true);
    Object.keys(live).forEach((id) => {
      const s = live[id];
      s.ph += dt * 5.4;
      s.wob *= Math.exp(-dt / 0.8);
      s.m += (s.mT - s.m) * (1 - Math.exp(-dt / 0.16));
      if (s.wob < 0.06 && Math.abs(s.m - s.mT) < 0.004 && !liveObj[s.a] && !liveObj[s.b]) {
        s.wob = 0;
        s.m = s.mT;
        delete live[id];
      }
      w.drawStr(s);
    });
    w.setBatch(false);
    w.declutter();
    if (Object.keys(live).length || Object.keys(liveObj).length) raf = requestAnimationFrame(tick);
  }

  function kick(s, amp) {
    if (!s) return;
    if (w.reduced()) {
      w.drawStr(s);
      return;
    }
    s.wob = Math.max(s.wob, amp == null ? 5 : amp);
    live[s.id] = s;
    startLoop();
  }

  function redrawObj(o) {
    (w.STR_OF[o.id] || []).forEach(w.drawStr);
  }

  function setLift(o, v) {
    if (o.liftT === v && o.lift === v) return;
    o.liftT = v;
    if (o.lift == null) o.lift = 0;
    liveObj[o.id] = o;
    startLoop();
  }

  function markLive(s) {
    live[s.id] = s;
  }

  function stopLoop() {
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
  }

  Object.assign(w, { startLoop, kick, setLift, markLive, stopLoop, redrawObj });
}
