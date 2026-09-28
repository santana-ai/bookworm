const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

export function esc(value) {
  return String(value).replace(/[&<>"']/g, (c) => ESCAPES[c]);
}

export function fmtInt(n) {
  return String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ".");
}

const SCORE_DIGITS = 2;
const SCORE_MAX_DIGITS = 4;

function roundTo(x, digits, how) {
  const f = Math.pow(10, digits);
  return (how || Math.round)(x * f) / f;
}

export function fmtScore(x, cut) {
  let digits = SCORE_DIGITS;
  let shown = roundTo(x, digits);
  if (typeof cut === "number") {
    while (digits < SCORE_MAX_DIGITS && x < cut !== shown < cut) shown = roundTo(x, ++digits);
    if (x < cut !== shown < cut) shown = roundTo(x, digits, x < cut ? Math.floor : Math.ceil);
  }
  return shown.toFixed(digits).replace(".", ",");
}

export function fmtCut(x) {
  let digits = SCORE_DIGITS;
  while (digits < SCORE_MAX_DIGITS && Math.abs(roundTo(x, digits) - x) > 1e-9) digits += 1;
  return roundTo(x, digits).toFixed(digits).replace(".", ",");
}

export function joinPt(items) {
  if (items.length < 2) return items.join("");
  return items.slice(0, -1).join(", ") + " e " + items[items.length - 1];
}

function stripAcc(s) {
  return String(s).normalize("NFD").replace(/[̀-ͯ]/g, "");
}

export function foldText(s) {
  return stripAcc(String(s).toLowerCase());
}

export function reEsc(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function clamp(v, a, b) {
  return Math.max(a, Math.min(b, v));
}

export function tno(i) {
  return Number(i) + 1;
}

export function plural(n, one, many) {
  return n === 1 ? one : many;
}

export function countLabel(n, one, many) {
  return fmtInt(n) + " " + plural(n, one, many);
}

export function wordsOf(text) {
  const out = [];
  const re = /[\p{L}\p{N}]+/gu;
  let m;
  while ((m = re.exec(text))) out.push({ w: foldText(m[0]), s: m.index, e: m.index + m[0].length });
  return out;
}

export function seqIndex(hay, needle) {
  if (!needle.length) return -1;
  outer: for (let i = 0; i + needle.length <= hay.length; i++) {
    for (let k = 0; k < needle.length; k++) if (hay[i + k].w !== needle[k].w) continue outer;
    return i;
  }
  return -1;
}

export function matchBlocks(a, b, minLen) {
  const out = [];
  const rec = (alo, ahi, blo, bhi) => {
    if (alo >= ahi || blo >= bhi) return;
    let best = { i: alo, j: blo, n: 0 };
    let prev = new Array(bhi - blo + 1).fill(0);
    for (let i = alo; i < ahi; i++) {
      const cur = new Array(bhi - blo + 1).fill(0);
      for (let j = blo; j < bhi; j++) {
        if (a[i] !== b[j]) continue;
        const v = prev[j - blo] + 1;
        cur[j - blo + 1] = v;
        if (v > best.n) best = { i: i - v + 1, j: j - v + 1, n: v };
      }
      prev = cur;
    }
    if (best.n < minLen) return;
    rec(alo, best.i, blo, best.j);
    out.push(best);
    rec(best.i + best.n, ahi, best.j + best.n, bhi);
  };
  rec(0, a.length, 0, b.length);
  return out;
}

export function markText(text, ranges) {
  if (!ranges.length) return esc(text);
  let pts = [0, text.length];
  ranges.forEach((r) => pts.push(r.s, r.e));
  pts = pts.filter((v, i, a) => a.indexOf(v) === i && v >= 0 && v <= text.length).sort((a, b) => a - b);
  let out = "";
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i];
    const b = pts[i + 1];
    if (b <= a) continue;
    const seg = esc(text.slice(a, b));
    const cls = [];
    const data = [];
    let tag = "span";
    ranges.forEach((r) => {
      if (r.s <= a && r.e >= b) {
        if (r.cls && cls.indexOf(r.cls) < 0) cls.push(r.cls);
        if (r.tag) tag = r.tag;
        if (r.d != null && data.indexOf(r.d) < 0) data.push(r.d);
      }
    });
    if (!cls.length && tag === "span") {
      out += seg;
      continue;
    }
    out += "<" + tag + (cls.length ? ' class="' + cls.join(" ") + '"' : "") + (data.length ? ' data-s="' + data.join(" ") + '"' : "") + ">" + seg + "</" + tag + ">";
  }
  return out;
}

export function titleCase(s) {
  const small = { de: 1, da: 1, do: 1, dos: 1, das: 1, e: 1 };
  return String(s)
    .toLowerCase()
    .split(/\s+/)
    .map((w, i) => (i > 0 && small[w] ? w : w.charAt(0).toUpperCase() + w.slice(1)))
    .join(" ");
}

export function shorten(s, n) {
  if (s.length <= n) return s;
  const cut = s.slice(0, n);
  const sp = cut.lastIndexOf(" ");
  return (sp > n * 0.6 ? cut.slice(0, sp) : cut).replace(/[\s,;:.]+$/, "") + "…";
}

export function tail(s, n) {
  if (s.length <= n) return s;
  const cut = s.slice(s.length - n);
  const sp = cut.indexOf(" ");
  return "…" + (sp >= 0 && sp < n * 0.4 ? cut.slice(sp + 1) : cut);
}

export function fmtDate(iso) {
  const parts = String(iso || "").split("-");
  return parts.length === 3 ? parts[2] + "/" + parts[1] + "/" + parts[0] : "";
}

const TITLES = new Set([
  "coronel", "deputado", "deputada", "delegado", "delegada", "senador", "senadora", "general", "capitão", "capitã", "sargento", "tenente", "major",
  "pastor", "pastora", "padre", "frei", "irmã", "dom", "dr.", "dra.", "doutor", "doutora", "professor", "professora", "prof.", "profa.",
  "ministro", "ministra", "mestre", "cacique", "pajé", "vereador", "vereadora", "prefeito", "prefeita", "juiz", "juíza", "desembargador", "desembargadora", "comandante",
]);

export function firstName(name) {
  const m = String(name).match(/^(.*?)\s*\(([^)]+)\)\s*$/);
  if (m) return m[2];
  const parts = String(name).split(/\s+/).filter(Boolean);
  if (parts.length > 1 && TITLES.has(parts[0].toLowerCase())) return parts[0] + " " + parts[1];
  return parts[0] || "";
}

export function tilt(seed) {
  let v = Math.sin(seed * 12.9898) * 43758.5453;
  v -= Math.floor(v);
  return (v - 0.5) * 2.6;
}

export function hashToken(value) {
  const m = String(value || "").match(/^#?h(\d+)$/);
  return m ? Number(m[1]) : null;
}

export function actorName(a) {
  const n = (a && (a.display_name || a.name)) || "";
  return n && n === n.toUpperCase() && /\p{Lu}{2}/u.test(n) ? titleCase(n) : n;
}
