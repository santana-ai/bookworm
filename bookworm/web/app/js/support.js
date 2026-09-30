import { bandList, bandOf, bandRange } from "./copy.js";
import { clamp, esc, fmtCut, fmtScore } from "./text.js";

const ICON_PATH = {
  weak: '<circle cx="8" cy="8" r="5.6" fill="none" stroke="currentColor" stroke-width="1.8"/>',
  uncertain: '<circle cx="8" cy="8" r="5.6" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M8 2.4a5.6 5.6 0 0 1 0 11.2z" fill="currentColor"/>',
  strong: '<circle cx="8" cy="8" r="6.5" fill="currentColor"/>',
};

export function supportIcon(b) {
  return '<svg class="ap-ic" viewBox="0 0 16 16" aria-hidden="true" focusable="false">' + ICON_PATH[b.k] + "</svg>";
}

export function supportText(p, C) {
  const near = C.high != null && Math.abs(p - C.high) < Math.abs(p - C.low) ? C.high : C.low;
  return fmtScore(p, near);
}

export function supportOf(u, C) {
  const sg = u.signals;
  if (!C || !sg || !sg.scored || !sg.verifier) return null;
  return supportFrom(sg.verifier.probability, C);
}

export function supportFrom(p, C) {
  const b = bandOf(p, C);
  return { p, b, text: supportText(p, C), C };
}

export function supportChip(sup, withValue) {
  return '<span class="ap" data-b="' + sup.b.k + '">' + supportIcon(sup.b) + "<span>" + esc(sup.b.label) + "</span>" + (withValue ? "<b>" + esc(sup.text) + "</b>" : "") + "</span>";
}

function pct(x) {
  return (clamp(x, 0, 1) * 100).toFixed(1) + "%";
}

export function supportRuler(sup, label) {
  const C = sup.C;
  const bands = bandList(C);
  const edges = [0, C.low].concat(C.high != null ? [C.high] : []).concat([1]);
  const zones = bands.map((b, i) => '<span class="ap-zone" data-b="' + b.k + '" style="left:' + pct(edges[i]) + ";width:" + pct(edges[i + 1] - edges[i]) + '"></span>').join("");
  const cuts = edges.slice(1, -1);
  return (
    '<div class="ap-rul" role="img" aria-label="' + esc(label) + '"><div class="ap-rul-bar">' + zones +
    cuts.map((c) => '<span class="ap-rul-cut" style="left:' + pct(c) + '"></span>').join("") +
    '<span class="ap-rul-mk" style="left:' + pct(sup.p) + '"></span></div>' +
    '<div class="ap-rul-ax" aria-hidden="true"><span style="left:0">0</span>' + cuts.map((c) => '<span style="left:' + pct(c) + '">' + fmtCut(c) + "</span>").join("") + '<span style="left:100%">1</span></div>' +
    '<div class="ap-rul-names" aria-hidden="true">' + bands.map((b, i) => '<span data-b="' + b.k + '" style="left:' + pct((edges[i] + edges[i + 1]) / 2) + '">' + esc(b.k === "uncertain" ? "incerto" : b.k === "weak" ? "fraco" : "forte") + "</span>").join("") + "</div></div>"
  );
}

export function supportAria(sup) {
  return "apoio " + sup.text + ", " + sup.b.label + ", faixa " + bandRange(sup.b, sup.C);
}
