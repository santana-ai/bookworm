import { bucketCounts, bucketSentence, statementsSentence } from "./copy.js";
import { esc, fmtDate, foldText, reEsc } from "./text.js";

const NAMES_SHOWN = 4;

export function dotsHtml(tiers) {
  return bucketCounts(tiers)
    .map((b) => ('<i class="dot" data-k="' + b.key + '"></i>').repeat(b.n))
    .join("");
}

function highlight(text, words) {
  const folded = foldText(text);
  if (!words.length || folded.length !== text.length) return esc(text);
  const marks = [];
  words.forEach((wd) => {
    const re = new RegExp(reEsc(wd), "g");
    let m;
    while ((m = re.exec(folded))) {
      marks.push([m.index, m.index + m[0].length]);
      if (!m[0].length) re.lastIndex++;
    }
  });
  if (!marks.length) return esc(text);
  marks.sort((a, b) => a[0] - b[0]);
  let out = "";
  let last = 0;
  marks.forEach(([s, e]) => {
    if (s < last) return;
    out += esc(text.slice(last, s)) + "<mark>" + esc(text.slice(s, e)) + "</mark>";
    last = e;
  });
  return out + esc(text.slice(last));
}

function sortByDate(rows, dir) {
  const sign = dir === "asc" ? 1 : -1;
  return rows.slice().sort((a, b) => {
    const da = a.article_date || "";
    const db = b.article_date || "";
    if (!da && db) return 1;
    if (da && !db) return -1;
    if (da !== db) return da < db ? -sign : sign;
    return (a.id - b.id) * sign;
  });
}

export function createHome(els, index, onPick) {
  const rows = index.hearings.map((h) => ({ h, actors: h.actors, hay: foldText([h.title, h.assunto, h.actors.join(" ")].join(" ")) }));
  const byId = new Map(rows.map((r) => [r.h.id, r]));
  const state = { q: "", sort: "desc", timer: 0 };

  function queryWords() {
    return foldText(state.q)
      .split(/[^\p{L}\p{N}]+/u)
      .filter((x) => x.length > 0);
  }

  function visibleRows() {
    const words = queryWords();
    const hits = rows.filter((r) => words.every((wd) => r.hay.indexOf(wd) >= 0));
    return sortByDate(hits.map((r) => r.h), state.sort).map((h) => byId.get(h.id));
  }

  function namesLine(r, words) {
    if (!r.actors.length) return "";
    const matched = words.length ? r.actors.filter((a) => words.some((wd) => foldText(a).indexOf(wd) >= 0)) : [];
    const rest = r.actors.filter((a) => matched.indexOf(a) < 0);
    const shown = matched.concat(rest).slice(0, Math.max(NAMES_SHOWN, matched.length));
    const more = r.actors.length - shown.length;
    return "Com " + shown.map((a) => highlight(a, words)).join(", ") + (more > 0 ? " e mais " + more : "");
  }

  function cardHtml(r, words) {
    const h = r.h;
    const date = fmtDate(h.article_date);
    return (
      '<li class="clip"><a class="clip-a" href="#h' + h.id + '">' +
      '<span class="clip-k"><span>' + (date ? "Matéria de " + esc(date) : "Matéria sem data") + "</span><span>Audiência " + esc(h.id) + "</span></span>" +
      '<span class="clip-t">' + highlight(h.title, words) + "</span>" +
      '<span class="clip-n">' + esc(statementsSentence(h.n_udvs, h.n_people)) + "</span>" +
      '<span class="dots" aria-hidden="true">' + dotsHtml(h.tiers) + "</span>" +
      '<span class="clip-s">' + esc(bucketSentence(h.tiers)) + "</span>" +
      (r.actors.length ? '<span class="clip-p">' + namesLine(r, words) + "</span>" : "") +
      "</a></li>"
    );
  }

  function render() {
    const words = queryWords();
    const vis = visibleRows();
    els.list.innerHTML = vis.map((r) => cardHtml(r, words)).join("");
    els.empty.hidden = vis.length > 0;
    els.count.textContent =
      vis.length === rows.length
        ? rows.length + " matérias, " + (state.sort === "desc" ? "das mais recentes para as mais antigas." : "das mais antigas para as mais recentes.")
        : "Mostrando " + vis.length + " de " + rows.length + " matérias.";
    els.lucky.disabled = vis.length === 0;
  }

  function onInput() {
    clearTimeout(state.timer);
    state.timer = setTimeout(() => {
      state.q = els.q.value;
      render();
    }, 120);
  }

  function onSort() {
    state.sort = els.sort.value === "asc" ? "asc" : "desc";
    render();
  }

  function onLucky() {
    const vis = visibleRows();
    if (!vis.length) return;
    const r = vis[Math.floor(Math.random() * vis.length)];
    onPick(r.h.id);
  }

  els.q.addEventListener("input", onInput);
  els.sort.addEventListener("change", onSort);
  els.lucky.addEventListener("click", onLucky);
  els.q.value = state.q;
  render();

  return {
    ordered: () => sortByDate(index.hearings, "asc"),
  };
}
