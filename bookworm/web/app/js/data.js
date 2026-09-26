import { TIER_ORDER } from "./copy.js";

const cache = new Map();
const CACHE_LIMIT = 4;

const SUPPORT_NAMES = ["direct_quote", "semantic_with_short_quote", "semantic_similarity"];
const WITH_EVIDENCE = ["quote_found", "semantic_match_high", "semantic_match_weak"];

export class DataError extends Error {
  constructor(kind, url, detail) {
    super(kind + ": " + url + (detail ? " (" + detail + ")" : ""));
    this.kind = kind;
    this.url = url;
    this.detail = detail || "";
  }
}

function kind(test, want) {
  test.want = want;
  return test;
}

const INT = kind((v) => Number.isInteger(v), "número inteiro");
const NUM = kind((v) => typeof v === "number" && Number.isFinite(v), "número");
const STR = kind((v) => typeof v === "string", "texto");
const BOOL = kind((v) => typeof v === "boolean", "verdadeiro ou falso");
const nullable = (t) => kind((v) => v === null || t(v), t.want + " ou null");
const oneOf = (values) => kind((v) => values.indexOf(v) >= 0, "um de " + values.join(", "));
const counts = (names) => Object.fromEntries(names.map((n) => [n, INT]));

const RUN = { name: STR, encoder: STR, revision: STR, threshold: NUM };

const INDEX = {
  run: RUN,
  hearings: [
    {
      id: INT,
      split: nullable(STR),
      article_date: nullable(STR),
      assunto: STR,
      title: STR,
      n_udvs: INT,
      n_people: INT,
      n_people_resolved: INT,
      tiers: counts(TIER_ORDER),
      support_types: counts(SUPPORT_NAMES),
      transcript_words: INT,
      actors: [STR],
    },
  ],
};

const EVIDENCE = {
  text: STR,
  support_type: oneOf(SUPPORT_NAMES),
  score: nullable(NUM),
  quote_prefix: nullable(STR),
  start_char: INT,
  end_char: INT,
  speaker_turn: INT,
};

const HEARING = {
  hearing: { id: INT, split: nullable(STR), article_date: nullable(STR), assunto: STR, materia: STR, transcript_chars: INT, transcript_words: INT },
  transcript: STR,
  turns: [{ index: INT, speaker: STR, party: STR, start: INT, end: INT, sentences: [{ text: STR, start: nullable(INT), end: nullable(INT) }] }],
  people: [{ index: INT, name: STR, role: STR, turns: [INT], resolved: BOOL }],
  udvs: [
    {
      id: STR,
      hearing_id: INT,
      actor: { name: STR, role: STR },
      proposition: STR,
      evidence: { nullable: EVIDENCE },
      tier: oneOf(TIER_ORDER),
      candidates: [{ text: STR, score: NUM, turn: INT, start: nullable(INT), end: nullable(INT) }],
      n_candidates: INT,
      quotes: [STR],
    },
  ],
  run: RUN,
};

function walk(url, value, shape, at) {
  if (typeof shape === "function") {
    if (!shape(value)) throw new DataError("format", url, at + ": esperava " + shape.want);
    return;
  }
  if (Array.isArray(shape)) {
    if (!Array.isArray(value)) throw new DataError("format", url, at + ": esperava uma lista");
    value.forEach((item, i) => walk(url, item, shape[0], at + "[" + i + "]"));
    return;
  }
  if (shape.nullable) {
    if (value !== null) walk(url, value, shape.nullable, at);
    return;
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new DataError("format", url, at + ": esperava um objeto");
  Object.keys(shape).forEach((key) => {
    const where = at ? at + "." + key : key;
    if (!(key in value)) throw new DataError("format", url, where + ": campo ausente");
    walk(url, value[key], shape[key], where);
  });
}

function sum(obj) {
  return Object.keys(obj).reduce((a, k) => a + obj[k], 0);
}

function checkIndex(url, data) {
  walk(url, data, INDEX, "");
  data.hearings.forEach((h, i) => {
    const at = "hearings[" + i + "]";
    if (sum(h.tiers) !== h.n_udvs) throw new DataError("format", url, at + ".tiers: a soma não é n_udvs");
    if (h.actors.length !== h.n_people) throw new DataError("format", url, at + ".actors: o tamanho não é n_people");
  });
  return data;
}

function checkHearing(url, data, id) {
  walk(url, data, HEARING, "");
  const T = data.transcript;
  const bad = (at, what) => {
    throw new DataError("format", url, at + ": " + what);
  };
  if (data.hearing.id !== id) bad("hearing.id", "esperava " + id);
  if (T.length !== data.hearing.transcript_chars) bad("hearing.transcript_chars", "difere do tamanho de transcript, então as posições não batem com o texto");
  const turnIds = new Set(data.turns.map((t) => t.index));
  const people = new Map();
  data.people.forEach((p, i) => {
    if (people.has(p.name)) bad("people[" + i + "].name", "nome repetido");
    if (p.resolved !== (p.turns.length > 0)) bad("people[" + i + "].resolved", "não corresponde a turns");
    p.turns.forEach((t) => {
      if (!turnIds.has(t)) bad("people[" + i + "].turns", "turno " + t + " não existe");
    });
    people.set(p.name, p);
  });
  data.udvs.forEach((u, i) => {
    const at = "udvs[" + i + "]";
    const p = people.get(u.actor.name);
    if (!p) bad(at + ".actor.name", "não está em people");
    if ((u.tier === "person_not_resolved") === p.resolved) bad(at + ".tier", "não corresponde a people.resolved");
    const ev = u.evidence;
    if ((ev !== null) !== (WITH_EVIDENCE.indexOf(u.tier) >= 0)) bad(at + ".evidence", "não corresponde ao tier " + u.tier);
    if (!ev) return;
    if ((ev.support_type === "direct_quote") !== (u.tier === "quote_found")) bad(at + ".evidence.support_type", "não corresponde ao tier " + u.tier);
    if (ev.support_type !== "direct_quote" && ev.score === null) bad(at + ".evidence.score", "ausente numa evidência por semelhança");
    if (!(ev.start_char >= 0 && ev.start_char <= ev.end_char && ev.end_char <= T.length)) bad(at + ".evidence", "posições fora da transcrição");
    if (p.turns.indexOf(ev.speaker_turn) < 0) bad(at + ".evidence.speaker_turn", "não é um turno de " + p.name);
  });
  return data;
}

async function getJson(url) {
  let response;
  try {
    response = await fetch(url, { cache: "no-cache" });
  } catch (error) {
    throw new DataError("network", url, error && error.message);
  }
  if (response.status === 404) throw new DataError("missing", url);
  if (!response.ok) throw new DataError("http", url, String(response.status));
  try {
    return await response.json();
  } catch (error) {
    throw new DataError("format", url, "JSON inválido");
  }
}

let indexPromise = null;

export function loadIndex() {
  if (!indexPromise) {
    const url = "data/index.json";
    indexPromise = getJson(url).then((data) => checkIndex(url, data));
    indexPromise.catch(() => {
      indexPromise = null;
    });
  }
  return indexPromise;
}

function sameRun(a, b) {
  return Object.keys(RUN).every((key) => a[key] === b[key]);
}

export async function loadHearing(id, indexRun) {
  const key = String(id);
  if (cache.has(key)) {
    const hit = cache.get(key);
    cache.delete(key);
    cache.set(key, hit);
    return hit;
  }
  const url = "data/hearings/" + encodeURIComponent(key) + ".json";
  const data = checkHearing(url, await getJson(url), id);
  if (indexRun && !sameRun(data.run, indexRun)) throw new DataError("format", url, "run: difere do run de data/index.json, então o arquivo é de outra exportação");
  cache.set(key, data);
  while (cache.size > CACHE_LIMIT) cache.delete(cache.keys().next().value);
  return data;
}
