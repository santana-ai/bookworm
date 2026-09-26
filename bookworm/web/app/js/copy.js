import { countLabel, joinPt } from "./text.js";

const TIERS = {
  quote_found: { k: "q", label: "Achamos o começo das aspas na fala", short: "aspas na fala", tag: "começo das aspas" },
  semantic_match_high: { k: "h", label: "Achamos um trecho parecido", short: "parecido", tag: "trecho parecido" },
  semantic_match_weak: { k: "w", label: "Só achamos algo pouco parecido", short: "pouco parecido", tag: "pouco parecido" },
  no_evidence: { k: "n", label: "Não havia frase para comparar", short: "sem frase", tag: "" },
  person_not_resolved: { k: "u", label: "Não achamos a pessoa entre quem fala", short: "pessoa não achada", tag: "" },
};

export const TIER_ORDER = ["quote_found", "semantic_match_high", "semantic_match_weak", "no_evidence", "person_not_resolved"];

export function tierOf(name) {
  return TIERS[name];
}

const BUCKETS = [
  { key: "q", tiers: ["quote_found"], label: "com o começo das aspas achado na fala" },
  { key: "h", tiers: ["semantic_match_high"], label: "com um trecho parecido" },
  { key: "w", tiers: ["semantic_match_weak"], label: "só com algo pouco parecido" },
  { key: "u", tiers: ["no_evidence", "person_not_resolved"], label: "sem trecho achado" },
];

export function bucketCounts(tiers) {
  return BUCKETS.map((b) => ({ key: b.key, n: b.tiers.reduce((a, t) => a + tiers[t], 0), label: b.label }));
}

export function bucketSentence(tiers) {
  const parts = bucketCounts(tiers)
    .filter((b) => b.n > 0)
    .map((b) => b.n + " " + b.label);
  return parts.length ? joinPt(parts) + "." : "";
}

export function statementsSentence(nUdvs, nPeople) {
  return countLabel(nUdvs, "afirmação atribuída", "afirmações atribuídas") + " a " + countLabel(nPeople, "pessoa", "pessoas");
}

export const NOT_CHECKED =
  "Os resultados vêm de um procedimento automático e ainda não foram conferidos por uma pessoa. Servem de pista para ler a transcrição, não de prova.";

export const NOT_CHECKED_SHORT = "Resultados de um procedimento automático, que ainda não foram conferidos por uma pessoa: servem de pista, não de prova.";
