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

export const CASE_NOTE =
  "As notas vêm de modelos e nenhuma é a chance de a afirmação estar certa; uma nota baixa diz só que a frase escolhida dá pouco apoio a ela.";

export const NO_SECOND_OPINION = "A segunda opinião, a do verificador, não foi calculada nesta exportação.";

export const SPLIT_GAP = 0.3;

const QUESTIONS = {
  p1_nli: { text: "A frase implica a afirmação, é neutra ou a contradiz?", value: "chance de “implica”" },
  p2_nli_reversed: { text: "A mesma pergunta, com as opções na ordem inversa", value: "chance de “implica”" },
  p3_inferable: { text: "Dá para inferir a afirmação a partir da frase?", value: "chance de “sim”" },
  p4_supports: { text: "A frase sustenta a afirmação?", value: "chance de “sim”" },
  p5_position: { text: "Que posição a frase toma sobre o que a afirmação diz?", value: "chance de “a mesma posição”" },
  p6_position_reversed: { text: "A mesma pergunta, com as opções na ordem inversa", value: "chance de “a mesma posição”" },
  p7_coverage: { text: "Quanto da afirmação está dito na frase?", value: "de nada (0) a tudo (1)" },
  p8_similarity: { text: "Quão parecidos no sentido são a frase e a afirmação?", value: "de nada (0) a mesmo sentido (1)" },
};

export function questionCopy(q) {
  return QUESTIONS[q.id] || { text: q.instructions, value: "" };
}

export const PROFILE_NOTE =
  "Este perfil foi escrito por um modelo de linguagem, gerado a partir das falas da pessoa nas audiências; confira a evidência. Cada item aponta para a frase das falas que mais se parece com ele, e só vale como pista para ler a transcrição.";

export const PROFILE_MATCH_NOTE =
  "A frase de cada item é a mais parecida por palavras (TF-IDF, sem acentos e sem palavras muito comuns) entre as falas das audiências que o perfil leu; parecida não quer dizer que a frase sustenta o item. Uma afirmação da matéria (UDV) fica ligada ao item quando a frase é a mesma que a UDV usa como evidência, ou quando o item é parecido com o texto da UDV. Os dois cortes foram escolhidos para a leitura da página, sem medida de acerto.";

export const CLAIM_STATES = {
  udv: { k: "udv", label: "ligado a uma afirmação da matéria", short: "com UDV" },
  passage: { k: "pas", label: "só com uma frase parecida nas falas", short: "só frase" },
  none: { k: "none", label: "sem frase parecida nas falas lidas", short: "sem frase" },
};

export const UDV_RULES = {
  same_sentence: "a frase do item é a evidência desta afirmação",
  similar_text: "o item é parecido com o texto desta afirmação",
};

export const SPLIT_NAMES = { train: "treino", validation: "validação", test: "teste" };

export function claimState(c) {
  if (c.udv) return CLAIM_STATES.udv;
  if (c.passage) return CLAIM_STATES.passage;
  return CLAIM_STATES.none;
}

export const DISCLOSE = { open: "Ver análise completa", close: "Recolher" };

export const DISCLOSE_HINTS = {
  home: "Aqui estão as primeiras; a lista completa segue a mesma ordem e a mesma busca.",
  wall: "Como ler os fios, todas as afirmações desta matéria, a busca por palavras na audiência e as notas de método.",
  case: "As frases em volta, as duas notas com seus cortes, as oito perguntas ao modelo, a cópia em inglês e as outras frases parecidas.",
  profile: "Todos os itens do perfil, a evidência de cada um, as audiências, as afirmações das matérias e como o perfil foi feito.",
  atlas: "Todos os perfis de atores e todas as audiências, com quem a matéria cita em cada uma.",
  reader: "As frases antes e depois, no mesmo ponto da transcrição.",
};

export const SUMMARY_JOBS = {
  find: "Onde está o trecho?",
  support: "O trecho sustenta a afirmação?",
};

export function findingLine(tierName) {
  if (tierName === "quote_found") return "Trecho encontrado por citação direta";
  if (tierName === "semantic_match_high") return "Trecho encontrado por semelhança de sentido";
  if (tierName === "semantic_match_weak") return "Só um trecho pouco parecido, abaixo do corte";
  if (tierName === "no_evidence") return "Nenhum trecho: a pessoa não tem frase para comparar";
  return "Nenhum trecho: a pessoa não foi achada entre quem fala";
}

export function findingDetail(tierName, cosText, cutText) {
  if (tierName === "quote_found") return "O começo das aspas aparece igual na fala; não há nota de semelhança.";
  if (tierName === "semantic_match_high" || tierName === "semantic_match_weak") return "Semelhança (cosseno) " + cosText + "; corte " + cutText + ".";
  if (tierName === "no_evidence") return "A pessoa fala, mas nenhuma frase tem pelo menos 4 palavras.";
  return "Pode ser que não tenha falado ou que apareça com outro nome.";
}

export function verifierLine(v, text) {
  if (!v) return "verificador: não calculado";
  return "verificador: " + (v.supported ? "sustenta" : "não sustenta") + " (" + text + ")";
}

export const VERIFIER_GAUGE_NOTE = "Nota de 0 a 1 com corte; não é a chance de a afirmação estar certa.";

export const VERIFIER_NONE = "Sem trecho, o verificador não tem o que ler.";

export const PROFILE_NOTE_SHORT = "Perfil escrito por um modelo de linguagem a partir das falas. Confira a evidência de cada item.";

export const PROFILE_BADGES = {
  udv: "ligado a afirmação de matéria",
  pas: "frase parecida nas falas",
  none: "sem frase parecida",
};

export const PROFILE_TOP_TITLE = "Três posições do perfil";

export const PROFILE_TOP_NOTE = "Primeiro os itens com mais evidência; a frase ligada é parecida por palavras, o que não quer dizer que ela sustenta o item.";

export const READER_SHORT = "Só a frase marcada. A análise completa mostra o que vem antes e depois.";

export const FINDING_SHORT = {
  quote_found: "Citação direta",
  semantic_match_high: "Parecido pelo sentido",
  semantic_match_weak: "Pouco parecido",
  no_evidence: "Nenhum trecho",
  person_not_resolved: "Nenhum trecho",
};

const BUCKET_SHORT = { q: "com aspas na fala", h: "com trecho parecido", w: "pouco parecido", u: "sem trecho" };

export function statTiles(tiers, nUdvs, nPeople) {
  const tiles = [{ n: nUdvs, label: nUdvs === 1 ? "afirmação" : "afirmações" }];
  if (nPeople != null) tiles.push({ n: nPeople, label: nPeople === 1 ? "pessoa citada" : "pessoas citadas" });
  bucketCounts(tiers).forEach((b) => tiles.push({ n: b.n, label: BUCKET_SHORT[b.key], k: b.key }));
  return tiles;
}

export const ATLAS_COPY = {
  lede: "As audiências, quem fala nelas e os perfis de quem aparece em mais de uma, num lugar só. A tecla <kbd>M</kbd> abre este mapa de qualquer página.",
  profiles: "Cada perfil foi escrito por um modelo a partir das falas; confira a evidência.",
  topActors: "Atores com mais afirmações nas matérias",
  recentHearings: "Audiências mais recentes",
  foundActors: "Atores que combinam com a busca",
  foundHearings: "Audiências que combinam com a busca",
  count: "Comece por um ator ou por uma audiência; a análise completa lista todos, em ordem alfabética e por data.",
  nActors: 4,
  nHearings: 3,
};

export const HOME_LABELS = { open: "Ver todas as matérias", close: "Recolher" };

export const HOME_FIRST = 12;
