import { countLabel, fmtCut, fmtDate, fmtInt, joinPt, plural } from "./text.js";

const TIERS = {
  quote_found: { k: "q", label: "Aspas achadas na fala", short: "aspas achadas na fala", tag: "aspas achadas" },
  semantic_match_high: { k: "h", label: "Trecho parecido pelo sentido", short: "trecho parecido", tag: "trecho parecido" },
  semantic_match_weak: { k: "w", label: "Só um trecho pouco parecido", short: "pouco parecido", tag: "pouco parecido" },
  no_evidence: { k: "n", label: "Sem frase para comparar", short: "sem frase", tag: "" },
  person_not_resolved: { k: "u", label: "Pessoa não achada entre quem fala", short: "pessoa não achada", tag: "" },
};

export const TIER_ORDER = ["quote_found", "semantic_match_high", "semantic_match_weak", "no_evidence", "person_not_resolved"];

export function tierOf(name) {
  return TIERS[name];
}

const BUCKETS = [
  { key: "q", tiers: ["quote_found"], label: TIERS.quote_found.short },
  { key: "h", tiers: ["semantic_match_high"], label: TIERS.semantic_match_high.short },
  { key: "w", tiers: ["semantic_match_weak"], label: TIERS.semantic_match_weak.short },
  { key: "u", tiers: ["no_evidence", "person_not_resolved"], label: "sem trecho" },
];

export const BUCKET_LABELS = Object.fromEntries(BUCKETS.map((b) => [b.key, b.label]));

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

export const MEASURES = { similarity: "semelhança", support: "apoio", words: "palavras em comum" };

let SITE_VALIDATION = null;

export function setValidation(v) {
  SITE_VALIDATION = v || null;
}

export function validation() {
  return SITE_VALIDATION;
}

export const SPLIT_NAMES = { train: "treino", validation: "validação", test: "teste" };

const JUDGMENT_WORDS = { correta: "correto", parcial: "parcialmente correto", incorreta: "incorreto" };
const SPEAKER_WORDS = { falou: "achou fala desta pessoa na transcrição", nao_falou: "não achou fala desta pessoa na transcrição", nao_sei: "não soube dizer se a pessoa falou" };
const STRATUM_NAMES = {
  direct_quote: TIERS.quote_found.short,
  semantic_match_high: "trecho parecido, sem aspas curtas",
  semantic_with_short_quote: "trecho parecido com aspas curtas",
  semantic_match_weak: TIERS.semantic_match_weak.short,
};
const METRIC_NAMES = { strict_precision: "a parte de corretas", tolerant_precision: "a parte de corretas e parciais" };

function pctText(x) {
  return Math.round(x * 100) + "%";
}

function splitsText(splits) {
  const names = (splits || []).map((s) => SPLIT_NAMES[s] || s);
  if (!names.length) return "";
  return names.length === 1 ? " do conjunto de " + names[0] : " dos conjuntos de " + joinPt(names);
}

function whoChecked(V) {
  return V.single_annotator ? "Uma pessoa" : "Anotadores";
}

function judgedTotal(V) {
  return Object.keys(V.tiers).reduce((a, t) => a + V.tiers[t].udvs, 0);
}

function failedCriteria(V) {
  return V.criteria.filter((c) => c.status !== "PASS");
}

function criteriaSentence(V) {
  const n = V.criteria.length;
  if (!n) return "";
  const failed = failedCriteria(V).length;
  if (!failed) return plural(n, "O critério fixado antes da conferência foi atingido.", "Os " + n + " critérios fixados antes da conferência foram atingidos.");
  if (failed === n) return plural(n, "O critério fixado antes da conferência não foi atingido.", (n === 2 ? "Os dois" : "Os " + n) + " critérios fixados antes da conferência não foram atingidos.");
  return failed + " dos " + n + " critérios fixados antes da conferência não foram atingidos.";
}

function tierCountText(t, row) {
  return TIERS[t].short + ", " + row.correta + " de " + row.udvs + " " + plural(row.correta, "correta", "corretas") + (row.parcial ? " e " + row.parcial + " " + plural(row.parcial, "parcial", "parciais") : "");
}

export function validationLead() {
  return SITE_VALIDATION ? "Conferência humana." : "Ainda sem conferência humana.";
}

export function validationShort() {
  const V = SITE_VALIDATION;
  if (!V) return "Resultados de um procedimento automático, que ainda não foram conferidos por uma pessoa: servem de pista, não de prova.";
  const parts = Object.keys(V.tiers).map((t) => tierCountText(t, V.tiers[t]));
  return whoChecked(V) + " conferiu " + fmtInt(judgedTotal(V)) + " afirmações" + splitsText(V.splits) + ": " + parts.join("; ") + ". " + criteriaSentence(V) + " O resto são pistas para ler a transcrição, não prova.";
}

export function validationLong() {
  const V = SITE_VALIDATION;
  if (!V) return ["Os resultados vêm de um procedimento automático e ainda não foram conferidos por uma pessoa. Servem de pista para ler a transcrição, não de prova."];
  const level = Math.round(V.confidence_level * 100);
  const out = [];
  out.push(
    whoChecked(V) + " conferiu " + fmtInt(judgedTotal(V)) + " afirmações" + splitsText(V.splits) + ", perguntando se o trecho escolhido, dito pela própria pessoa, sustenta o que a matéria atribui a ela. " +
      (V.single_annotator ? "Foi uma única pessoa, então não há medida de concordância entre anotadores. " : "") +
      "A conferência não diz se a afirmação é verdadeira.",
  );
  Object.keys(V.tiers).forEach((t) => {
    const r = V.tiers[t];
    const st = r.strict;
    const to = r.tolerant;
    out.push(
      TIERS[t].label + ": " + r.correta + " de " + r.udvs + " corretas (" + pctText(st.estimate) + "; intervalo de " + level + "%: " + pctText(st.low) + " a " + pctText(st.high) + ")" +
        (r.parcial ? ", mais " + r.parcial + " " + plural(r.parcial, "parcial", "parciais") + "; contando as parciais, " + to.successes + " de " + to.trials + " (" + pctText(to.estimate) + "; " + pctText(to.low) + " a " + pctText(to.high) + ")" : "") +
        ".",
    );
  });
  V.criteria.forEach((c) => {
    const stratum = STRATUM_NAMES[c.stratum] || c.stratum;
    const metric = METRIC_NAMES[c.metric] || c.metric;
    out.push(
      "Critério fixado em " + fmtDate(V.criteria_declared_on) + ", antes da conferência: em " + stratum + ", o limite inferior do intervalo para " + metric + " precisava chegar a " + pctText(c.min_wilson_lower) +
        "; ficou em " + pctText(c.observed.low) + " (" + c.observed.successes + " de " + c.observed.trials + "). " + (c.status === "PASS" ? "Atingido." : "Não atingido."),
    );
  });
  if (V.inheritance) {
    const I = V.inheritance;
    out.push(
      "A conferência foi feita com os trechos da primeira versão da busca. Nesta versão, " + I.judged_inherited + " dos " + (I.judged_inherited + I.judged_reannotated) +
        " itens herdaram o julgamento, porque o trecho ficou igual ou só ganhou frases do mesmo turno; os outros " + I.judged_reannotated +
        " foram julgados de novo. A herança supõe que acrescentar uma frase do mesmo turno não tira o apoio que o trecho dava.",
    );
  }
  if (V.speaker_check && V.speaker_check.udvs) {
    const S = V.speaker_check;
    out.push(
      "Em " + S.udvs + " afirmações sem trecho conferidas, a pessoa que conferiu achou fala da pessoa citada na transcrição em " + S.falou + ": nesses casos a pessoa falou, mas a busca não registrou fala dela que servisse para comparar. A resposta é dada por pessoa e vale para todas as afirmações dela, então as afirmações não são julgamentos independentes.",
    );
  }
  return out;
}

export function judgmentLine(udvId) {
  const V = SITE_VALIDATION;
  const j = V && V.udvs ? V.udvs[udvId] : null;
  if (!j) return "";
  if (j.question === "pessoa_falou") return "Esta afirmação está na amostra conferida: a pessoa que conferiu " + (SPEAKER_WORDS[j.judgment] || j.judgment) + ".";
  return "Esta afirmação está na amostra conferida: a pessoa que conferiu julgou o trecho " + (JUDGMENT_WORDS[j.judgment] || j.judgment) + ".";
}

export function judgmentOf(udvId) {
  const V = SITE_VALIDATION;
  return V && V.udvs ? V.udvs[udvId] || null : null;
}

export const CASE_NOTE =
  "Nem a semelhança nem o apoio são a chance de a afirmação estar certa. Um apoio baixo diz só que o trecho escolhido sustenta pouco a afirmação; a busca pode ter escolhido o trecho errado, ou a matéria pode ter resumido várias falas.";

export const NO_SECOND_OPINION = "O apoio do verificador não foi calculado nesta exportação.";

export const SPLIT_GAP = 0.3;

const QUESTIONS = {
  p1_nli: { text: "O trecho implica a afirmação, é neutro ou a contradiz?", value: "chance de “implica”" },
  p2_nli_reversed: { text: "A mesma pergunta, com as opções na ordem inversa", value: "chance de “implica”" },
  p3_inferable: { text: "Dá para inferir a afirmação a partir do trecho?", value: "chance de “sim”" },
  p4_supports: { text: "O trecho sustenta a afirmação?", value: "chance de “sim”" },
  p5_position: { text: "Que posição o trecho toma sobre o que a afirmação diz?", value: "chance de “a mesma posição”" },
  p6_position_reversed: { text: "A mesma pergunta, com as opções na ordem inversa", value: "chance de “a mesma posição”" },
  p7_coverage: { text: "Quanto da afirmação está dito no trecho?", value: "de nada (0) a tudo (1)" },
  p8_similarity: { text: "Quão parecidos no sentido são o trecho e a afirmação?", value: "de nada (0) a mesmo sentido (1)" },
};

export const P4_ID = "p4_supports";

export function questionCopy(q) {
  return QUESTIONS[q.id] || { text: q.instructions, value: "" };
}

export const PROFILE_NOTE =
  "Este perfil foi escrito por um modelo de linguagem, gerado a partir das falas da pessoa nas audiências; confira a evidência. Cada item aponta para a frase das falas que mais se parece com ele, e só vale como pista para ler a transcrição.";

export const PROFILE_MATCH_NOTE =
  "A frase de cada item é a que tem mais palavras em comum com ele entre as falas das audiências que o perfil leu, pesando mais as palavras raras (TF-IDF, sem acentos e sem palavras muito comuns); ter palavras em comum não quer dizer que a frase sustenta o item. Uma afirmação da matéria fica ligada ao item quando a frase é a mesma que a afirmação usa como trecho, ou quando o item tem muitas palavras em comum com o texto da afirmação. Os dois cortes foram escolhidos para a leitura da página, sem medida de acerto.";

export const CLAIM_STATES = {
  udv: { k: "udv", label: "ligado a uma afirmação da matéria", short: "com afirmação da matéria" },
  passage: { k: "pas", label: "só com uma frase parecida nas falas", short: "só frase parecida" },
  none: { k: "none", label: "sem frase parecida nas falas lidas", short: "sem frase parecida" },
};

export const UDV_RULES = {
  same_sentence: "a frase do item é o trecho desta afirmação",
  similar_text: "o item tem muitas palavras em comum com o texto desta afirmação",
};

export function claimState(c) {
  if (c.udv) return CLAIM_STATES.udv;
  if (c.passage) return CLAIM_STATES.passage;
  return CLAIM_STATES.none;
}

export const DISCLOSE = { open: "Ver análise completa", close: "Recolher" };

export const DISCLOSE_HINTS = {
  home: "Aqui estão as primeiras; a lista completa segue a mesma ordem e a mesma busca.",
  wall: "Como ler os fios, todas as afirmações desta matéria com o apoio de cada uma, a busca por palavras na audiência e as notas de método.",
  case: "As frases em volta, a semelhança e o apoio com seus cortes, as oito perguntas ao modelo, a cópia em inglês e os outros trechos parecidos.",
  profile: "Todos os itens do perfil, a evidência de cada um, as audiências, as afirmações das matérias e como o perfil foi feito.",
  atlas: "Todos os perfis de atores e todas as audiências, com quem a matéria cita em cada uma.",
  reader: "As frases antes e depois, no mesmo ponto da transcrição.",
};

export const SUMMARY_JOBS = {
  find: "Onde está o trecho?",
  support: "Quanto o trecho apoia a afirmação?",
};

export function findingDetail(tierName, simText, cutText) {
  if (tierName === "quote_found") return "O começo das aspas aparece igual na fala; a semelhança não foi usada para escolher o trecho.";
  if (tierName === "semantic_match_high") return "Semelhança de sentido " + simText + "; a partir de " + cutText + " o trecho conta como parecido.";
  if (tierName === "semantic_match_weak") return "Semelhança de sentido " + simText + ", abaixo de " + cutText + ": é o mais parecido que a busca achou.";
  if (tierName === "no_evidence") return "A transcrição registra turnos desta pessoa, mas sem frase de pelo menos 4 palavras para comparar.";
  return "Ninguém com este nome aparece entre quem fala. Pode ser que não tenha falado ou que apareça com outro nome.";
}

const BANDS = {
  weak: { k: "weak", label: "apoio fraco", Label: "Apoio fraco", lvl: 0 },
  uncertain: { k: "uncertain", label: "apoio incerto", Label: "Apoio incerto", lvl: 1 },
  strong: { k: "strong", label: "apoio forte", Label: "Apoio forte", lvl: 2 },
};

export function verifierCut(signals) {
  const v = signals.verifier;
  const udv = v.udv_threshold;
  if (!udv) return { cut: v.threshold, train: null, low: v.threshold, high: null };
  return { cut: udv.value, train: v.threshold, rule: udv.rule, low: udv.value, high: v.threshold > udv.value ? v.threshold : null };
}

export function cutsOf(low, high) {
  return { low, high: high != null && high > low ? high : null };
}

export function bandOf(p, C) {
  if (p < C.low) return BANDS.weak;
  if (C.high != null && p < C.high) return BANDS.uncertain;
  return BANDS.strong;
}

export function bandList(C) {
  return C.high != null ? [BANDS.weak, BANDS.uncertain, BANDS.strong] : [BANDS.weak, BANDS.strong];
}

export function bandRange(b, C) {
  if (b.k === "weak") return "abaixo de " + fmtCut(C.low);
  if (b.k === "uncertain") return "de " + fmtCut(C.low) + " a " + fmtCut(C.high);
  return "a partir de " + fmtCut(C.high != null ? C.high : C.low);
}

function sameCut(a, b) {
  return typeof a === "number" && typeof b === "number" && Math.abs(a - b) < 1e-9;
}

export function bandEvidence(b, C) {
  const V = SITE_VALIDATION;
  const vb = V && V.verifier_bands;
  if (!vb || C.high == null || !sameCut(vb.cuts[0], C.low) || !sameCut(vb.cuts[1], C.high)) return null;
  return vb.bands.find((x) => x.band === b.k) || null;
}

export function bandEvidenceLine(b, C) {
  const e = bandEvidence(b, C);
  if (!e || !e.udvs) return "";
  return "Na conferência humana, trechos nesta faixa estavam corretos em " + e.correta + " de " + e.udvs + " casos" + (e.parcial ? " (mais " + e.parcial + " " + plural(e.parcial, "parcial", "parciais") + ")" : "") + ".";
}

export const SUPPORT_NOTE = "O apoio vai de 0 a 1 e não é a chance de a afirmação estar certa.";

export function supportLine(b, text) {
  return b.label + " (" + text + ")";
}

export function p4Mismatch(b, p4, C) {
  if (typeof p4 !== "number") return false;
  if (b.k === "weak") return p4 >= (C.high != null ? C.high : C.low);
  return p4 < C.low;
}

export function p4Note(p4Text) {
  return "O apoio combina as oito respostas do modelo, nas leituras em português e em inglês; não é a resposta da pergunta “O trecho sustenta a afirmação?”, que aqui ficou em " + p4Text + ".";
}

export function questionsDisagree(tierK, b) {
  if ((tierK === "q" || tierK === "h") && b.k === "weak") return "A busca achou um trecho, mas o verificador vê pouco apoio nele. Vale ler os outros trechos parecidos, na parte 4: a busca pode ter escolhido o trecho errado.";
  if (tierK === "w" && b.k === "strong") return "A busca achou só um trecho pouco parecido, mas o verificador vê apoio forte nele. Vale ler o trecho em volta.";
  return "";
}

const SENTENCE_UNIT = {
  window: false,
  chosen: "a frase escolhida",
  Chosen: "A frase escolhida",
  read: "uma frase só",
  heading: "Outras frases parecidas que a pessoa disse",
  count: (n) => countLabel(n, "frase", "frases"),
  top: (n) => plural(n, "A mais parecida", "As " + n + " mais parecidas"),
  rank: (r) => " e é a " + r + "ª desta lista pela semelhança.",
  notIn: (n) => ", e não está entre estas " + n + " frases mais parecidas pela semelhança.",
  gap: (g) => "A segunda frase ficou " + g + " abaixo da primeira.",
  only: "Só havia esta frase para comparar.",
  best: "a frase escolhida",
};

const WINDOW_SIZES = { window2: "duas", window3: "três" };

function windowUnit(size) {
  const what = size + " frases seguidas do mesmo turno";
  return {
    window: true,
    chosen: "o trecho escolhido (" + what + ")",
    Chosen: "O trecho escolhido, " + what,
    read: "um trecho de " + size + " frases",
    heading: "Outros trechos parecidos que a pessoa disse",
    count: (n) => countLabel(n, "trecho de " + size + " frases", "trechos de " + size + " frases"),
    top: (n) => plural(n, "O mais parecido", "Os " + n + " mais parecidos"),
    rank: (r) => " e é o " + r + "º desta lista pela semelhança.",
    notIn: (n) => ", e não está entre estes " + n + " trechos mais parecidos pela semelhança (as aspas podem cobrir mais de duas frases).",
    gap: (g) => "O segundo trecho ficou " + g + " abaixo do primeiro.",
    only: "Só havia este trecho para comparar.",
    best: "o trecho escolhido",
  };
}

export function unitCopy(run) {
  const size = run && WINDOW_SIZES[run.semantic_unit];
  return size ? windowUnit(size) : SENTENCE_UNIT;
}

export const VERIFIER_NONE = "Sem trecho, o verificador não tem o que ler.";

export const PROFILE_NOTE_SHORT = "Perfil escrito por um modelo de linguagem a partir das falas. Confira a evidência de cada item.";

export const PROFILE_BADGES = {
  udv: "ligado a afirmação de matéria",
  pas: "frase parecida nas falas",
  none: "sem frase parecida",
};

export const PROFILE_TOP_TITLE = "Três posições do perfil";

export const PROFILE_TOP_NOTE = "Primeiro os itens com mais evidência; a frase ligada tem palavras em comum com o item, o que não quer dizer que ela sustenta o item.";

export const READER_SHORT = "A análise completa mostra o que vem antes e depois.";

export function statTiles(tiers, nUdvs, nPeople) {
  const tiles = [{ n: nUdvs, label: nUdvs === 1 ? "afirmação" : "afirmações" }];
  if (nPeople != null) tiles.push({ n: nPeople, label: nPeople === 1 ? "pessoa citada" : "pessoas citadas" });
  bucketCounts(tiers).forEach((b) => tiles.push({ n: b.n, label: b.label, k: b.key }));
  return tiles;
}

export const GLOSSARY = [
  ["UDV", "Unidade Deliberativa Verificável: uma afirmação que a matéria atribui a alguém, ligada a essa pessoa e ao trecho da fala dela que a busca escolheu."],
  ["Semelhança", "Quanto o sentido do trecho se parece com o da afirmação, de 0 a 1. É o cosseno entre as representações das duas frases, calculadas por um modelo de frases em português. Não é a chance de a afirmação estar certa."],
  ["Apoio", "O valor de 0 a 1 que um verificador treinado dá ao trecho como sustentação da afirmação. Também não é a chance de a afirmação estar certa."],
  ["Corte", "O valor a partir do qual a página muda o resultado: um para a semelhança e dois para o apoio, que separam as faixas fraco, incerto e forte."],
  ["Rodada", "Uma execução completa da busca, com nome próprio (por exemplo, udv_v2), para que cada número da página aponte para os arquivos que o geraram."],
];

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
