import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "challenge"))

from utils.udv_pipeline import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    best_embedding_match,
    best_semantic_match,
    extract_quote,
    find_quote_evidence,
    get_embedding_model,
    resolve_person_speech,
    resolve_turn_name,
    split_into_turns,
    split_sentences,
)

INITIAL_SAMPLE_SIZE = 10
ADDITIONAL_SAMPLE_SIZE = 10
RANDOM_SEED = 42

HIGH_JUDGMENTS = [
    ("correta", "evidência é quase paráfrase direta da afirmação"),
    ("correta", "evidência cita quase literalmente o mesmo fato"),
    ("correta", "evidência confirma o risco de reincidência mencionado na opinião"),
    ("correta", "evidência cita o mesmo valor e destino da promoção"),
    ("parcial", "mesmo assunto, mas evidência não confirma a pergunta específica"),
    ("correta", "evidência repete quase literalmente a falta de profissionais especializados"),
    ("correta", "evidência confirma ausência de prejuízo e preservação do FGTS"),
    ("correta", "evidência cita as mesmas 26 recomendações"),
    ("correta", "evidência é quase citação direta da convicção relatada"),
    ("correta", "evidência confirma a articulação entre teoria e prática citada"),
    ("correta", "evidência confirma o pedido de desculpas relatado"),
    ("correta", "evidência confirma o pedido por debates antes de mudar as regras"),
    ("correta", "evidência é quase citação direta da referência à PNAB"),
    ("correta", "evidência é quase citação direta da acusação sobre o modelo de negócios"),
    ("parcial", "evidência confirma a lista de acusações, mas não a citação direta mais específica sobre a Mina 18"),
    ("correta", "evidência confirma a posição sobre segregação patrimonial e interesse do Congresso"),
    ("parcial", "evidência confirma a falta de aviso prévio, mas não a falta de explicação dos motivos"),
    ("correta", "evidência confirma a comparação de dificuldade de acompanhamento entre autistas"),
    ("correta", "evidência confirma o foco em comunidades e sustentabilidade ambiental"),
    ("correta", "evidência é quase citação direta sobre o momento de agir"),
]

WEAK_JUDGMENTS = [
    ("incorreta", "evidência trata de um assunto não relacionado ao comportamento citado"),
    ("parcial", "evidência é uma pergunta retórica sobre o mesmo tema, mas não afirma a conclusão"),
    ("parcial", "evidência cita o mesmo caso, mas não o compilado de e-mails específico"),
    ("incorreta", "evidência é uma saudação, sem relação com a afirmação"),
    ("parcial", "evidência fala do mesmo evento, mas não confirma a afirmação específica"),
    ("incorreta", "evidência parece genérica, não confirma a afirmação"),
    ("parcial", "evidência confirma o tema central, mas não a afirmação específica"),
    ("incorreta", "evidência é uma pergunta genérica, não confirma a estatística citada"),
    ("correta", "evidência confirma a necessidade de legislação alinhada a outros países"),
    ("parcial", "evidência menciona o mesmo tema, mas não confirma a afirmação específica"),
    ("incorreta", "evidência está truncada e não sustenta a afirmação"),
    ("correta", "evidência confirma o mesmo percentual e a mesma crítica de falta de controle"),
    ("incorreta", "evidência não tem relação com a afirmação sobre passagens de 2024"),
    ("incorreta", "evidência é uma pergunta crítica, não um elogio à gestão"),
    ("parcial", "evidência confirma o encaminhamento da reforma, mas não a transição energética"),
    ("parcial", "evidência confirma o tom transformador, mas fala de descarbonização, não de reestruturação do setor"),
    ("incorreta", "evidência é um slogan vago, sem relação com a afirmação sobre o mercado livre"),
    ("correta", "evidência confirma quase literalmente a citação sobre o investidor saber o risco"),
    ("incorreta", "evidência é genérica sobre o debate orçamentário, não confirma a correção de desigualdades"),
    ("parcial", "evidência confirma parte das recomendações internacionais, mas não todas as citadas"),
]

EMBEDDING_DIFF_JUDGMENTS = [
    ("parcial", "confirma o número de municípios afetados, não confirma a ausência de caladão generalizado"),
    ("incorreta", "pergunta genérica sobre negociação de passagens, não confirma o valor nem a crítica ao lucro"),
    ("parcial", "mesmo tema de milhas de clientes usadas, não confirma especificamente a falta de pagamento"),
    ("incorreta", "referencia só ter discutido o tema, não confirma convencimento de acabar com o mecanismo"),
    ("correta", "confirma diretamente o pedido de desculpas aos consumidores"),
    ("correta", "confirma a necessidade de debate plural, aberto, com todos à mesa"),
    ("parcial", "confirma a negociação em si, não confirma o foco em sustentabilidade ambiental e segurança das comunidades"),
    ("correta", "confirma a visão da agência e a extensão do processo por eventos climáticos"),
    ("parcial", "crítica recebida por comentário capacitista, tema próximo mas não confirma bate-boca literalmente"),
    ("parcial", "descreve a ordem de bloqueio, não confirma falta de base legal nem a remoção específica de postagens"),
    ("correta", "confirma diretamente que todos os expositores estavam de um lado só"),
    ("incorreta", "descarta causa eólica/solar, não fala sobre o relatório final não ter saído"),
    ("correta", "confirma que a recuperação foi feita a contento"),
    ("parcial", "mesmo tema de negativados e taxas do saque-aniversário, não confirma o número de 75%"),
    ("correta", "confirma as duas partes da opinião: precisa aprimorar, acabar agora seria cruel"),
    ("parcial", "confirma que funcionou, não confirma especificamente a redução da necessidade de marketing"),
    ("correta", "confirma o percentual de 43% e o tom de crítica"),
    ("incorreta", "trata de um problema de sistemas em 2022, ano e tema diferentes"),
    ("incorreta", "fala da formação acadêmica do próprio orador, não avalia a gestão do ministro"),
    ("correta", "confirma a reforma como oportuna e sua ligação com redução de emissões"),
    ("parcial", "fala em redefinir encargos, tema relacionado mas não confirma reestruturar o setor"),
    ("incorreta", "propõe reduzir tarifas por essencialidade, mecanismo diferente do citado"),
    ("correta", "confirma de perto a citação sobre o investidor entender o ativo e avaliar o risco"),
    ("parcial", "fala do caráter extensionista do PIBID na comunidade, ligação temática solta"),
    ("parcial", "confirma a existência de um arcabouço regulatório em geral, não as três instituições citadas"),
]


def build_pools(lds_records):
    high_pool = []
    weak_pool = []

    for hearing in lds_records[:20]:
        turns = [
            dict(turn, resolved_name=resolve_turn_name(turn))
            for turn in split_into_turns(hearing["transcricao"])
        ]
        for pessoa in hearing["metadados"]["envolvidos"]:
            matched_turns, speech = resolve_person_speech(pessoa, turns)
            if not matched_turns:
                continue
            sentences = split_sentences(speech)
            for opinion_text in pessoa["opinioes"]:
                quote = extract_quote(opinion_text)
                if quote is not None and find_quote_evidence(quote, speech) is not None:
                    continue
                best_sentence, score = best_semantic_match(opinion_text, sentences)
                if best_sentence is None:
                    continue
                entry = {
                    "hearing_id": hearing["id"],
                    "person": pessoa["nome"],
                    "opinion": opinion_text,
                    "evidence": best_sentence,
                    "score": score,
                }
                (high_pool if score >= CONFIDENCE_THRESHOLD else weak_pool).append(entry)

    return high_pool, weak_pool


def build_review_items(lds_records):
    random.seed(RANDOM_SEED)
    high_pool, weak_pool = build_pools(lds_records)

    initial_high = random.sample(high_pool, INITIAL_SAMPLE_SIZE)
    initial_weak = random.sample(weak_pool, INITIAL_SAMPLE_SIZE)
    remaining_high = [entry for entry in high_pool if entry not in initial_high]
    remaining_weak = [entry for entry in weak_pool if entry not in initial_weak]
    additional_high = random.sample(remaining_high, ADDITIONAL_SAMPLE_SIZE)
    additional_weak = random.sample(remaining_weak, ADDITIONAL_SAMPLE_SIZE)

    sample_high = initial_high + additional_high
    sample_weak = initial_weak + additional_weak

    items = []
    for entry, (judgment, note) in zip(sample_high, HIGH_JUDGMENTS):
        items.append({"tier": "semantic_match_high", **entry, "judgment": judgment, "note": note})
    for entry, (judgment, note) in zip(sample_weak, WEAK_JUDGMENTS):
        items.append({"tier": "semantic_match_weak", **entry, "judgment": judgment, "note": note})
    return items


def build_embedding_diff_items(lds_records, review_items):
    model = get_embedding_model()
    hearings_by_id = {hearing["id"]: hearing for hearing in lds_records[:20]}

    divergent = []
    for item in review_items:
        hearing = hearings_by_id[item["hearing_id"]]
        turns = [
            dict(turn, resolved_name=resolve_turn_name(turn))
            for turn in split_into_turns(hearing["transcricao"])
        ]
        pessoa = next(p for p in hearing["metadados"]["envolvidos"] if p["nome"] == item["person"])
        _, speech = resolve_person_speech(pessoa, turns)
        sentences = split_sentences(speech)
        embedding_sentence, embedding_score = best_embedding_match(item["opinion"], sentences, model)
        if embedding_sentence != item["evidence"]:
            divergent.append(
                {
                    "hearing_id": item["hearing_id"],
                    "person": item["person"],
                    "opinion": item["opinion"],
                    "evidence": embedding_sentence,
                    "score": embedding_score,
                    "tfidf_evidence": item["evidence"],
                    "tfidf_judgment": item["judgment"],
                }
            )

    items = []
    for entry, (judgment, note) in zip(divergent, EMBEDDING_DIFF_JUDGMENTS):
        items.append({"tier": "embedding_diff", **entry, "judgment": judgment, "note": note})
    return items


if __name__ == "__main__":
    dataset_path = REPO_ROOT / "challenge" / "dataset" / "PublicHearingBR_LDS.jsonl"
    output_path = REPO_ROOT / "challenge" / "udv_manual_review.json"
    with open(dataset_path) as f:
        lds_records = [json.loads(line) for line in f]
    review_items = build_review_items(lds_records)
    embedding_diff_items = build_embedding_diff_items(lds_records, review_items)
    all_items = review_items + embedding_diff_items
    with open(output_path, "w") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    print(f"{len(all_items)} itens escritos em {output_path}")
