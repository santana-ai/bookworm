"""Revisao dos 153 pares de nomes ambiguos de challenge/artifacts/hearing_actors/ambiguous_names.json.

Vereditos PROPOSTOS por agente (Claude, 23/09/2026) para revisao do usuario; nao sao julgamento
humano e nao devem ser copiados para dentro de challenge/ como se fossem.

Evidencia usada, toda do proprio dataset (PublicHearingBR_LDS.jsonl):
- metadados.envolvidos (nome + cargo) das audiencias de cada lado;
- contexto da transcricao imediatamente antes do primeiro turno de cada chave (a apresentacao
  feita pelo presidente costuma trazer nome completo e cargo);
- inicio da propria fala (auto-apresentacao);
- partido/UF do cabecalho, quando existe;
- carimbo de data da materia, para os casos de mudanca de cargo (Paulo Teixeira, Rodrigo Agostinho).

Vereditos: "same" (mesma pessoa, mesclar), "different" (pessoas distintas, manter separadas),
"uncertain" (precisa de olho humano). Confianca: "alta" ou "media".

Resolucoes do usuario (24/09/2026): par 122 confirmado como mesma pessoa (o RODRIGO da aud. 148 e o
Cel. Bordeaux); par 52 confirmado como mesma pessoa (Duarte = Duarte Goncalves Jr., partido do
cabecalho da aud. 115 tratado como erro de transcricao); par 6 delegado ao agente, decisao registrada
em REASSIGNMENTS (dividir a chave ALEXANDRE SAMPAIO por audiencia). Par 149 segue incerto e fora da
mescla. Os grupos confirmados foram aplicados na secao [merges] de
challenge/configs/hearing_actors.toml; este script imprime o trecho TOML correspondente.

O arquivo de pares lido aqui e a copia pre-mescla preservada em backup/, porque o build regenera
challenge/artifacts/hearing_actors/ambiguous_names.json ja com a mescla aplicada.

Uso (da raiz do repo): uv run --project challenge python claude_ambiguous_names_review.py
Imprime o resumo, os grupos de mescla, o efeito nos arquivos single/multi e o TOML da secao [merges].
"""

import json
from pathlib import Path

AMBIGUOUS_PATH = Path("backup/ambiguous_names_v00.json")
CURRENT_PAIRS_PATH = Path("challenge/artifacts/hearing_actors/ambiguous_names.json")

VERDICTS: dict[int, dict[str, str]] = {}


def add(indices: list[int], verdict: str, confidence: str, evidence: str) -> None:
    for index in indices:
        VERDICTS[index] = {"verdict": verdict, "confidence": confidence, "evidence": evidence}


add([0, 1], "different", "alta",
    "A e o Deputado Cabo Albuquerque (Republicanos-RR); B e a representante do Ministerio da "
    "Cidadania na aud. 84 (mesma pessoa nos pares 0 e 1, ver par 30).")
add([2], "different", "alta",
    "B e Carlos Antonio, da FENAFIM, convidado virtual da aud. 79; A e deputado de RR.")
add([3], "different", "alta",
    "B e Renata Albuquerque Ribeiro, coordenadora de energia do IDEC (aud. 104); A e deputado.")
add([4], "different", "alta",
    "A e o Secretario Nacional dos Direitos da Pessoa Idosa (5 audiencias, cargo nos envolvidos); "
    "B representa o MDIC na aud. 130.")
add([5], "different", "alta",
    "A e Superintendente-Geral da CVM (aud. 8); B fala 'pelo FICO e pela Camisa 12 do Inter' e se "
    "diz consul (aud. 101).")
add([6], "split", "alta",
    "B (Diretor da CNC, aud. 108) e a mesma pessoa que A na aud. 38 (Diretor da CNC), mas A na "
    "aud. 14 e 'Presidente da Associacao dos Empreendedores e Vitimas da Mineracao em Maceio': a "
    "chave ALEXANDRE SAMPAIO junta duas pessoas. Resolucao (delegada pelo usuario em 24/09/2026): "
    "o turno da aud. 38 e reatribuido para ALEXANDRE SAMPAIO DE ABREU e a aud. 14 fica como ator "
    "proprio; ver REASSIGNMENTS.")
add([7], "different", "alta",
    "A e a Dra. Aline Duarte, da ASFAV (aud. 201); B e o Deputado Duarte (PSB-MA).")
add(list(range(8, 30)) , "different", "media",
    "ANA (aud. 100) e participante local sem sobrenome no cabecalho (contexto: delegadas de Goias "
    "convidadas a falar; 2 turnos curtos); nenhuma das 'Ana' de nome completo tem papel compativel "
    "com esse contexto. Excecao: par 30 nao esta neste bloco.")
add([30], "same", "alta",
    "Mesma audiencia 84 e mesma expositora do Ministerio da Cidadania; o cabecalho ganha o "
    "'E MELO' no meio da exposicao dela.")
add([31], "different", "alta",
    "A e professora de pedagogia (UNIRIO, aud. 125); B se apresenta como 'Cristina, da Associacao "
    "Maes na Luta' (aud. 100).")
add([32], "same", "media",
    "Ambas em audiencias sobre cannabis medicinal; na aud. 146 e representante da FACT; sobrenome "
    "raro (Aboin); na aud. 126 o cargo nao e dito.")
add([33], "same", "alta",
    "Mesma audiencia 99, turnos consecutivos da mesma professora de sociologia da UnB; o segundo "
    "cabecalho sai encurtado.")
add([34], "different", "alta",
    "A e assessora juridica da CONDSEF (aud. 55); B se apresenta como 'Candido, da ONG Gestos', "
    "homem (aud. 123).")
add([35], "same", "alta",
    "Mesma pauta (motoristas de aplicativo): na aud. 112 fala pela Federacao Nacional dos "
    "Sindicatos dos Motoristas de Aplicativos; na aud. 87 e Presidente do sindicato da categoria "
    "no RS.")
add([36], "different", "alta",
    "A e Carla Almeida, da Articulacao Social de Luta contra a Tuberculose; B e Patricia Almeida, "
    "jornalista do Movimento Down (aud. 53).")
add([37], "different", "alta",
    "B e Patricia Gomes, diretora da ABIMAQ (aud. 61); A e da area de tuberculose.")
add([38], "different", "alta",
    "A se apresenta: 'Eu me chamo Veras; venho de Sousa, na Paraiba', ABRAVIT (aud. 152); B e o "
    "Deputado Carlos Veras (PT-PE).")
add([39], "different", "alta",
    "A e Secretario de Controle Externo do TCU (aud. 131); B e o Deputado Rafael Simoes "
    "(UNIAO-MG).")
add([40], "different", "alta",
    "A e Diretor da SUSEP (aud. 152); B e o Deputado Roberto Alves (Republicanos-SP).")
add([41], "same", "alta",
    "Mesmo cargo explicito nas tres audiencias: Secretaria-Geral de Articulacao Institucional da "
    "Defensoria Publica da Uniao.")
add([42], "same", "media",
    "Nome raro (Cleo Bohn); ambas em pautas de deficiencia (federacao de Sindrome de Down na aud. "
    "160; orgulho autista na aud. 94, onde e chamada de 'Cleo Lima').")
add([43], "different", "alta",
    "B e diretora-executiva da Mercy for Animals (aud. 191); A e da Associacao Maes na Luta.")
add([44], "different", "alta", "B e consultora da OPAS sobre envelhecimento (aud. 52).")
add([45], "different", "alta", "B e a Deputada Silvia Cristina (PL-RO).")
add([46], "different", "alta", "B e coordenadora-adjunta do Comite Pro-Rio Doce (aud. 13).")
add([47], "different", "alta", "B e conselheira do Conselho Nacional de Saude (aud. 127).")
add([48], "different", "alta", "B e Secretaria de Turismo de Aparecida-SP (aud. 64).")
add([49], "different", "alta",
    "A e o Frei David, da EDUCAFRO (aud. 180); B e o Deputado Raimundo Santos (PSD-PA).")
add([50], "same", "alta",
    "Ambos da Federacao Unica dos Petroleiros; os envolvidos da aud. 71 registram 'Deyvid Bacelar, "
    "Coordenador-geral da FUP'.")
add([51], "different", "alta",
    "A e o Deputado Domingos Neto (PSD-CE, aud. 91); B e 'historiador e ex-deputado federal' nos "
    "envolvidos da aud. 39.")
add([52], "same", "alta",
    "Confirmado pelo usuario em 24/09/2026: mesma pessoa (Deputado Duarte Jr., PSB-MA); o "
    "'Bloco/PODE - MG' do cabecalho da aud. 115 e tratado como erro de transcricao. Por "
    "transitividade com o par 53, o grupo vira DUARTE + DUARTE JR. + DUARTE GONCALVES JR, o que "
    "resolve tambem o primeiro achado extra.")
add([53], "same", "alta",
    "Mesmo partido/UF (Bloco/PSB-MA) e o presidente chama 'Deputado Duarte'; cabecalho da aud. 53 "
    "sai sem o 'Jr.'.")
add([54], "different", "alta", "B e Desembargadora do TJMG (aud. 121); A e deputado do MA.")
add([55], "different", "alta", "B e professora da UnB (aud. 162).")
add([56], "different", "alta", "B e presidente do SINDSEPEM/VAL (aud. 55).")
add([57], "different", "alta", "B e depoente da CPI (caso 123Milhas, aud. 6).")
add([58], "same", "alta",
    "Mesma empresa: Urbano Norte (proprietario na aud. 177; Diretor-Presidente na aud. 87).")
add([59], "same", "media",
    "Mesma pauta (motoristas de aplicativo, auds. 87 e 177); na 87 e Vice-Presidente da federacao "
    "nacional dos motoristas; na 177 fala como motorista; mesmo prenome duplo.")
add([60], "same", "alta",
    "Mesmo cargo explicito nas duas audiencias: Coordenador do Departamento de Doencas Oculares "
    "da SBD.")
add([61], "different", "alta",
    "A e o Deputado Fernando Monteiro (PP-PE); B e cardiologista convidado da aud. 60.")
add([62], "same", "alta",
    "Mesmo cargo: Procurador da Republica do MPF nas auds. 98 e 100; sobrenome raro (Lodder).")
add([63], "different", "alta",
    "O GLAUBER da aud. 94 e um convidado que faz apresentacao musical ('Eu vou tocar.') e nao e "
    "tratado como deputado; B e o Deputado Glauber Braga (PSOL-RJ).")
add([64], "same", "alta",
    "Mesma pauta (motoristas de aplicativo) nas auds. 87, 112 e 177; o presidente chama 'Gleidson "
    "Veras' nas tres.")
add([65], "different", "alta",
    "A e Coronel do Exercito, chefe de operacoes de fronteira (aud. 205); B e o Deputado Luiz "
    "Lima (PL-RJ). Falso positivo ja conhecido do criterio de subconjunto.")
add([66], "same", "alta",
    "Pro-Reitor de Extensao da UFU (aud. 18) e Prof. Helder Eterno da Silveira (aud. 51), ambas "
    "audiencias de educacao (PIBID); nome raro.")
add([67], "same", "media",
    "Mesma pauta (motoristas de aplicativo, auds. 112 e 177); na 112 e o primeiro convidado, "
    "chamado de 'Jair'; na 177 nao ha cargo explicito.")
add([68], "same", "alta",
    "Ambos da Confederacao Nacional de Municipios, area de defesa civil (auds. 11 e 153); na aud. "
    "11 se apresenta como 'Johnny Liberato'.")
add([69], "same", "media",
    "Aud. 95: Coordenadora de Campanhas da Protecao Animal Mundial; aud. 191: Coordenadora de "
    "Programas de Food Systems (programa dessa mesma ONG); nome raro (Ishida) e mesma area.")
add([70], "same", "alta",
    "Mesmo cargo explicito nas duas audiencias: assessor parlamentar do CONASS.")
add([71], "different", "media",
    "A e pesquisadora do Parent in Science (aud. 188); B apresenta pelo Ministerio da Saude "
    "(aud. 19); nada liga os dois papeis.")
add([72], "different", "media",
    "B e especialista em monitoramento de extrema direita on-line (aud. 57); papel incompativel "
    "com o de A.")
add([73], "different", "media",
    "Ministerio da Saude (aud. 19) vs monitoramento de extrema direita (aud. 57); sem evidencia "
    "de ser a mesma pessoa.")
add(list(range(74, 87)), "different", "media",
    "LUCAS (aud. 126) e participante de plateia em audiencia sobre cannabis medicinal, com relato "
    "pessoal (mae com fibromialgia); nenhum dos 'Lucas' de nome completo tem vinculo com essa "
    "pauta ou audiencia.")
add([87], "same", "alta",
    "Mesma organizacao: Coalizao (Brasileira) pelo Fim da Violencia contra Criancas e "
    "Adolescentes (aud. 30 e envolvidos da aud. 74).")
add([88], "different", "alta",
    "B e o Deputado Lucas Ramos (PSB-PE); A e secretario-executivo de ONG.")
add([89], "same", "alta",
    "Mesmo circulo e pauta (CIEDDS/tuberculose): na aud. 114 fala pela Stop TB Brasil; na aud. "
    "123 (mesma frente, mesma presidencia) intervem como participante.")
add([90], "different", "alta",
    "A e o Deputado Marcio Correa (MDB-GO); B e Coronel, Subsecretario de Defesa Civil do RJ "
    "(aud. 107).")
add([91], "same", "alta",
    "Mesma audiencia 130 e mesmo cargo dito pela presidencia: Secretario-Executivo do Ministerio "
    "da Cultura; dois formatos de cabecalho.")
add([92], "different", "alta",
    "Dois deputados distintos; a aud. 153 registra a presenca de ambos na mesma lista ('... "
    "Marcon, Maria do Rosario, Mauricio Marcon ...').")
add([93], "different", "alta",
    "A e do MST no Ceara (aud. 186); B e chamada de 'Gaia' em pauta de loterias/FENAE (aud. 21).")
add([94], "same", "alta",
    "Mesma organizacao (Conectas Direitos Humanos) e mesma forma de participacao (remota) nas "
    "auds. 32 e 198; na aud. 32 o presidente a chama de 'Marina Barbosa'.")
add([95], "same", "alta",
    "Deputado Distrital do DF (aud. 47); na aud. 42 (pauta do DF, tarifa zero) e tratado como "
    "Deputado Max Maciel Cavalcanti.")
add([96], "same", "alta",
    "Envolvidos das duas pontas dao o mesmo cargo: Presidente da CAPES.")
add([97], "different", "alta",
    "A e o Mestre Paulao Kikongo (nome civil no cabecalho da aud. 82: Paulo Henrique da Silva); "
    "B e o Deputado Paulao (PT-AL).")
add([98], "same", "alta", "Ministra da Saude nas tres audiencias (129, 171, 199).")
add([99], "same", "alta",
    "Mesma pessoa em cargos diferentes: as audiencias em que e 'Deputado Paulo Teixeira (PT-SP)' "
    "sao de jun-set/2022 (65, 74, 83, pelo carimbo da materia) e a de Ministro do Desenvolvimento "
    "Agrario e de ago/2023 (aud. 45); ele assumiu o ministerio em jan/2023.")
add(list(range(100, 107)), "different", "media",
    "MONICA (aud. 170) e da Coordenacao de Atencao a Saude da Populacao Negra do Ministerio da "
    "Saude; nenhuma das 'Monica' de nome completo tem esse vinculo.")
add([107], "same", "alta",
    "Os envolvidos da aud. 11 dao o nome completo de A: 'Paulo Pedro de Carvalho, Coordenador da "
    "Articulacao do Semiarido Brasileiro (ASA)'; a aud. 169 traz o mesmo nome e a mesma ASA.")
add([108, 109], "different", "alta",
    "PAULO PEDRO e o coordenador da ASA; PEDRO PAULO e o Deputado (PSD-RJ), relator na aud. 79; "
    "a ordem invertida dos nomes e coincidencia (unico par equal_tokens).")
add([110], "different", "alta",
    "A e representante de torcidas organizadas (ANATORG, aud. 101); B e gestor de politicas "
    "publicas do Acre (aud. 107).")
add([111], "different", "alta", "B e Procurador do Trabalho (aud. 85).")
add([112], "different", "alta", "B e da Secretaria Nacional de Periferias (aud. 113).")
add([113], "same", "alta",
    "Na aud. 53 a presidencia diz 'O Phellip Ponce, do IBDFAM, chegou'; nome raro; reaparece em "
    "outra audiencia presidida pela mesma deputada (Erika Kokay, aud. 94).")
add([114], "same", "alta",
    "Procuradora do MPF nas duas pontas; os envolvidos da aud. 180 usam exatamente a forma curta "
    "'Raquel Branquinho'.")
add([115], "different", "alta",
    "A e o Deputado Reginaldo Lopes (PT-MG); B e Reginaldo (Lopes) Minare, diretor tecnico da "
    "CNA (aud. 81).")
add([116], "same", "alta",
    "Presidente do CNPq nas tres audiencias; envolvidos identicos ('Ricardo Galvao | Presidente "
    "do CNPq').")
add([117], "same", "alta",
    "Mesmo partido/UF (PL-RJ); 'Roberto Monteiro Pai' e o nome parlamentar; na aud. 199 o "
    "cabecalho omite o 'Pai' e o comportamento e de membro da comissao.")
add([118, 119, 120, 121] + list(range(123, 141)), "different", "media",
    "RODRIGO (aud. 148) e participante de audiencia sobre controle de javalis/caca, agradece 'ao "
    "Deputado Pollon' como convidado de plateia; nenhum dos outros 'Rodrigo' tem vinculo com essa "
    "pauta (varios sao deputados de outros estados ou dirigentes setoriais).")
add([122], "same", "alta",
    "Confirmado pelo usuario em 24/09/2026: o RODRIGO da aud. 148 e o proprio Cel. Rodrigo "
    "Bordeaux Mattos de volta ao microfone.")
add([141], "same", "alta",
    "Mesma pessoa em cargos diferentes: Deputado (PSB-SP) na aud. 70 (ago/2022) e Presidente do "
    "Ibama nas auds. 44 (mai/2023) e 183 (ago/2023; envolvidos: 'Rodrigo Agostinho | Presidente "
    "do Ibama'); assumiu o Ibama no inicio de 2023.")
add([142], "same", "alta",
    "Mesmo cargo: Diretor-Executivo do Instituto Livre Mercado (aud. 87 usa so 'Rodrigo "
    "Marinho').")
add([143], "same", "alta",
    "Na aud. 17 o presidente anuncia o nome completo ('Rodrigo Saraiva Marinho, Diretor-Executivo "
    "do Instituto Livre Mercado'), mas o cabecalho sai truncado como 'RODRIGO SARAIVA'.")
add([144], "same", "alta",
    "Os envolvidos da aud. 206 identificam Rogean Vinicius Santos Soares como 'Vinicius Soares, "
    "Presidente da ANPG'; na aud. 168 e o Presidente da ANPG.")
add([145], "different", "alta",
    "A e Secretario Nacional do Ministerio da Igualdade Racial (aud. 187); B e Procurador do "
    "Trabalho (CONALIS, aud. 97).")
add([146], "different", "alta",
    "Dois deputados distintos: Vicentinho (PT-SP) e Vicentinho Junior (PP-TO).")
add([147], "same", "alta",
    "Mesmo cargo nas duas audiencias: Diretor de Energia Eletrica da ABRACE.")
add([148], "different", "alta",
    "A e chamado de 'professor' em audiencia de motoristas de aplicativo (aud. 112); B e "
    "presidente de tribunal no Parana (aud. 155).")
add([149], "uncertain", "media",
    "Ambos em audiencias de motoristas de aplicativo; A e tratado como 'professor' (aud. 112); B "
    "e 'Wellington Mais Chegado', motorista de Manaus (aud. 177); provavelmente distintos, mas a "
    "pauta comum impede descartar.")
add([150, 151, 152], "same", "alta",
    "Secretario Nacional de Protecao e Defesa Civil nas tres chaves (auds. 25, 109, 118); "
    "envolvidos e apresentacoes dao o mesmo cargo.")

EXTRA_FINDINGS = [
    (
        "Par que o criterio NAO capturou: DUARTE JR. (Bloco/PSB-MA) vs DUARTE GONCALVES JR "
        "(Bloco/PODE-MG). 'JR.' com ponto nao e igual a 'JR' sem ponto apos a normalizacao, entao "
        "o par ficou fora do ambiguous_names.json. Resolvido em 24/09/2026 pela transitividade do "
        "grupo do par 52."
    ),
    (
        "A chave ALEXANDRE SAMPAIO (mescla exata) junta duas pessoas distintas: na aud. 14 e o "
        "presidente da associacao de vitimas da mineracao em Maceio, na aud. 38 e diretor da CNC. "
        "Ou seja, alem de pares nao mesclados, existe pelo menos um falso positivo da propria "
        "mescla por nome exato (detectado via par 6). Resolvido em 24/09/2026 via REASSIGNMENTS."
    ),
]

REASSIGNMENTS = [
    {"key": "ALEXANDRE SAMPAIO", "hearing_id": 38, "canonical": "ALEXANDRE SAMPAIO DE ABREU"},
]


def load_pairs() -> list[dict]:
    return json.loads(AMBIGUOUS_PATH.read_text())["pairs"]


def merge_groups(pairs: list[dict]) -> list[dict]:
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    hearing_ids: dict[str, set[int]] = {}
    for index, pair in enumerate(pairs):
        for side in ("a", "b"):
            hearing_ids[pair[side]["key"]] = set(pair[side]["hearing_ids"])
        if VERDICTS[index]["verdict"] == "same":
            parent[find(pair["a"]["key"])] = find(pair["b"]["key"])
    groups: dict[str, list[str]] = {}
    for key in parent:
        groups.setdefault(find(key), []).append(key)
    result = []
    for members in groups.values():
        if len(members) < 2:
            continue
        merged = sorted(set().union(*(hearing_ids[m] for m in members)))
        result.append(
            {
                "keys": sorted(members),
                "hearings": merged,
                "was_multi": any(len(hearing_ids[m]) >= 2 for m in members),
                "becomes_multi": len(merged) >= 2,
            }
        )
    return sorted(result, key=lambda group: group["keys"][0])


def alias_map(pairs: list[dict], groups: list[dict]) -> dict[str, str]:
    turns: dict[str, int] = {}
    for pair in pairs:
        for side in ("a", "b"):
            turns[pair[side]["key"]] = pair[side]["turns"]
    mapping: dict[str, str] = {}
    for group in groups:
        canonical = max(group["keys"], key=lambda key: (turns[key], len(key)))
        for key in group["keys"]:
            if key != canonical:
                mapping[key] = canonical
    return mapping


def unreviewed_current_pairs(pairs: list[dict], mapping: dict[str, str]) -> list[dict] | None:
    if not CURRENT_PAIRS_PATH.exists():
        return None
    current = json.loads(CURRENT_PAIRS_PATH.read_text())["pairs"]
    reviewed = {
        frozenset(
            (mapping.get(pair["a"]["key"], pair["a"]["key"]),
             mapping.get(pair["b"]["key"], pair["b"]["key"]))
        )
        for pair in pairs
    }
    return [
        pair for pair in current
        if frozenset((pair["a"]["key"], pair["b"]["key"])) not in reviewed
    ]


def merge_config_toml(pairs: list[dict], groups: list[dict]) -> str:
    mapping = alias_map(pairs, groups)
    canonicals: dict[str, list[str]] = {}
    for alias, canonical in mapping.items():
        canonicals.setdefault(canonical, []).append(alias)
    lines: list[str] = []
    for canonical in sorted(canonicals, key=lambda key: min([key] + canonicals[key])):
        alias_list = ", ".join(f'"{alias}"' for alias in sorted(canonicals[canonical]))
        lines += ["[[merges.groups]]", f'canonical = "{canonical}"', f"aliases = [{alias_list}]", ""]
    for entry in REASSIGNMENTS:
        lines += [
            "[[merges.reassignments]]",
            f'key = "{entry["key"]}"',
            f"hearing_id = {entry['hearing_id']}",
            f'canonical = "{entry["canonical"]}"',
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    pairs = load_pairs()
    assert len(pairs) == len(VERDICTS) == 153, (len(pairs), len(VERDICTS))
    counts: dict[str, int] = {}
    for entry in VERDICTS.values():
        label = f"{entry['verdict']}/{entry['confidence']}"
        counts[label] = counts.get(label, 0) + 1
    print("resumo:", dict(sorted(counts.items())))
    groups = merge_groups(pairs)
    new_multi = [g for g in groups if g["becomes_multi"] and not g["was_multi"]]
    print(f"grupos de mescla: {len(groups)}")
    print(f"  viram ator multi-audiencia novo: {len(new_multi)}")
    for group in groups:
        flag = " [NOVO MULTI]" if group["becomes_multi"] and not group["was_multi"] else ""
        print(f"  {' + '.join(group['keys'])} -> auds. {group['hearings']}{flag}")
    print("\nreatribuicoes por audiencia:")
    for entry in REASSIGNMENTS:
        print(f"  {entry['key']} (aud. {entry['hearing_id']}) -> {entry['canonical']}")
    print("\nincertos (precisam de olho humano):")
    for index, pair in enumerate(pairs):
        if VERDICTS[index]["verdict"] == "uncertain":
            print(f"  [{index}] {pair['a']['key']} vs {pair['b']['key']}")
            print(f"      {VERDICTS[index]['evidence']}")
    print("\nachados extras:")
    for finding in EXTRA_FINDINGS:
        print(f"  - {finding}")
    fresh = unreviewed_current_pairs(pairs, alias_map(pairs, groups))
    if fresh is None:
        print("\npares atuais: arquivo regenerado ainda nao existe")
    else:
        print(f"\npares no ambiguous_names.json atual sem cobertura desta revisao: {len(fresh)}")
        for pair in fresh:
            print(f"  {pair['a']['key']} vs {pair['b']['key']}")
    print("\ntrecho TOML para configs/hearing_actors.toml:\n")
    print(merge_config_toml(pairs, groups))


if __name__ == "__main__":
    main()
