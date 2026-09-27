"""Model annotation of the UDV validation sample human_validation_v1_udv_v1.

Three independent passes by Claude Fable 5.1 subagents on 2026-09-26, each blind to the strata key
and to the other passes, reading only the blank sheet, challenge/annotation_guide.md and the
hearing transcripts. These labels are a model annotation (LLM judge), not the human validation:
they must never be copied into the sample directory's annotation.csv.

Usage (from the repository root):
    uv run --project challenge python claude_udv_model_annotation.py <output_dir>
writes <output_dir>/annotation.csv with the majority label (ties of three distinct labels take the
middle label; none occurred), plus annotation_key.json and sample_report.json copied from the sample,
so that challenge/utils/precision_report.py can be run on <output_dir> with --without-reannotation.
"""

import collections
import csv
import shutil
import sys
from pathlib import Path

SAMPLE_DIR = Path("challenge/artifacts/validation/human_validation_v1_udv_v1")
ORDER = {"incorreta": 0, "parcial": 1, "correta": 2}
PASSES = {
 "pass_1": {
  "A001": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A002": [
   "parcial",
   "sim",
   "Trecho fala de prefeitos; as críticas (PRF, Lesa Pátria, veto) estão em outros pontos; melhor: \"Eu gostaria só, em nome da população brasileira, que a\"."
  ],
  "A003": [
   "parcial",
   "sim",
   "Trecho é a proposta, não o impacto de 50% nem o modelo de mensalidade; melhor: \"Para a empresa X, que cobra 10%, isso significa 50%\"."
  ],
  "A004": [
   "parcial",
   "sim",
   "Trecho cobre só os venezuelanos; a alteração da legislação vem parágrafos depois; melhor: \"Mas estamos trabalhando, hoje, na alteração da nossa legislação para\"."
  ],
  "A005": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A006": [
   "parcial",
   "nao",
   "Trecho sustenta \"única e última\"; os números (4 afastados, 10 PADs, 29 de 276) estão em parágrafo posterior."
  ],
  "A007": [
   "parcial",
   "nao",
   "A metáfora do respirador está nas duas frases seguintes, não no trecho."
  ],
  "A008": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A009": [
   "correta",
   "nao",
   "Citação está no trecho; retirada da urgência e 45 dias constam no mesmo turno."
  ],
  "A010": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A011": [
   "parcial",
   "sim",
   "Trecho cobre só o banimento de contas, não a ausência de base legal; melhor: \"E com qual base legal? Nenhuma. Não há uma lei, não\"."
  ],
  "A012": [
   "parcial",
   "sim",
   "Trecho fala dos próprios dados, não da parceria; melhor, no turno 18: \"Para cuidarmos das nossas crianças, tem que haver uma parceria\"."
  ],
  "A013": [
   "parcial",
   "sim",
   "Trecho diz \"se mantenha como empresa importante\", não controle integral; melhor: \"Então, o nosso pedido é que se suspenda essa aprovação\"."
  ],
  "A014": [
   "parcial",
   "sim",
   "Trecho cobre só parte; melhor, no turno 14: \"Ela disse que a nossa rota é mais competitiva do\"."
  ],
  "A015": [
   "parcial",
   "nao",
   "Trecho só traz \"não querem privilégio\"; o resto da citação vem em seguida."
  ],
  "A016": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A017": [
   "parcial",
   "sim",
   "Trecho fala em destinar recurso, sem estados e municípios; melhor, no turno 22: \"é fundamental que tenhamos financiamentos específicos, inclusive, com garantia de\"."
  ],
  "A018": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A019": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A020": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A021": [
   "correta",
   "nao",
   "Conteúdo central no trecho; duas frases curtas do fim da citação estão adjacentes."
  ],
  "A022": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A023": [
   "parcial",
   "sim",
   "Trecho sustenta só parte da afirmação; a ausência de previsão legal está em outro ponto; melhor: \"Srs. Parlamentares, concidadãos, em nenhum lugar da lei brasileira está\"."
  ],
  "A024": [
   "correta",
   "nao",
   "Fragmento se resolve pelo contexto imediato e sustenta a afirmação."
  ],
  "A025": [
   "correta",
   "nao",
   "O \"um deles\" se resolve pelo contexto imediato (\"vários consensos\")."
  ],
  "A026": [
   "falou",
   "",
   "Consta como O SR. MARLISON SOARES CUNHA (turno 10), com o conteúdo atribuído (desburocratização dos repasses, 10 ou 15 dias)."
  ],
  "A027": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A028": [
   "correta",
   "nao",
   "Conteúdo central no trecho; \"esse custo não é barato\" é adjacente."
  ],
  "A029": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A030": [
   "parcial",
   "sim",
   "Trecho não menciona o El Niño; a comparação está nas frases seguintes; melhor: \"O El Niño fica nessa faixa do Oceano Pacífico, na região\"."
  ],
  "A031": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A032": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A033": [
   "parcial",
   "nao",
   "Trecho trata da unificação de informações geológicas, hidrológicas e meteorológicas; a passagem sobre radar também é parcial."
  ],
  "A034": [
   "correta",
   "nao",
   "Conteúdo central no trecho; a frase sobre financiamento é adjacente."
  ],
  "A035": [
   "correta",
   "nao",
   "Citação está no trecho; a parte sobre composição e renovável é redação editorial, não fala do participante."
  ],
  "A036": [
   "correta",
   "nao",
   "Conteúdo central no trecho; banimento vitalício é adjacente."
  ],
  "A037": [
   "correta",
   "nao",
   "Trecho apresenta a proteção aos vulneráveis como o grande desafio; \"absoluta\" é ênfase editorial."
  ],
  "A038": [
   "parcial",
   "sim",
   "Trecho traz o tema, mas a crítica principal vem em outra frase; melhor: \"Dizer-se inclusivo porque simplesmente está recebendo uma pessoa na sua\"."
  ],
  "A039": [
   "parcial",
   "nao",
   "Trecho é a primeira frase de citação longa; 1 milhão e 70% estão nas frases seguintes."
  ],
  "A040": [
   "parcial",
   "sim",
   "Trecho é sobre Goiás; a citação vem de parágrafo anterior; melhor: \"Precisamos sensibilizar o Ministério para investir mais, a fim de\"."
  ],
  "A041": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A042": [
   "correta",
   "nao",
   "PL 970 nas frases adjacentes; \"estudo técnico prévio\" não consta, mas audiências públicas sim."
  ],
  "A043": [
   "incorreta",
   "sim",
   "Trecho é procedimental (\"se ater só à questão das armas\"); melhor, no turno 25: \"No dia 1º de janeiro de 2023, início do Governo\"."
  ],
  "A044": [
   "parcial",
   "sim",
   "Trecho cobre só parte da afirmação; melhor: \"Acontece que a voracidade do setor financeiro criou a possibilidade\"."
  ],
  "A045": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A046": [
   "parcial",
   "sim",
   "Trecho trata dos prejuízos no Centro e Sudeste, não da evapotranspiração; melhor: \"Na medida em que se desmata, reduz-se a chuva e\"."
  ],
  "A047": [
   "parcial",
   "sim",
   "Trecho do turno 86 sustenta só \"discutidos com os técnicos\"; melhor, no turno 38: \"Nós não podemos tampar os olhos e dizer que os\"."
  ],
  "A048": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A049": [
   "parcial",
   "nao",
   "Trecho sustenta a cobertura às famílias; a citação sobre motoristas mortos está nas frases anteriores e a categoria em parágrafo anterior."
  ],
  "A050": [
   "falou",
   "",
   "Consta como O SR. WOLNEI WOLFF BARREIROS (turno 21), com quase 2 mil Municípios em emergência ou calamidade."
  ],
  "A051": [
   "correta",
   "nao",
   "Citação está no trecho; 71.867 está no parágrafo anterior do mesmo turno."
  ],
  "A052": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A053": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A054": [
   "parcial",
   "nao",
   "Trecho traz a frase central da citação; a taxa variável e \"ganhando menos\" estão em parágrafos posteriores."
  ],
  "A055": [
   "parcial",
   "nao",
   "Trecho traz uma frase da citação; as outras duas são adjacentes."
  ],
  "A056": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A057": [
   "correta",
   "nao",
   "Citação está no trecho; o restante em frases adjacentes."
  ],
  "A058": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A059": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A060": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A061": [
   "falou",
   "",
   "Fala em LIBRAS (turnos 11 e 33) vocalizada pela intérprete Adriana Lopes (turnos 12 e 34), com 63.106 alunos, 7 mil, 11 milhões e língua de instrução."
  ],
  "A062": [
   "parcial",
   "nao",
   "Trecho só nomeia o Twitter Files Brasil; nada sobre a publicação dos e-mails."
  ],
  "A063": [
   "correta",
   "nao",
   "O \"isso\" do trecho se resolve no contexto imediato e sustenta a afirmação."
  ],
  "A064": [
   "parcial",
   "sim",
   "Trecho cobre só parte da afirmação; melhor: \"O Juiz Alexandre de Moraes fica muito preocupado em entender\"."
  ],
  "A065": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A066": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A067": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A068": [
   "parcial",
   "sim",
   "Trecho sustenta só a primeira frase da citação; melhor: \"Então, eu vou fazer um novo requerimento, que sugiro também\"."
  ],
  "A069": [
   "parcial",
   "nao",
   "Trecho só diz \"não é simplesmente do Governo\"; o resto está nas frases seguintes."
  ],
  "A070": [
   "correta",
   "nao",
   "Primeira frase da citação está no trecho; \"Ninguém escolhe morar em área de risco\" é a frase seguinte."
  ],
  "A071": [
   "parcial",
   "sim",
   "Trecho é genérico; a conexão à GD está no parágrafo seguinte; melhor: \"Elas não obedecem às normas, não obedecem à lei, não\"."
  ],
  "A072": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A073": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A074": [
   "parcial",
   "sim",
   "Trecho é fragmento sobre intersetorialidade; a integração de adaptação e mitigação vem no parágrafo seguinte; melhor: \"E, agora, eu acho muito louvável que o Governo Federal\"."
  ],
  "A075": [
   "parcial",
   "nao",
   "Trecho cobre só parte; o ponto sobre base e oposição está antes no mesmo turno 9 e o resto nas frases vizinhas."
  ],
  "A076": [
   "correta",
   "nao",
   "1,1 bilhão e menos de 1 real estão no trecho."
  ],
  "A077": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A078": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A079": [
   "parcial",
   "nao",
   "Trecho cobre a primeira metade da citação; a segunda (\"A luta é ainda maior\") é a frase seguinte."
  ],
  "A080": [
   "correta",
   "nao",
   "Trecho reproduz a citação quase literalmente; \"vitória\" é a frase anterior."
  ],
  "A081": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A082": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A083": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A084": [
   "parcial",
   "nao",
   "Trecho traz só a razão (almoço e jantar); menor dedicação e ganho mais baixo estão nas frases vizinhas."
  ],
  "A085": [
   "falou",
   "",
   "Consta como A SRA. LUANE CARVALHO COSTA (turno 25), mesma coordenação e mesmo conteúdo; o sobrenome coincide e o primeiro nome difere."
  ],
  "A086": [
   "correta",
   "nao",
   "Ofícios ao Planejamento e à Fazenda e visita a Tebet em agendamento constam no trecho e contexto imediato."
  ],
  "A087": [
   "incorreta",
   "sim",
   "Trecho é encerramento procedimental; melhor, no turno 55: \"No dia 27 de março, após a aprovação do requerimento\"; o teor da resposta do ministério não está transcrito."
  ],
  "A088": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A089": [
   "parcial",
   "nao",
   "Trecho traz a primeira oração da citação; \"transforma passivo em ativo\" é a frase seguinte e a vocação está no turno 1."
  ],
  "A090": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A091": [
   "correta",
   "nao",
   "Tsunami e 100 a 150 bilhões estão no trecho; \"Caixa fora\" é a frase seguinte."
  ],
  "A092": [
   "correta",
   "nao",
   "Demanda orçamentária está no trecho; a primeira frase da citação é a anterior."
  ],
  "A093": [
   "parcial",
   "nao",
   "Lista de três itens; trecho tem só a quebra dos protocolos, o relaxamento vem antes e as revistas depois."
  ],
  "A094": [
   "parcial",
   "sim",
   "Trecho não traz os números de 23% e 4%; melhor: \"Então, é muito importante que saibamos que nós estamos impedindo\"."
  ],
  "A095": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A096": [
   "correta",
   "nao",
   "Trecho traz a posição central; a menção ao consumidor não poder pagar é a frase seguinte."
  ],
  "A097": [
   "correta",
   "nao",
   "Conteúdo central no trecho; 8 reais nas frases adjacentes."
  ],
  "A098": [
   "correta",
   "sim",
   "Trecho sustenta a afirmação, mas há passagem mais direta; melhor: \"Com a ferramenta tecnológica do eSocial e o Fundo de Garantia Digital\"."
  ],
  "A099": [
   "parcial",
   "nao",
   "Trecho traz o princípio geral; a posição em si está na frase seguinte."
  ],
  "A100": [
   "parcial",
   "sim",
   "Trecho é sobre carreira para a Defesa Civil; melhor: \"Eu trabalhei fortemente pelo financiamento, para colocar recursos no Fundo\"."
  ],
  "A101": [
   "parcial",
   "sim",
   "Trecho cobre só parte da proposta; melhor: \"Nós precisamos resgatar e fortalecer o Fundo de Garantia, para\"."
  ],
  "A102": [
   "parcial",
   "nao",
   "IRA e EUA estão no trecho; Green Deal e UE nas frases seguintes."
  ],
  "A103": [
   "parcial",
   "nao",
   "Trecho traz só a primeira frase da citação; financiamento perene e fluxo contínuo estão nas frases vizinhas."
  ],
  "A104": [
   "parcial",
   "sim",
   "Trecho cobre só parte; melhor: \"Ao contrário do que vem sendo anunciado, o setor não\"."
  ],
  "A105": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A106": [
   "parcial",
   "nao",
   "Trecho cobre só parte; o ENEM está na frase anterior e a lista de estados na seguinte."
  ],
  "A107": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A108": [
   "parcial",
   "nao",
   "Trecho diz 43 milhões e a afirmação 43 bilhões; os 60% batem."
  ],
  "A109": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A110": [
   "parcial",
   "nao",
   "2 bilhões está no trecho; 250 milhões é a frase seguinte."
  ],
  "A111": [
   "falou",
   "",
   "Fala em LIBRAS (turnos 8 e 30) vocalizada por intérprete (turnos 9 e 31), incluindo a banca de avaliação com professores surdos."
  ],
  "A112": [
   "correta",
   "nao",
   "6 e 6 meses estão no trecho; \"últimos três anos\" é redação editorial."
  ],
  "A113": [
   "parcial",
   "sim",
   "Trecho traz 51% do subconjunto de alienação; o 66,3% está em outro ponto; melhor: \"Na próxima página, estou ilustrando para os senhores o total\"."
  ],
  "A114": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A115": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A116": [
   "correta",
   "nao",
   "Posição contra o MEI está no trecho; a razão (sem sustentabilidade, empregador não contribui) é adjacente."
  ],
  "A117": [
   "parcial",
   "sim",
   "A frase anterior traz o ponto central (\"não contra alguma facção\"); melhor: \"A guerra em curso, senhores, é contra a Palestina, contra\"."
  ],
  "A118": [
   "parcial",
   "sim",
   "Trecho sustenta só a frase sobre revogação; melhor: \"Quando passamos em sala de aula, a galera do primeiro\"."
  ],
  "A119": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A120": [
   "parcial",
   "sim",
   "Trecho é um ponto factual do PL; a defesa está em outro ponto do turno 21; melhor: \"O PL traz alguns pontos principais, como a criação de\"."
  ],
  "A121": [
   "parcial",
   "sim",
   "Trecho é fragmento; melhor: \"Eu, pessoalmente, não faço questão de ser por hora ou\"."
  ],
  "A122": [
   "parcial",
   "nao",
   "Trecho sustenta o prejuízo ao próprio app; concentração de mercado vem depois no turno e 99/Uber não são nomeadas."
  ],
  "A123": [
   "parcial",
   "sim",
   "Trecho diz que a extensão deve incluir obras, não que devem ficar no contrato atual; melhor: \"Nós acreditamos que essa obra poderia ser contemplada não necessariamente\"."
  ],
  "A124": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ],
  "A125": [
   "parcial",
   "sim",
   "Trecho cobre só as crianças; melhor: \"E é claro também que mães, gestantes e bebês recém-nascidos\"."
  ],
  "A126": [
   "falou",
   "",
   "Consta como O SR. THIAGO XAVIER (turnos 19, 21 e 23), com o conteúdo atribuído (90% de redução, critérios não explicitados)."
  ],
  "A127": [
   "correta",
   "nao",
   "Trecho sustenta o conteúdo principal da afirmação e a posição atribuída."
  ]
 },
 "pass_2": {
  "A001": [
   "correta",
   "nao",
   "Trecho diz que o Governo tem perdido prazos e o contexto cita o prazo vencido da reforma do IR."
  ],
  "A002": [
   "incorreta",
   "sim",
   "Trecho critica Prefeitos, não o Governo. Melhor: Nesses últimos 4 anos do Governo Bolsonaro, foi criado dentro da"
  ],
  "A003": [
   "parcial",
   "sim",
   "Trecho é a proposta de base no faturamento, sem a metade do faturamento nem a mensalidade. Melhor: Vejam este papel. Nós fizemos uma simulação envolvendo as empresas X"
  ],
  "A004": [
   "parcial",
   "sim",
   "Trecho cobre só os venezuelanos. Melhor: Mas estamos trabalhando, hoje, na alteração da nossa legislação para ampliar"
  ],
  "A005": [
   "correta",
   "nao",
   "Trecho diz que orienta portos e cria incentivos para terminais, inclusive privados, reduzirem a pegada."
  ],
  "A006": [
   "correta",
   "nao",
   "Trecho sustenta fuga excepcional, única e última. Os números de servidores citados na afirmação não aparecem na fala."
  ],
  "A007": [
   "parcial",
   "nao",
   "Trecho cobre só a retomada do setor. A metáfora do respirador e o PERSE vêm nas frases seguintes do mesmo turno."
  ],
  "A008": [
   "correta",
   "nao",
   "Trecho afirma que a volta dos presos a Mossoró demonstra confiança na segurança do presídio."
  ],
  "A009": [
   "parcial",
   "sim",
   "Trecho cobre só a citação sobre ganhos reais, e a retirada da urgência está na frase seguinte. Melhor: eu vou defender também, como Presidente da Comissão de Legislação Participativa"
  ],
  "A010": [
   "correta",
   "nao",
   "Trecho traz a frase central da citação. Os exemplos seguintes estão na sequência do turno."
  ],
  "A011": [
   "parcial",
   "sim",
   "Trecho cobre o banimento de contas, não a falta de base legal nem os parlamentares. Melhor: O Ministro Moraes simplesmente ignorou essa data de expiração e continua"
  ],
  "A012": [
   "parcial",
   "sim",
   "Trecho fala do cuidado com os dados que nós publicamos, não do controle sobre os filhos. Melhor: Para cuidarmos das nossas crianças, tem que haver uma parceria entre"
  ],
  "A013": [
   "parcial",
   "nao",
   "Trecho defende que a Caixa se mantenha importante nas loterias, sem falar em controle integral. Não há passagem mais forte."
  ],
  "A014": [
   "parcial",
   "sim",
   "Trecho só diz que é entusiasta dos combustíveis da transição. Melhor: Ela disse que a nossa rota é mais competitiva do que a dos"
  ],
  "A015": [
   "parcial",
   "nao",
   "Trecho traz só o início da citação. O restante e a crítica à responsabilização estão nas frases vizinhas."
  ],
  "A016": [
   "correta",
   "nao",
   "Trecho corresponde à citação sobre neuropediatra e equipe multidisciplinar."
  ],
  "A017": [
   "parcial",
   "sim",
   "Trecho fala em destinar recurso sem citar Estados e Municípios. Melhor: A segunda questão se refere à questão de financiamento. Eu também quero"
  ],
  "A018": [
   "correta",
   "nao",
   "Trecho diz que a legislação prevê direitos, mas nas cidades as pessoas com TEA não conseguem usufruir deles."
  ],
  "A019": [
   "correta",
   "nao",
   "Trecho corresponde à citação sobre audiências e pesquisas em que os trabalhadores foram ouvidos."
  ],
  "A020": [
   "correta",
   "nao",
   "Trecho corresponde à citação com os percentuais de 16% e 9%."
  ],
  "A021": [
   "correta",
   "nao",
   "Trecho traz a frase principal da citação. A frase final está na sequência imediata."
  ],
  "A022": [
   "correta",
   "nao",
   "Trecho contém a citação inteira."
  ],
  "A023": [
   "parcial",
   "sim",
   "Trecho cobre só a segunda citação. Melhor: Srs. Parlamentares, concidadãos, em nenhum lugar da lei brasileira está escrito"
  ],
  "A024": [
   "correta",
   "nao",
   "Trecho, com o início da frase no contexto, registra o convite enviado ao Ministro Haddad para a audiência sobre o PERSE."
  ],
  "A025": [
   "correta",
   "nao",
   "Trecho, lido com a frase anterior sobre consensos construídos na Casa, sustenta a afirmação."
  ],
  "A026": [
   "falou",
   "",
   "Fala no turno 10 como O SR. MARLISON SOARES CUNHA, com uma letra a menos no nome, e trata da desburocratização dos repasses."
  ],
  "A027": [
   "correta",
   "nao",
   "Trecho parabeniza o Ministério por pensar a segurança com base nos direitos humanos sem populismo penal."
  ],
  "A028": [
   "correta",
   "nao",
   "Trecho corresponde à citação. A frase sobre o custo não ser barato vem em seguida."
  ],
  "A029": [
   "correta",
   "nao",
   "Trecho traz a pergunta central da citação. A resposta e a frase das bandeiras vêm na sequência."
  ],
  "A030": [
   "parcial",
   "sim",
   "Trecho não explica a distinção com o El Niño. Melhor: O El Niño fica nessa faixa do Oceano Pacífico, na região do"
  ],
  "A031": [
   "correta",
   "nao",
   "Trecho anuncia as soluções inovadoras de transferência de riscos. Os exemplos de vários países vêm na frase seguinte."
  ],
  "A032": [
   "correta",
   "nao",
   "Trecho, com a PEC 504/2010 nomeada na frase anterior, sustenta a posição favorável."
  ],
  "A033": [
   "parcial",
   "nao",
   "Trecho fala em unificar informações de meteorologia, sem afirmar que a previsão do tempo é essencial. Não há passagem melhor."
  ],
  "A034": [
   "correta",
   "nao",
   "Trecho corresponde à primeira frase da citação. A frase sobre financiamento vem em seguida."
  ],
  "A035": [
   "correta",
   "nao",
   "Trecho corresponde à citação sobre gás natural e diesel importado. A parte sobre ser renovável não aparece na fala."
  ],
  "A036": [
   "parcial",
   "nao",
   "Trecho cobre a gradação da punição. O banimento vitalício de reincidentes está nas frases seguintes."
  ],
  "A037": [
   "correta",
   "nao",
   "Trecho apresenta a proteção aos vulneráveis como o grande desafio, sem a expressão prioridade absoluta."
  ],
  "A038": [
   "parcial",
   "sim",
   "Trecho cobre o planejamento individualizado, não a citação. Melhor: Dizer-se inclusivo porque simplesmente está recebendo uma pessoa na sua escola"
  ],
  "A039": [
   "parcial",
   "nao",
   "Trecho traz só a primeira frase da citação. O restante vem nas frases seguintes do mesmo turno."
  ],
  "A040": [
   "parcial",
   "sim",
   "Trecho é o exemplo de Goiás, com a mesma posição, mas sem a citação. Melhor: Precisamos sensibilizar o Ministério para investir mais, a fim de que"
  ],
  "A041": [
   "correta",
   "nao",
   "Trecho corresponde à citação sobre os grupos paralelos."
  ],
  "A042": [
   "parcial",
   "nao",
   "Trecho lista exigências da lei de rodovias. O PL 970/24 está na frase seguinte e o estudo técnico prévio não aparece."
  ],
  "A043": [
   "incorreta",
   "sim",
   "Trecho é só pedido de ordem ao orador. Melhor: No dia 1º de janeiro de 2023, início do Governo Lula, um decreto"
  ],
  "A044": [
   "parcial",
   "sim",
   "Trecho não traz as 28 operações. Melhor: Outro ponto é que a antecipação do saque deturpou a finalidade"
  ],
  "A045": [
   "correta",
   "nao",
   "Trecho traz os dados de 2012 a 2021 com o Brasil no fim da lista. A citação vem logo depois."
  ],
  "A046": [
   "parcial",
   "sim",
   "Trecho fala dos prejuízos no Sudeste sem o mecanismo da umidade. Melhor: E, como o lado mais desmatado da Amazônia é bem nesse início"
  ],
  "A047": [
   "parcial",
   "sim",
   "Trecho cobre só a discussão técnica. Melhor: Nós não podemos tampar os olhos e dizer que os CACs não"
  ],
  "A048": [
   "correta",
   "nao",
   "Trecho corresponde à citação sobre não demorar com a decisão."
  ],
  "A049": [
   "parcial",
   "sim",
   "Trecho cobre a cobertura às famílias, não a citação nem a categoria. Melhor: Em relação à Previdência Social, como bem lembrou o Abel, muitos motoristas"
  ],
  "A050": [
   "falou",
   "",
   "Fala como O SR. WOLNEI WOLFF BARREIROS e cita os quase 2 mil Municípios em emergência."
  ],
  "A051": [
   "parcial",
   "nao",
   "Trecho cobre a primeira frase da citação. O número 71.867 e a frase final estão em outras partes do mesmo turno."
  ],
  "A052": [
   "correta",
   "nao",
   "Trecho corresponde à citação sobre debêntures de curto prazo."
  ],
  "A053": [
   "correta",
   "nao",
   "Trecho corresponde à citação."
  ],
  "A054": [
   "parcial",
   "nao",
   "Trecho cobre a queda da qualidade. A taxa variável e os valores pagos pelos usuários estão em outras frases do turno."
  ],
  "A055": [
   "parcial",
   "nao",
   "Trecho traz só a primeira frase da citação. As outras duas vêm em seguida."
  ],
  "A056": [
   "correta",
   "nao",
   "Trecho corresponde aos dados citados."
  ],
  "A057": [
   "parcial",
   "sim",
   "Trecho cobre só a consulta à PGFN. Melhor: Estaria sendo estudada uma transferência, mas sem a possibilidade de privatização"
  ],
  "A058": [
   "correta",
   "nao",
   "Trecho registra a revogação, no primeiro dia, do decreto que trazia de volta escolas segregadas."
  ],
  "A059": [
   "correta",
   "nao",
   "Trecho corresponde à citação."
  ],
  "A060": [
   "correta",
   "nao",
   "Trecho corresponde à citação sobre a retirada de 120 a 244 bilhões."
  ],
  "A061": [
   "falou",
   "",
   "O cabeçalho dele registra só manifestação em LIBRAS, mas a intérprete Adriana Lopes traduz a fala dele nos turnos 12 e 34, com o censo de 63.106 alunos."
  ],
  "A062": [
   "parcial",
   "nao",
   "Trecho só menciona o Twitter Files Brasil. A divulgação dos e-mails não é descrita em nenhuma fala dele."
  ],
  "A063": [
   "parcial",
   "sim",
   "Trecho é só a citação, sem o Marco Civil nem as decisões genéricas. Melhor: O Marco Civil da Internet é muito claro, salvo engano, no"
  ],
  "A064": [
   "parcial",
   "sim",
   "Trecho não cita Moraes nem o bloqueio de contas. Melhor: O Juiz Alexandre de Moraes fica muito preocupado em entender qualquer"
  ],
  "A065": [
   "correta",
   "nao",
   "Trecho traz a frase principal da citação. A frase sobre o brilho dos esportes vem em seguida."
  ],
  "A066": [
   "correta",
   "nao",
   "Trecho reproduz a citação, mas atribui o efeito aos grupos de trabalho paralelos, que no contexto são o trabalho das frentes."
  ],
  "A067": [
   "correta",
   "nao",
   "Trecho traz a falta de fundamentação legal e o ato discriminatório. O assédio moral vem na frase seguinte."
  ],
  "A068": [
   "parcial",
   "sim",
   "Trecho cobre só a primeira frase da citação. Melhor: Então, eu vou fazer um novo requerimento, que sugiro também aos"
  ],
  "A069": [
   "parcial",
   "nao",
   "Trecho traz só a primeira frase da citação. O restante vem nas frases seguintes."
  ],
  "A070": [
   "correta",
   "nao",
   "Trecho corresponde à citação. A última frase vem em seguida."
  ],
  "A071": [
   "parcial",
   "sim",
   "Trecho fala só em não cumprir a lei. Melhor: Elas não obedecem às normas, não obedecem à lei, não obedecem"
  ],
  "A072": [
   "correta",
   "nao",
   "Trecho traz os 3 reais por mês e a indenização de 15 mil reais."
  ],
  "A073": [
   "correta",
   "nao",
   "Trecho corresponde à citação."
  ],
  "A074": [
   "parcial",
   "sim",
   "Trecho é só uma referência ao Plano Clima. Melhor: E, agora, eu acho muito louvável que o Governo Federal tenha"
  ],
  "A075": [
   "parcial",
   "sim",
   "Trecho cobre só a primeira frase da citação. Melhor: Portanto, não há conflitos. Eu sou Deputado do PL, mas, ao"
  ],
  "A076": [
   "correta",
   "nao",
   "Trecho traz o investimento de 1,1 bilhão absorvível no contrato atual com menos de 1 real de tarifa."
  ],
  "A077": [
   "correta",
   "nao",
   "Trecho corresponde à citação."
  ],
  "A078": [
   "correta",
   "nao",
   "Trecho traz a frase principal da citação. A frase sobre pessoas trancadas na rodovia vem em seguida."
  ],
  "A079": [
   "parcial",
   "nao",
   "Trecho cobre só a primeira metade da citação. A segunda vem na frase seguinte."
  ],
  "A080": [
   "correta",
   "nao",
   "Trecho corresponde à citação. A vitória está na frase anterior."
  ],
  "A081": [
   "correta",
   "nao",
   "Trecho diz que o projeto dos túneis estava em pauta desde 2010."
  ],
  "A082": [
   "parcial",
   "nao",
   "Trecho cobre a citação sobre o fechamento dos clubes. A restrição por calibre é tratada em outra parte do turno."
  ],
  "A083": [
   "correta",
   "nao",
   "Trecho corresponde à primeira citação. A segunda vem em seguida."
  ],
  "A084": [
   "parcial",
   "nao",
   "Trecho cobre só o horário de almoço e jantar. O menor tempo de trabalho e o ganho mais baixo estão nas frases vizinhas."
  ],
  "A085": [
   "falou",
   "",
   "Consta como A SRA. LUANE CARVALHO COSTA, turno 25, da mesma Coordenação-Geral do Ministério da Saúde, e a fala traz a atualização da política de 2002 e os grupos condutores. O primeiro nome diverge."
  ],
  "A086": [
   "correta",
   "nao",
   "Trecho traz os ofícios aos dois ministérios e a ida ao gabinete da Ministra do Planejamento. Com a Fazenda cita só o ofício."
  ],
  "A087": [
   "incorreta",
   "sim",
   "Trecho é encerramento da reunião. Melhor: No dia 27 de março, após a aprovação do requerimento de realização"
  ],
  "A088": [
   "correta",
   "nao",
   "Trecho corresponde à citação sobre os túneis deixados fora do contrato."
  ],
  "A089": [
   "parcial",
   "nao",
   "Trecho cobre só a primeira metade da citação. A vocação e o passivo ambiental estão nas frases vizinhas."
  ],
  "A090": [
   "correta",
   "nao",
   "Trecho diz que o novo PNLD prevê obras de sociologia e filosofia."
  ],
  "A091": [
   "correta",
   "nao",
   "Trecho traz o tsunami e o faturamento de 100 a 150 bilhões. A frase sobre a Caixa vem em seguida."
  ],
  "A092": [
   "correta",
   "nao",
   "Trecho pede o fortalecimento orçamentário do Ministério. A ligação com envelhecimento ativo não aparece."
  ],
  "A093": [
   "parcial",
   "nao",
   "Trecho cobre só a quebra de protocolos. O relaxamento e as revistas estão nas frases vizinhas."
  ],
  "A094": [
   "parcial",
   "sim",
   "Trecho não traz os juros de 23% nem os 4%. Melhor: Então, é muito importante que saibamos que nós estamos impedindo a"
  ],
  "A095": [
   "correta",
   "nao",
   "Trecho corresponde à citação."
  ],
  "A096": [
   "parcial",
   "sim",
   "Trecho fala em inviabilizar o serviço, não no consumidor. Melhor: Não adianta pensarmos num custo de modo que o consumidor"
  ],
  "A097": [
   "correta",
   "nao",
   "Trecho cobre o mecanismo da citação. Os 8 reais por hora estão nas frases vizinhas."
  ],
  "A098": [
   "correta",
   "nao",
   "Trecho traz o consignado em folha a partir do eSocial. O FGTS Digital vem na frase seguinte."
  ],
  "A099": [
   "parcial",
   "nao",
   "Trecho cobre a primeira frase da citação. A segunda vem em seguida."
  ],
  "A100": [
   "incorreta",
   "sim",
   "Trecho fala de carreira para a Defesa Civil, não de recursos. Melhor: Ele falou de recursos. Desde que eu assumi o meu mandato"
  ],
  "A101": [
   "parcial",
   "sim",
   "Trecho cobre só o papel de fundo de investimento. Melhor: Eu estou convencidíssimo de que o melhor, levando-se em consideração a"
  ],
  "A102": [
   "parcial",
   "nao",
   "Trecho cobre só o Inflation Reduction Act. O Green Deal vem nas frases seguintes."
  ],
  "A103": [
   "parcial",
   "nao",
   "Trecho cobre só a primeira frase da citação. O financiamento perene está nas frases vizinhas."
  ],
  "A104": [
   "parcial",
   "sim",
   "Trecho fala em primeiro a fechar e último a voltar. Melhor: Ao contrário do que vem sendo anunciado, o setor não se"
  ],
  "A105": [
   "correta",
   "nao",
   "Trecho corresponde à citação. Os acordos aparecem na frase seguinte."
  ],
  "A106": [
   "parcial",
   "nao",
   "Trecho cobre a primeira frase da citação. O ENEM e os Estados estão nas frases vizinhas."
  ],
  "A107": [
   "correta",
   "nao",
   "Trecho corresponde à citação."
  ],
  "A108": [
   "parcial",
   "nao",
   "Trecho diz 43 milhões de dólares, e a afirmação 43 bilhões. Única ocorrência na fala."
  ],
  "A109": [
   "parcial",
   "nao",
   "Trecho cobre só a frase sobre o maior medo. Leis, atendimento e centros estão nas frases vizinhas."
  ],
  "A110": [
   "parcial",
   "nao",
   "Trecho cobre os 2 bilhões. Os 250 milhões estão na frase seguinte."
  ],
  "A111": [
   "falou",
   "",
   "O cabeçalho dele registra só manifestação em LIBRAS, mas a intérprete traduz a fala dele nos turnos 9 e 31, incluindo a banca com professores surdos."
  ],
  "A112": [
   "correta",
   "nao",
   "Trecho traz os 6 meses de chuva e 6 de estiagem."
  ],
  "A113": [
   "parcial",
   "sim",
   "Trecho traz 51% de outro recorte, sem os 66,3%. Melhor: Hoje, são 66,3% dos trabalhadores. Do total daqueles que participam"
  ],
  "A114": [
   "correta",
   "nao",
   "Trecho corresponde à citação."
  ],
  "A115": [
   "correta",
   "nao",
   "Trecho pede revisitar o regramento e garantir o fundo. O FUNCAP é nomeado antes no mesmo turno."
  ],
  "A116": [
   "correta",
   "nao",
   "Trecho afirma que o MEI não rola pelo Governo. A razão previdenciária vem nas frases seguintes."
  ],
  "A117": [
   "parcial",
   "sim",
   "Trecho é só um fragmento da citação. Melhor: A guerra em curso, senhores, é contra a Palestina, contra a"
  ],
  "A118": [
   "parcial",
   "sim",
   "Trecho cobre só a revogação. Melhor: Nós queremos aula de sociologia, nós queremos aula de filosofia, porque"
  ],
  "A119": [
   "correta",
   "nao",
   "Trecho corresponde à citação."
  ],
  "A120": [
   "parcial",
   "sim",
   "Trecho é um item da lista de pontos do PL, sem defesa explícita. Melhor: O PL traz alguns pontos principais, como a criação de uma"
  ],
  "A121": [
   "parcial",
   "sim",
   "Trecho é fragmento. Melhor: Eu, pessoalmente, não faço questão de ser por hora ou por"
  ],
  "A122": [
   "parcial",
   "nao",
   "Trecho cobre o fim do fator concorrencial. 99 e Uber não são citadas na fala dele."
  ],
  "A123": [
   "incorreta",
   "sim",
   "Trecho diz que a extensão deve contemplar obras, o contrário do que a afirmação diz. Melhor: Nós acreditamos que essa obra poderia ser contemplada não necessariamente na"
  ],
  "A124": [
   "correta",
   "nao",
   "Trecho contém a parte principal da citação. A frase sobre políticas públicas vem em seguida."
  ],
  "A125": [
   "parcial",
   "sim",
   "Trecho cobre só as crianças. Melhor: E é claro também que mães, gestantes e bebês recém-nascidos que"
  ],
  "A126": [
   "falou",
   "",
   "Fala como O SR. THIAGO XAVIER nos turnos 19, 21 e 23, com a redução de 90% do programa."
  ],
  "A127": [
   "correta",
   "nao",
   "Trecho diz que todos colocarão pontos convergentes e divergentes para o projeto chegar amadurecido ao plenário."
  ]
 },
 "pass_3": {
  "A001": [
   "correta",
   "nao",
   "O trecho diz que o Governo tem perdido prazos e a frase seguinte cita o prazo vencido da reforma do Imposto de Renda."
  ],
  "A002": [
   "incorreta",
   "sim",
   "O trecho critica prefeitos que ignoram a seguranca publica, nao o governo, as criticas ao governo estao em outros pontos da fala, melhor: Eu queria dizer a V.Exa. que isso é eleitoreiro. Nós vamos"
  ],
  "A003": [
   "parcial",
   "sim",
   "O trecho e a proposta de calcular a contribuicao sobre o faturamento, nao a afirmacao de que a contribuicao consumiria metade do faturamento, que esta no inicio do turno, melhor: Para a empresa X, que cobra 10%, isso significa 50% do"
  ],
  "A004": [
   "parcial",
   "sim",
   "O trecho cobre so a parte dos venezuelanos, os ianomamis e a situacao de rua vem nas frases seguintes, e a alteracao da legislacao esta bem mais adiante no turno, melhor: Mas estamos trabalhando, hoje, na alteração da nossa legislação para"
  ],
  "A005": [
   "correta",
   "nao",
   "O trecho diz que o Ministerio orienta portos e cria incentivos para terminais, inclusive privados, reduzirem a pegada de carbono."
  ],
  "A006": [
   "parcial",
   "nao",
   "O trecho cobre a parte principal (fuga excepcional, unica e ultima), os numeros de afastados, PADs, plantao e efetivo vem cerca de 1.500 caracteres depois no mesmo turno."
  ],
  "A007": [
   "parcial",
   "sim",
   "O trecho e so a introducao, a metafora do respirador esta nas duas frases seguintes, melhor: Faço uma metáfora péssima: o que se está fazendo com a"
  ],
  "A008": [
   "correta",
   "nao",
   "O trecho diz que a volta dos presos para Mossoro demonstra confianca absoluta na seguranca do presidio."
  ],
  "A009": [
   "correta",
   "nao",
   "O trecho traz a ideia de trabalhar por um projeto com ganhos reais e a mesma frase continua, logo depois, com a defesa da retirada da urgencia constitucional, os 45 dias sao citados antes no mesmo turno."
  ],
  "A010": [
   "correta",
   "nao",
   "O trecho reproduz a primeira frase da citacao e as frases seguintes trazem o restante, inclusive o doutorado."
  ],
  "A011": [
   "parcial",
   "sim",
   "O trecho cobre o banimento de contas, mas a ausencia de base legal esta em outra parte da fala, melhor: E com qual base legal? Nenhuma. Não há uma lei,"
  ],
  "A012": [
   "parcial",
   "sim",
   "O trecho fala do cuidado com os dados que os adultos postam, nao do controle sobre o que os filhos fazem, melhor: É importante ter algumas precauções em casa. Uma opção seria"
  ],
  "A013": [
   "parcial",
   "sim",
   "O trecho pede que a Caixa se mantenha como empresa importante nas loterias, sem falar em controle integral, a oposicao a transferencia para uma subsidiaria aparece antes no turno, melhor: Isso é algo que nos preocupa bastante, diante da decisão de"
  ],
  "A014": [
   "parcial",
   "sim",
   "O trecho so expressa entusiasmo pelos combustiveis da transicao, sem nada sobre competitividade ou navegacao, melhor: Ela disse que a nossa rota é mais competitiva do que a"
  ],
  "A015": [
   "parcial",
   "sim",
   "O trecho e so a primeira frase curta, a igualdade de tratamento e a critica a responsabilizacao estao nas frases vizinhas, melhor: Nós queremos ser tratados de forma igual. Quem cometeu crime vai"
  ],
  "A016": [
   "correta",
   "nao",
   "O trecho traz a dificuldade de manter neuropediatra e equipe multidisciplinar, com as duas categorias raras."
  ],
  "A017": [
   "parcial",
   "sim",
   "O trecho fala genericamente de investimento publico como prioridade, sem mencionar Estados e Municipios, que aparecem no turno 22, melhor: é fundamental que tenhamos financiamentos específicos, inclusive, com garantia de"
  ],
  "A018": [
   "correta",
   "nao",
   "O trecho diz que, apesar da legislacao, as pessoas com TEA nao conseguem usufruir seus direitos nas cidades e Estados."
  ],
  "A019": [
   "correta",
   "nao",
   "O trecho reproduz quase literalmente a citacao sobre audiencias e pesquisas em que os trabalhadores foram ouvidos."
  ],
  "A020": [
   "correta",
   "nao",
   "O trecho reproduz os dois percentuais e a descricao quase literalmente."
  ],
  "A021": [
   "correta",
   "nao",
   "O trecho reproduz a primeira parte da citacao e as duas frases seguintes trazem o restante."
  ],
  "A022": [
   "correta",
   "nao",
   "O trecho contem a citacao quase literalmente."
  ],
  "A023": [
   "parcial",
   "sim",
   "O trecho so cobre a frase final, a tese principal esta nas frases anteriores, melhor: em nenhum lugar da lei brasileira está escrito que é crime"
  ],
  "A024": [
   "correta",
   "nao",
   "O trecho, com a frase anterior, diz que foi enviado convite ao ministro Fernando Haddad para esta audiencia."
  ],
  "A025": [
   "correta",
   "nao",
   "O trecho, com a frase anterior sobre consensos construidos na Casa, sustenta a afirmacao."
  ],
  "A026": [
   "falou",
   "",
   "Fala no turno 10 sob o cabecalho O SR. MARLISON SOARES CUNHA (um s a menos), inclusive o ponto dos 10 ou 15 dias para o repasse."
  ],
  "A027": [
   "correta",
   "nao",
   "O trecho parabeniza o Ministerio por pensar a seguranca com base nos direitos humanos, sem populismo penal."
  ],
  "A028": [
   "correta",
   "nao",
   "O trecho reproduz a citacao e a frase seguinte diz que esse custo nao e barato."
  ],
  "A029": [
   "correta",
   "nao",
   "O trecho traz a pergunta retorica sobre a biometria e a resposta vem logo em seguida no contexto."
  ],
  "A030": [
   "correta",
   "nao",
   "O trecho, com a frase seguinte que localiza o El Nino no Pacifico Equatorial, sustenta que o aquecimento dos oceanos vai alem do El Nino, os mapas de anomalia sao descritos logo antes."
  ],
  "A031": [
   "correta",
   "nao",
   "O trecho anuncia as solucoes inovadoras de transferencia de riscos e a frase seguinte cita exemplos publico-privados de diversos paises."
  ],
  "A032": [
   "correta",
   "nao",
   "O trecho, com a frase anterior sobre a PEC 504/2010, mostra apoio a PEC pela importancia para a Caatinga e o Nordeste."
  ],
  "A033": [
   "parcial",
   "sim",
   "O trecho pede a unificacao das informacoes geologicas, hidrologicas e meteorologicas, nao um sistema nacional de previsao do tempo, melhor: Reforço aqui a fala do Coronel Henguel no sentido de que"
  ],
  "A034": [
   "parcial",
   "nao",
   "O trecho cobre a revisao dos protocolos, a frase sobre financiamento adequado e a seguinte no mesmo turno."
  ],
  "A035": [
   "correta",
   "nao",
   "O trecho reproduz a citacao sobre gas natural nos veiculos pesados e dependencia do diesel importado, o detalhe da composicao igual ao gas veicular nao aparece no turno, mas e enquadramento da materia."
  ],
  "A036": [
   "parcial",
   "nao",
   "O trecho cobre a gradacao das punicoes, o banimento vitalicio para reincidentes esta so na frase seguinte do mesmo turno."
  ],
  "A037": [
   "parcial",
   "nao",
   "O trecho diz que o grande desafio e criar um arcabouco legal de protecao aos vulneraveis, mas a expressao prioridade absoluta nao aparece no turno."
  ],
  "A038": [
   "parcial",
   "sim",
   "O trecho cobre a necessidade de planejamento (PEI), mas a citacao sobre a crianca que nao aprende esta algumas frases depois, melhor: Claro que é importante a pessoa estar na escola, mas ela"
  ],
  "A039": [
   "correta",
   "nao",
   "O trecho traz a primeira frase da citacao, o 1 milhao de pessoas, os 70% e a catastrofe alimentar vem nas frases seguintes, e o risco de morte no IPC 5 na anterior."
  ],
  "A040": [
   "parcial",
   "sim",
   "O trecho defende a mesma ideia com o exemplo de Goias, mas a citacao esta em outro ponto do turno, melhor: Precisamos sensibilizar o Ministério para investir mais, a fim de"
  ],
  "A041": [
   "correta",
   "nao",
   "O trecho reproduz a citacao quase literalmente, a apresentacao das propostas vem na frase seguinte."
  ],
  "A042": [
   "parcial",
   "nao",
   "O trecho e o contexto ligam o PL 970/24 a exigencias como audiencias publicas e discussao com o TCU, mas nao ha mencao a estudo tecnico previo em nenhum ponto da fala."
  ],
  "A043": [
   "incorreta",
   "sim",
   "O trecho e so um pedido procedimental para o ministro se ater as armas, a critica ao decreto esta em outro turno, melhor: No dia 1º de janeiro de 2023, início do Governo Lula,"
  ],
  "A044": [
   "parcial",
   "sim",
   "O trecho fala do comprometimento do saldo com varios aniversarios, mas nao traz o numero 28, melhor: Acontece que a voracidade do setor financeiro criou a possibilidade"
  ],
  "A045": [
   "correta",
   "nao",
   "O trecho traz o periodo, os dados por pais e o Brasil entre os tres ultimos, a conclusao citada vem na frase seguinte."
  ],
  "A046": [
   "parcial",
   "sim",
   "O trecho fala dos prejuizos da agricultura no Centro e Sudeste, o efeito do desmatamento sobre a chuva e a umidade levada ao resto do Brasil esta nas frases anteriores, melhor: Na medida em que se desmata, reduz-se a chuva e aumenta-se"
  ],
  "A047": [
   "parcial",
   "sim",
   "O trecho e o contexto cobrem a abertura a ajustes discutidos tecnicamente, mas a citacao esta em outro turno, melhor: Nós não podemos tampar os olhos e dizer que os CACs"
  ],
  "A048": [
   "correta",
   "nao",
   "O trecho traz a frase citada sobre nao demorar com a decisao."
  ],
  "A049": [
   "parcial",
   "sim",
   "O trecho cobre so a demanda de cobertura as familias, a criacao da categoria e a invisibilidade estao no paragrafo anterior e a citacao vem logo antes, melhor: Em relação à Previdência Social, como bem lembrou o Abel, muitos"
  ],
  "A050": [
   "falou",
   "",
   "Fala no turno 21 como O SR. WOLNEI WOLFF BARREIROS, inclusive a frase sobre quase 2 mil municipios em emergencia."
  ],
  "A051": [
   "parcial",
   "nao",
   "O trecho e o contexto cobrem a citacao, mas o numero 71.867 esta cerca de 800 caracteres antes no mesmo turno, fora do contexto mostrado."
  ],
  "A052": [
   "correta",
   "nao",
   "O trecho reproduz a citacao e a frase anterior liga a indefinicao ao aumento do custo de capital."
  ],
  "A053": [
   "correta",
   "nao",
   "O trecho reproduz a citacao quase literalmente."
  ],
  "A054": [
   "parcial",
   "nao",
   "O trecho e a frase seguinte trazem a citacao sobre queda da qualidade e usuarios pagando o mesmo, mas a reclamacao da taxa variavel esta em outra passagem do turno."
  ],
  "A055": [
   "parcial",
   "nao",
   "O trecho tem so a primeira frase da citacao, o planejamento das empresas e a seguranca juridica vem nas duas frases seguintes."
  ],
  "A056": [
   "correta",
   "nao",
   "O trecho traz os 95% e os 88% com as mesmas faixas etarias."
  ],
  "A057": [
   "correta",
   "nao",
   "O trecho reproduz a citacao sobre a subsidiaria integral e a consulta a PGFN, a frase seguinte traz o entendimento contrario a exploracao em caso de privatizacao."
  ],
  "A058": [
   "correta",
   "nao",
   "O trecho diz que o Governo revogou no primeiro dia o decreto que trazia de volta escolas segregadas, que e o decreto referido na afirmacao."
  ],
  "A059": [
   "correta",
   "nao",
   "O trecho reproduz a citacao e o contexto seguinte confirma que 8 bilhoes no PROAGRO esta de bom tamanho."
  ],
  "A060": [
   "correta",
   "nao",
   "O trecho traz a citacao dos 120 a 244 bilhoes e o contexto seguinte fala da credibilidade e dos financiamentos comprometidos."
  ],
  "A061": [
   "falou",
   "",
   "Os turnos 11 e 33 com o nome dele sao so rubrica (Manifestacao em LIBRAS), mas a fala dele em Libras esta registrada em primeira pessoa nos turnos 12 e 34 da interprete Adriana Lopes, inclusive os 63.106 alunos."
  ],
  "A062": [
   "parcial",
   "nao",
   "O trecho so nomeia o Twitter Files Brasil, nada sobre o compilado de e-mails, os funcionarios do X ou o periodo, e as falas dele nao trazem isso."
  ],
  "A063": [
   "parcial",
   "sim",
   "O trecho e so a citacao do meio, o Marco Civil e as decisoes genericas estao nas frases vizinhas, melhor: O Marco Civil da Internet é muito claro, salvo engano, no"
  ],
  "A064": [
   "parcial",
   "sim",
   "O trecho fala de censura em geral sem citar Moraes nem bloqueio de contas, o apoio esta nas frases seguintes, melhor: O Juiz Alexandre de Moraes fica muito preocupado em entender"
  ],
  "A065": [
   "correta",
   "nao",
   "O trecho reproduz a primeira frase da citacao e a segunda e a frase seguinte no mesmo turno."
  ],
  "A066": [
   "correta",
   "nao",
   "O trecho reproduz a citacao, referindo-se aos grupos de trabalho paralelos que colocam o pagador de impostos e os consumidores dentro do Parlamento."
  ],
  "A067": [
   "correta",
   "nao",
   "O trecho diz que a solicitacao nao tem fundamentacao legal clara e pode ser ato discriminatorio, o assedio moral vem na frase seguinte."
  ],
  "A068": [
   "parcial",
   "sim",
   "O trecho traz o inicio da citacao, mas o novo requerimento com os mesmos convidados esta bem mais adiante no turno, melhor: Então, eu vou fazer um novo requerimento, que sugiro também aos"
  ],
  "A069": [
   "correta",
   "nao",
   "O trecho traz a frase central de que o projeto nao e simplesmente do Governo e as frases seguintes trazem o restante da citacao."
  ],
  "A070": [
   "correta",
   "nao",
   "O trecho reproduz a citacao do Minha Casa, Minha Vida e a frase seguinte traz a parte de que ninguem escolhe morar em area de risco."
  ],
  "A071": [
   "parcial",
   "sim",
   "O trecho fala so de descumprimento da lei, sem geracao distribuida nem resolucoes da ANEEL, melhor: Elas não obedecem às normas, não obedecem à lei, não obedecem"
  ],
  "A072": [
   "correta",
   "nao",
   "O trecho traz a cobranca de 3 reais ao mes na conta de energia e a indenizacao de 15 mil reais por unidade residencial."
  ],
  "A073": [
   "correta",
   "nao",
   "O trecho reproduz a afirmacao sobre a maior matanca de criancas da historia em guerras."
  ],
  "A074": [
   "parcial",
   "sim",
   "O trecho se refere ao carater intersetorial da revisao do Plano Clima, nao a integracao de adaptacao e mitigacao, que vem algumas frases depois, melhor: E, agora, eu acho muito louvável que o Governo Federal"
  ],
  "A075": [
   "parcial",
   "nao",
   "O trecho e as frases seguintes trazem a citacao completa, mas o dialogo entre base e oposicao acima de disputas ideologicas esta em outra passagem, mais no inicio do turno."
  ],
  "A076": [
   "correta",
   "nao",
   "O trecho diz que 1,1 bilhao para os tuneis poderia ser absorvido no contrato remanescente com incremento de menos de 1 real na tarifa, os numeros da afirmacao sao arredondamentos disso."
  ],
  "A077": [
   "correta",
   "nao",
   "O trecho reproduz a citacao quase literalmente."
  ],
  "A078": [
   "correta",
   "nao",
   "O trecho reproduz a citacao sobre o deslizamento e a frase seguinte traz as pessoas trancadas na rodovia."
  ],
  "A079": [
   "correta",
   "nao",
   "O trecho traz a primeira parte da citacao e a frase seguinte traz a luta maior pelo atendimento nos centros."
  ],
  "A080": [
   "correta",
   "nao",
   "O trecho reproduz a citacao e a frase anterior chama a lei de grande vitoria."
  ],
  "A081": [
   "correta",
   "nao",
   "O trecho diz que o projeto dos dois tuneis estava em pauta desde 2010."
  ],
  "A082": [
   "parcial",
   "nao",
   "O trecho e o contexto trazem a citacao, a proibicao de 1 km so aparece por referencia (essa proibicao) e a restricao por calibre vem bem depois no mesmo turno."
  ],
  "A083": [
   "correta",
   "nao",
   "O trecho traz a primeira citacao sobre conversar e combinar o jogo antes de enviar o projeto, a segunda (enfrentar esse debate) esta duas frases depois."
  ],
  "A084": [
   "correta",
   "nao",
   "O trecho diz que a plataforma funciona no almoco e no jantar e a frase seguinte conclui que isso gera ganho nominal mais baixo."
  ],
  "A085": [
   "falou",
   "",
   "Nao ha Aline Costa na transcricao, mas a representante da Coordenacao-Geral de Saude da Pessoa com Deficiencia fala no turno 25 como A SRA. LUANE CARVALHO COSTA e diz exatamente o conteudo atribuido (politica de 2002 atualizada em 2023, grupos condutores), trata-se de variante do nome."
  ],
  "A086": [
   "correta",
   "nao",
   "O trecho cita oficios ao Planejamento e a Fazenda pela reposicao de recursos e a ida ao gabinete da ministra do Planejamento sendo marcada, a reuniao com a Fazenda nao e explicita."
  ],
  "A087": [
   "incorreta",
   "sim",
   "O trecho e formula de encerramento da reuniao, a resposta do ministerio e apenas anunciada no turno 55, sem que o teor do oficio apareca na transcricao, melhor: Infelizmente, apenas ontem, às 17h11min, obtivemos a seguinte resposta:"
  ],
  "A088": [
   "correta",
   "nao",
   "O trecho traz o planejamento dos tuneis deixado fora da concessao e a frase anterior diz que o tema e caro ha bastante tempo."
  ],
  "A089": [
   "correta",
   "nao",
   "O trecho traz a citacao da economia circular e a frase seguinte completa com o passivo ambiental virando ativo energetico, a vocacao e o saneamento estao logo antes."
  ],
  "A090": [
   "correta",
   "nao",
   "O trecho diz que o novo PNLD preve obras de sociologia e filosofia e a frase seguinte as chama de obras referenciais."
  ],
  "A091": [
   "correta",
   "nao",
   "O trecho traz o tsunami e os 100 a 150 bilhoes, a frase seguinte diz que nao se pode imaginar a Caixa fora desse mercado."
  ],
  "A092": [
   "correta",
   "nao",
   "O trecho traz o pedido de fortalecer o orcamento do Ministerio e a frase anterior traz o restante da citacao."
  ],
  "A093": [
   "parcial",
   "nao",
   "O trecho tem so a quebra dos protocolos, o relaxamento e a falha nas revistas diarias estao nas frases vizinhas do mesmo turno."
  ],
  "A094": [
   "parcial",
   "sim",
   "O trecho inicia o raciocinio, mas os juros de 23% e de 4% ficam fora dele, melhor: Então, é muito importante que saibamos que nós estamos impedindo"
  ],
  "A095": [
   "correta",
   "nao",
   "O trecho reproduz quase literalmente a citacao sobre diversificacao pregada pela pesquisa e extensao rural."
  ],
  "A096": [
   "correta",
   "nao",
   "O trecho diz que o custo nao pode inviabilizar o servico e a frase seguinte traz o consumidor que nao conseguiria pagar."
  ],
  "A097": [
   "correta",
   "nao",
   "O trecho traz a exigencia do salario minimo de contribuicao e a possibilidade de recolher sem receber, os 8 reais por hora aparecem nas frases vizinhas."
  ],
  "A098": [
   "correta",
   "nao",
   "O trecho traz o consignado em folha a partir do eSocial e a frase seguinte acrescenta o Fundo de Garantia Digital, o foco no setor privado vem do diagnostico feito antes no mesmo turno."
  ],
  "A099": [
   "parcial",
   "nao",
   "O trecho tem so a primeira frase da citacao, a segunda, com a posicao concreta, e a frase seguinte do mesmo turno."
  ],
  "A100": [
   "parcial",
   "sim",
   "O trecho defende uma carreira para a Defesa Civil, nao mais recursos, o pedido de recursos vem logo depois no mesmo turno, melhor: Eu trabalhei fortemente pelo financiamento, para colocar recursos no Fundo"
  ],
  "A101": [
   "parcial",
   "sim",
   "O trecho cobre so a funcao de financiar habitacao, saneamento e infraestrutura, a protecao no desemprego e a ideia de resgatar o fundo estao em outras passagens do turno, melhor: Nós precisamos resgatar e fortalecer o Fundo de Garantia, para"
  ],
  "A102": [
   "parcial",
   "nao",
   "O trecho cobre so o Inflation Reduction Act, o Green Deal europeu e a busca de caminhos estao nas frases seguintes do mesmo turno."
  ],
  "A103": [
   "correta",
   "nao",
   "O trecho traz a frase central da citacao (emendas importantes mas pontuais) e as frases vizinhas trazem o restante sobre financiamento perene."
  ],
  "A104": [
   "parcial",
   "sim",
   "O trecho diz que o setor foi o primeiro a fechar e o ultimo a voltar, sem afirmar que nao se recuperou, melhor: Ao contrário do que vem sendo anunciado, o setor não"
  ],
  "A105": [
   "correta",
   "nao",
   "O trecho reproduz a citacao e a frase seguinte fala em reproduzir na regulamentacao os acordos feitos."
  ],
  "A106": [
   "correta",
   "nao",
   "O trecho traz a citacao sobre estarem sendo apagados do curriculo, o ENEM esta na frase anterior e os Estados na frase seguinte."
  ],
  "A107": [
   "correta",
   "nao",
   "O trecho reproduz quase literalmente a frase atribuida."
  ],
  "A108": [
   "parcial",
   "nao",
   "O trecho traz os 60% e a mesma estrutura, mas diz 43 milhoes de dolares, nao 43 bilhoes."
  ],
  "A109": [
   "correta",
   "nao",
   "O trecho traz o maior medo de morrer e deixar os filhos, o restante da citacao vem nas frases seguintes e a falta de atendimento nas anteriores."
  ],
  "A110": [
   "parcial",
   "nao",
   "O trecho traz os 2 bilhoes de pronta resposta, os 250 milhoes do orcamento estao na frase seguinte do mesmo turno."
  ],
  "A111": [
   "falou",
   "",
   "Os turnos 8 e 30 com o nome dele sao so rubrica (Manifestacao em LIBRAS), mas a fala dele em Libras esta registrada em primeira pessoa nos turnos 9 e 31 das interpretes."
  ],
  "A112": [
   "correta",
   "nao",
   "O trecho diz que o Acre sofre 6 meses com chuvas fortes e 6 meses com estiagem, o recorte de tres anos nao aparece, mas o conteudo principal esta la."
  ],
  "A113": [
   "parcial",
   "sim",
   "O trecho traz uma estatistica diferente (51% dos optantes tem renda ate 4 SM) e nao os 66,3%, melhor: Hoje, são 66,3% dos trabalhadores. Do total daqueles que participam"
  ],
  "A114": [
   "correta",
   "nao",
   "O trecho reproduz a citacao sobre o subsidio imediato quando a chuva chegar."
  ],
  "A115": [
   "correta",
   "nao",
   "O trecho pede revisitar o regramento para as situacoes novas e garantir o fundo, o nome FUNCAP aparece no inicio do mesmo paragrafo, pouco antes do contexto mostrado."
  ],
  "A116": [
   "correta",
   "nao",
   "O trecho traz a posicao contraria ao MEI, a justificativa dada em seguida e a falta de sustentabilidade da Previdencia e a familia desguarnecida, proxima ao que a afirmacao resume."
  ],
  "A117": [
   "correta",
   "nao",
   "O trecho traz o nucleo da citacao, a frase anterior diz que a guerra e contra a Palestina e nao contra uma faccao, e a seguinte traz o Fatah, o Hamas e as oliveiras."
  ],
  "A118": [
   "parcial",
   "sim",
   "O trecho traz so a segunda citacao (revogacao da reforma), a dificuldade de acesso e as aulas por serie estao em outra passagem do turno, melhor: Quando passamos em sala de aula, a galera do primeiro ano"
  ],
  "A119": [
   "correta",
   "nao",
   "O trecho reproduz a citacao quase literalmente."
  ],
  "A120": [
   "parcial",
   "sim",
   "O trecho e so um ponto do PL (direito a defesa em bloqueio), nao uma defesa explicita do projeto, a defesa aparece mais claramente adiante, melhor: Sabemos que tem de ser melhorado, mas ele garante o"
  ],
  "A121": [
   "parcial",
   "sim",
   "O trecho e so uma frase introdutoria, a afirmacao de que pode ser por hora ou por quilometro conforme a melhor formulacao esta na frase seguinte, melhor: Eu, pessoalmente, não faço questão de ser por hora ou"
  ],
  "A122": [
   "parcial",
   "nao",
   "O trecho sustenta que a proposta acabaria com o fator concorrencial do aplicativo, mas Uber e 99 nao sao citadas no turno, o favorecimento das maiores aparece so como concentracao de mercado, mais adiante."
  ],
  "A123": [
   "parcial",
   "sim",
   "O trecho diz que a extensao deve contemplar obras, sem dizer que os tuneis cabem no contrato atual, isso vem algumas frases depois, melhor: Nós acreditamos que essa obra poderia ser contemplada não necessariamente"
  ],
  "A124": [
   "correta",
   "nao",
   "O trecho reproduz a citacao sobre a sociedade capacitista, as barreiras de acesso as politicas publicas vem na frase seguinte."
  ],
  "A125": [
   "parcial",
   "sim",
   "O trecho fala so das criancas com menos de 2 anos, as mulheres aparecem mais adiante no turno, melhor: E é claro também que mães, gestantes e bebês recém-nascidos que"
  ],
  "A126": [
   "falou",
   "",
   "Fala nos turnos 19, 21 e 23 como O SR. THIAGO XAVIER, com a apresentacao da Tendencias no tempo cedido pelo FOHB, inclusive a reducao de 90%."
  ],
  "A127": [
   "correta",
   "nao",
   "O trecho diz que todos poderao expor os pontos divergentes para chegar ao plenario com o projeto amadurecido."
  ]
 }
}


def majority(votes: list[str]) -> str:
    label, count = collections.Counter(votes).most_common(1)[0]
    return label if count > 1 else sorted(votes, key=ORDER.get)[1]


def main() -> None:
    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = (SAMPLE_DIR / "annotation.csv").read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(raw.splitlines(), delimiter=";"))
    for row in rows:
        votes = [PASSES[name][row["item_id"]] for name in PASSES]
        row["julgamento"] = majority([vote[0] for vote in votes])
        row["existe_trecho_melhor"] = collections.Counter(vote[1] for vote in votes).most_common(1)[0][0]
        row["observacao"] = "model majority of three passes"
    with open(output_dir / "annotation.csv", "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=";", quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    for name in ("annotation_key.json", "sample_report.json"):
        shutil.copy(SAMPLE_DIR / name, output_dir / name)


if __name__ == "__main__":
    main()
