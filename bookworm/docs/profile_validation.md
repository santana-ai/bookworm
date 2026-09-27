# Conferência dos perfis de atores contra as UDVs

## Problema

Um perfil de ator é um texto escrito por um LLM a partir das falas de uma pessoa nas audiências de
treino. Ele resume a mesma fala que as UDVs ancoram, mas sem nenhuma conferência: nada no arquivo de
perfis diz se as posições que ele atribui à pessoa estão certas, se ele esqueceu posições que a
matéria registrou ou se ele descreve a pessoa de um jeito que a distingue das outras.

A UDV já resolve a pergunta análoga para a matéria: cada opinião que a matéria atribui a alguém
precisa ter um trecho da transcrição, dito por essa pessoa, que a sustente. A conferência do perfil
faz a mesma pergunta com os papéis trocados. Dada uma opinião verificada do ator (uma UDV com
evidência na transcrição), existe uma sentença do perfil que a sustente?

## O que é calculado

`bookworm validate-profiles` percorre as UDVs de uma rodada e, para cada uma:

1. descarta a UDV se o participante não foi resolvido para nenhum turno (`person_not_resolved`), se o
   nível não está em `[validation].tiers` (por padrão `quote_found` e `semantic_match_high`), se não
   há linha de ligação UDV → ator, se a ligação não tem ator ou se o ator não tem perfil. Cada motivo é
   contado no relatório;
2. procura o perfil pelo nome de exibição do ator (`UdvActorLink.actor == ProfileRecord.actor`);
3. segmenta o perfil em sentenças: cada linha não vazia passa por
   `bookworm.transcript.sentences.split_sentences`, o mesmo segmentador das falas (fragmentos com menos
   de quatro palavras são descartados). Um perfil sem nenhuma sentença é recusado;
4. calcula, com um `SentenceEncoder`, o cosseno entre a proposição da UDV e cada sentença do perfil. O
   score do par é o maior cosseno, e a sentença correspondente é gravada como a evidência no perfil;
5. calcula o mesmo score contra o perfil de todos os outros atores e grava a posição do perfil
   verdadeiro nessa ordem (`rank`). Empates contam contra o perfil verdadeiro: `rank = 1 + número de
   outros perfis com score maior ou igual`. Assim, um encoder que dá o mesmo score a todos os perfis
   obtém o pior rank, e não o melhor.

A partir dos ranks saem duas métricas de identificação do ator, que não dependem de corte nem de
rótulo humano: acerto no topo (`acc_at_1`, fração de pares com `rank == 1`) e MRR (média de
`1 / rank`). Para que esses números tenham referência, o relatório traz a linha de base do acaso com
`n` perfis candidatos: `acc_at_1 = 1 / n` e MRR esperado `= (1 + 1/2 + ... + 1/n) / n`.

O encoder segue a configuração de `udv.toml`: TF-IDF no núcleo (`kind = "tfidf"`) ou um modelo
`sentence-transformers` no extra `embeddings`, com o cache de embeddings em `cache_dir`. No TF-IDF, o
vocabulário e os pesos são ajustados só nas sentenças dos perfis; as proposições das UDVs, inclusive
as de audiências de teste, nunca entram no corpus ajustado.

### Grupos por split

Os pares são separados pelo split da audiência da UDV, segundo o manifesto de splits:

- `in_prompt`: a audiência está num split de geração (`generation_splits`, por padrão `train`) e
  está em `hearing_ids` do perfil, isto é, entrou no prompt. Mede cobertura: se o perfil guardou as
  posições que a matéria registrou;
- `held_out`: a audiência está num split não usado na geração (`held_out_splits`, por padrão `test`).
  Mede se o perfil permite reconhecer a posição do ator numa audiência que ele não leu.

Os dois grupos nunca são somados num mesmo número. UDVs de um split que não é de geração nem
reservado (por exemplo `validation`, com a configuração padrão) são contadas como
`split_not_evaluated`; UDVs de uma audiência de treino que não entrou no prompt daquele ator (a fala
dele nessa audiência foi descartada pela política de falas) são contadas como `not_in_prompt`.

### Recusas por vazamento

- Um perfil cujo `hearing_ids` contém uma audiência fora dos `generation_splits`, ou ausente do
  manifesto, é recusado antes de qualquer score: um perfil que leu audiências de teste não pode ser
  avaliado nelas.
- Um par `held_out` cuja audiência está em `hearing_ids` do perfil é recusado.
- A configuração recusa um split listado ao mesmo tempo em `generation_splits` e em
  `held_out_splits`.

### Intervalos

Para cada grupo, o relatório traz média, mediana e quartis do score, `acc_at_1` e MRR, com intervalos
bootstrap por audiência: cada réplica sorteia audiências inteiras com reposição (as UDVs de uma mesma
audiência compartilham tema, transcrição e falantes e não são independentes), com semente fixa
(`seed`) e `bootstrap_samples` réplicas. O intervalo é o percentil `(1 - confidence_level) / 2` e o
seu complemento.

## O que o cosseno mede e o que não mede

O cosseno mede proximidade de conteúdo entre a proposição e a sentença do perfil. Ele não mede
implicação: uma sentença com as mesmas palavras e a posição oposta pode ter score alto, e uma
paráfrase com outro vocabulário pode ter score baixo, principalmente com TF-IDF. Quando o score de um
par é zero, a sentença gravada como evidência é só a primeira do perfil e não tem relação com a
proposição. Por isso o score serve para ordenar, identificar atores e escolher a amostra humana; a
afirmação de que um perfil sustenta uma opinião vem do julgamento humano descrito abaixo.

## Revisão humana

`bookworm sample-profile-review` lê o arquivo de pares e sorteia, com a semente de `[review].seed`,
uma amostra estratificada por grupo e por faixa de score. As faixas são definidas pelos cortes de
`[review].score_bands` (por exemplo `[0.3, 0.5]` dá `[-inf, 0.30)`, `[0.30, 0.50)` e `[0.50, inf)`),
e `[review].sizes` dá quantos pares sortear por faixa em cada grupo. Quando uma faixa tem menos pares
que o pedido, todos entram e a falta é registrada. A ordem das linhas é embaralhada com uma segunda
sequência da mesma semente, para que os estratos não apareçam agrupados na planilha.

O comando grava:

- o CSV (`[review].output`, separado por `;`, UTF-8 com BOM, para abrir no Excel ou no Numbers), com
  só o que é preciso para julgar: `item_id`, `udv_id`, `actor`, `proposition`, `udv_evidence` (o
  trecho da transcrição que sustenta a UDV), `profile_evidence` (a sentença do perfil de maior
  score), `profiles_file` (o arquivo de perfis, onde o perfil completo é achado pelo `actor`), e as
  colunas vazias `julgamento` e `observacao`;
- `<output>_sample.json`, a chave da amostra, com o sha256 do arquivo de pares, a semente, as faixas,
  os tamanhos, a população e o número sorteado de cada estrato e, para cada `item_id`, o `udv_id`, o
  `actor`, o grupo, o split, a audiência, o score e a faixa de score.

A planilha é cega: o grupo, o split, o score e a faixa ficam só na chave, porque saber que um par
tem score alto ou que a audiência entrou no prompt pode puxar o julgamento para `sustentada`. A chave
não deve ser aberta antes do fim da anotação.

O campo `julgamento` nunca vem preenchido. Quem anota lê a proposição, a evidência da UDV e o perfil
completo, e escolhe um valor:

- `sustentada`: o perfil afirma a mesma posição, com o conteúdo principal da proposição;
- `compativel`: o perfil trata do mesmo assunto sem contradizer a proposição, mas não a afirma
  (falta a posição, o número ou o destinatário);
- `contradita`: o perfil atribui ao ator a posição oposta;
- `sem_relacao`: o perfil não trata do assunto da proposição.

`bookworm score-profile-review --annotations CSV` lê a planilha preenchida (aceita `;`, `,` ou
tabulação), normaliza os rótulos (espaços nas pontas, maiúsculas, acentos e hífens: `Compatível` vira
`compativel`) e recusa a planilha se algum rótulo for inválido, se algum item estiver sem julgamento,
se faltar ou sobrar item em relação à amostra, se as colunas `udv_id` ou `actor` tiverem sido
editadas, se a chave não bater com o arquivo de pares ou se o arquivo de pares mudou depois do
sorteio. O grupo e a faixa de cada item vêm da chave. O relatório
(`<output>_report.json`) traz, por grupo:

- a proporção de cada rótulo e de dois desfechos, `strict_support` (`sustentada`) e
  `tolerant_support` (`sustentada` ou `compativel`), com intervalo de Wilson;
- as mesmas proporções por faixa de score;
- a estimativa ponderada pela população de cada faixa, com intervalo pela aproximação normal do
  total estratificado. As proporções da amostra sem ponderação super-representam as faixas pequenas,
  porque cada faixa recebe o mesmo número de itens; a estimativa ponderada corrige isso e fica vazia
  quando alguma faixa com pares não teve item sorteado.

## Como rodar

```bash
uv run bookworm validate-profiles --config configs/profile_validation.toml
uv run bookworm sample-profile-review --config configs/profile_validation.toml
# preencher julgamento e observacao no CSV
uv run bookworm score-profile-review --config configs/profile_validation.toml \
    --annotations artifacts/profile_validation/profile_review.csv
```

`validate-profiles` e `sample-profile-review` não sobrescrevem arquivos existentes sem `--overwrite`.
Os caminhos da configuração são relativos ao diretório onde o comando roda.

## Configuração

```toml
[inputs]
udv_path = "artifacts/udv/udv_v1.jsonl"
links_path = "artifacts/udv/udv_v1_actor_links.jsonl"
profiles_path = "artifacts/actor_profiles/profiles.jsonl"
split_manifest = "artifacts/splits/temporal_v1.json"

[validation]
tiers = ["quote_found", "semantic_match_high"]
generation_splits = ["train"]
held_out_splits = ["test"]
cache_dir = "artifacts/cache/embeddings"
output_dir = "artifacts/profile_validation"
name = "profile_validation_v1"
seed = 42
bootstrap_samples = 1000
confidence_level = 0.95

[validation.encoder]
name = "PORTULAN/serafim-335m-portuguese-pt-sentence-encoder"
revision = "a01887015444f7599669c509447c5bdbce958916"
batch_size = 64
device = "auto"

[review]
seed = 42
score_bands = [0.3, 0.5]
sizes = { in_prompt = 20, held_out = 20 }
output = "artifacts/profile_validation/profile_review.csv"
```

`tiers`, `generation_splits`, `held_out_splits` e `confidence_level` têm os valores acima como
padrão. Para usar TF-IDF, troque a tabela do encoder por `kind = "tfidf"` (e, opcionalmente,
`max_features`). Os valores de `score_bands` e `sizes` acima são um exemplo: os cortes precisam ser
escolhidos antes de ver os julgamentos. A configuração usada nos testes está em
`tests/fixtures/profile_validation_mini.toml`.

## Formatos de saída

`<output_dir>/<name>_pairs.jsonl`, uma linha por par:

| Campo | Conteúdo |
| --- | --- |
| `udv_id`, `hearing_id`, `split`, `tier` | a UDV e o split da audiência |
| `group` | `in_prompt` ou `held_out` |
| `actor_key`, `actor` | a chave normalizada da ligação e o nome de exibição do ator |
| `udv_actor` | o nome do envolvido como a matéria o escreve |
| `proposition`, `udv_evidence` | a opinião e o trecho da transcrição que a sustenta |
| `profile_sentence`, `profile_sentence_index` | a sentença do perfil de maior cosseno e a sua posição |
| `score` | o maior cosseno contra o perfil do próprio ator |
| `rank`, `n_candidates` | a posição do perfil verdadeiro entre todos os perfis |
| `best_other_actor`, `best_other_score` | o perfil de outro ator com maior score |

`<output_dir>/<name>_report.json` traz o eco da configuração (`config`), o sha256 de cada entrada
(`inputs`), a versão do split, a identidade do encoder, a descrição do método, as contagens (`udvs`,
`profiles`, `profile_sentences`, `skipped` por motivo e `pairs` por grupo) e, em `groups`, as métricas
de cada grupo com a linha de base do acaso e os intervalos bootstrap.
