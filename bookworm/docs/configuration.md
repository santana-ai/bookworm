# Configuração

A configuração fica no projeto que usa a biblioteca, em TOML, e os caminhos relativos são resolvidos a
partir do diretório em que o comando roda. Esta página descreve os formatos de UDV e de split; os de
atores e perfis estão nas páginas de cada etapa.

## Configuração de UDV

Formato de `experiments/configs/udv.toml`:

```toml
[dataset]
lds_path = "dataset/PublicHearingBR_LDS.jsonl"
sha256 = "c4e392ab95ce22f6228eace16f15e9efe1c846724846b873dec420aec3d872c0"

[encoder]
name = "PORTULAN/serafim-335m-portuguese-pt-sentence-encoder"
revision = "a01887015444f7599669c509447c5bdbce958916"
batch_size = 64
device = "auto"

[evidence]
embedding_threshold = 0.45

[run]
seed = 42
output_dir = "artifacts/udv"
cache_dir = "artifacts/cache/embeddings"
```

Sem `kind`, `[encoder]` descreve um modelo `sentence-transformers`; com `kind = "tfidf"` (e
`max_features` opcional), o encoder TF-IDF é ajustado nas sentenças e opiniões das audiências
selecionadas. Chaves extras são ignoradas, e o TOML inteiro é copiado para o campo `config` do arquivo
de cobertura. O arquivo de `experiments/` tem também as chaves de registro da calibração do corte
(`calibration_source`, `calibration_method`, `threshold_decision`, `previous_embedding_threshold` e a
tabela `[calibration]`), que a biblioteca só copia.

Duas chaves opcionais de `[evidence]` ligam a construção de `udv_v2`
([ADR 0006](adr/0006-udv-v2-windows-full-quotes-and-verifier.md)); sem elas, a biblioteca faz
exatamente o que fazia em `udv_v1`.

- `semantic_unit`: `"sentence"` (padrão), `"window2"` ou `"window3"`. Com janela, a unidade candidata
  da busca semântica são 2 ou 3 sentenças candidatas consecutivas do mesmo turno (passo 1); o encoder
  recebe as sentenças juntadas por espaço, e a evidência guarda o trecho da transcrição da primeira à
  última sentença. Os embeddings ficam no cache com o rótulo `window2s_<id>` ou `window3s_<id>`.
- `quote_extent`: `"prefix_sentence"` (padrão) ou `"full_quote"`. Com `full_quote`, a evidência de
  `quote_found` vai da sentença do prefixo até o fim da citação, achado por um sufixo de 6, 4 ou 3
  palavras a no máximo 2 vezes o tamanho da citação; sem sufixo, cobre tantas partes de sentença quantas
  a citação tem, sem sair do turno.

`verify-udvs` lê as duas chaves da configuração gravada na cobertura e confere cada evidência com a
mesma regra. `export-hearing` e `export-site` leem as duas chaves de `--config`, recusam uma execução
cuja cobertura registra outro pipeline ou outro corte e ordenam as candidatas na mesma unidade da
construção.

## Configuração de split

Formato de `experiments/configs/splits.toml`:

```toml
[dataset]
lds_path = "dataset/PublicHearingBR_LDS.jsonl"
sha256 = "c4e392ab95ce22f6228eace16f15e9efe1c846724846b873dec420aec3d872c0"

[temporal]
split_version = "temporal_v1"
train_fraction = 0.70
validation_fraction = 0.15
min_boundary_gap_days = 2

[leakage]
near_duplicate_threshold = 0.5
near_duplicate_pairs = 10

[run]
seed = 42
output_dir = "artifacts/splits"
udv_path = "artifacts/udv/udv_v0.jsonl"
```

`train_fraction` e `validation_fraction` precisam estar entre 0 e 1 e somar menos que 1;
`min_boundary_gap_days` precisa ser pelo menos 1; `split_version` vira nome de arquivo, então só aceita
letras, dígitos, `.`, `_` e `-`. O TOML inteiro é copiado para o campo `config` do relatório.

## Configuração de atores e perfis

Os formatos de `experiments/configs/hearing_actors.toml`, `actor_profiles.toml` e
`profile_validation.toml` estão em [actors.md](actors.md), [profiles.md](profiles.md)
e [profile_validation.md](profile_validation.md).

