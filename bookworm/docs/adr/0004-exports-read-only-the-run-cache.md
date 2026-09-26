# ADR 0004: as exportações leem os embeddings só do cache da execução

- Status: aceita
- Data: 2026-09-24
- Código afetado: `src/bookworm/cli.py` (`export-hearing`, `export-site`, `run_cache_encoder`,
  `load_export_run`), `src/bookworm/features/encoders.py` (`RunCacheEncoder`,
  `CachedEncoder(cache_only=True)`), `src/bookworm/errors.py` (`EmbeddingCacheMissError`) e
  `src/bookworm/udv/site.py`
- Testes: `tests/unit/test_export.py`, `tests/unit/test_site.py`, `tests/unit/test_encoders.py` e,
  com o marcador `dataset`, `tests/integration/test_export_site.py`

## Contexto

`export-hearing` recalcula, para cada opinião, as sentenças do participante mais parecidas com ela e
confere que, nas UDV semânticas, a primeira é a evidência gravada na execução, com o mesmo score
arredondado a 4 casas. Até esta decisão, o encoder da exportação vinha da mesma fábrica de
`build-udvs`: com `sentence-transformers`, um `SentenceTransformerEncoder` que só carrega o modelo
quando precisa codificar um texto, envolvido por um `CachedEncoder` que, quando o arquivo não estava no
cache, codificava os textos e gravava o resultado. Isso trazia quatro problemas para a exportação:

1. Um arquivo ausente do cache fazia o comando carregar o modelo no dispositivo resolvido na hora
   (`device = "auto"` escolhe `mps` numa máquina Apple) e gravar embeddings novos no cache da
   execução, sem aviso.
2. O nome dos arquivos do cache inclui o dispositivo (`<nome>@<revisão>@<dispositivo>`). O dispositivo
   que vale é o da execução, gravado em `encoder_runtime.device` do arquivo de cobertura (`mps` em
   `udv_v1`). Resolver `auto` de novo numa máquina sem `mps` gera outro nome, e todos os arquivos
   passam a faltar.
3. Embeddings recalculados em outro dispositivo podem diferir dos que geraram a execução nas últimas
   casas decimais, e o arquivo exportado deixaria de descrever a mesma execução.
4. Uma audiência sem nenhuma sentença candidata chamava `encode([])` no encoder, e o
   `SentenceTransformerEncoder` carregava o modelo só para saber a dimensão da matriz vazia.

`export-site` exporta todas as audiências de uma execução numa passada, então esses casos se
repetiriam em cada uma delas.

## Decisão

- `export-hearing` e `export-site` montam o encoder a partir da execução. Com `sentence-transformers`,
  é um `RunCacheEncoder` com `name` e `revision` da configuração e o dispositivo de
  `encoder_runtime.device` da cobertura; ele só dá nome aos arquivos do cache e para com
  `EmbeddingCacheMissError` se alguém pedir que codifique um texto. Com TF-IDF, o encoder continua
  sendo reajustado, em CPU, no corpus das audiências da execução, como em `build-udvs`, porque a
  identidade dele inclui o hash desse corpus e não está gravada na cobertura.
- O cache é aberto com `CachedEncoder(..., cache_only=True)`: cada chamada lê o arquivo existente ou
  para com `EmbeddingCacheMissError`, que traz o rótulo (`sentences_<id>` ou `opinions_<id>`), a
  identidade do encoder e o caminho esperado. Nada é gravado no cache. Uma lista vazia de textos
  devolve uma matriz `(0, 0)` sem chamar o encoder.
- Com `sentence-transformers`, uma cobertura sem `encoder_runtime.device` em texto é recusada com
  código 2, porque sem ele não há como nomear os arquivos do cache.
- A fábrica de `create_app(encoder_factory)` continua sendo usada por `build-udvs`; os comandos de
  exportação não a chamam.

## Consequências

- Nenhuma exportação carrega modelo ou importa `torch` e `sentence_transformers`: um teste roda
  `export-site` num interpretador novo e confere `sys.modules` no fim.
- A exportação das 206 audiências de `udv_v1` lê todos os arquivos do cache existente sem nenhuma
  falta, e o diretório do cache termina com os mesmos nomes, tamanhos e datas de modificação
  (`tests/integration/test_export_site.py`).
- Uma execução cujo cache foi apagado, ou não foi copiado junto com os arquivos da execução, não pode
  ser exportada até que `build-udvs` rode de novo; a mensagem de erro diz qual arquivo falta. Como o
  dispositivo vem da cobertura, a máquina que exporta não precisa ter o dispositivo da execução.
- O teste de CLI que trocava a fábrica para provocar a recusa por revisão diferente passou a trocar a
  revisão no TOML, que é de onde a exportação lê o nome e a revisão do encoder.
