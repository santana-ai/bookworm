# Registros de decisão de arquitetura (ADR)

Cada ADR registra uma decisão que mudou a arquitetura da biblioteca ou os artefatos que ela produz: o
problema que a motivou, a decisão, o efeito medido e as consequências. Uma ADR aceita não é reescrita
quando uma decisão posterior a altera; a nova ADR cita a anterior, e a anterior ganha uma linha
"Atualizado em" no cabeçalho quando o texto dela deixa de descrever o código.

As ADRs citam os caminhos do repositório na data da decisão (`challenge/utils/...`); o caminho atual
de cada um está em [`docs/path_map.md`](../../../docs/path_map.md).

## Índice

| ADR | Título | Status | Data |
| --- | --- | --- | --- |
| [0001](0001-library-scaffold-and-parity.md) | Estrutura da biblioteca e porte com paridade | aceita | 2026-09-22 |
| [0002](0002-per-turn-sentence-segmentation.md) | Segmentação de sentenças por turno e escolha da ocorrência da citação | aceita | 2026-09-22 |
| [0003](0003-single-quoted-spans-as-quotes.md) | Trechos entre aspas simples contam como citação | aceita | 2026-09-22 |
| [0004](0004-exports-read-only-the-run-cache.md) | As exportações leem os embeddings só do cache da execução | aceita | 2026-09-24 |
| [0005](0005-one-transcript-pass-for-udvs-and-actor-profiles.md) | Uma passada pela transcrição para UDVs e perfis de atores, com o perfil conferido contra as UDVs | aceita | 2026-09-26 |
| [0006](0006-udv-v2-windows-full-quotes-and-verifier.md) | `udv_v2`: janelas de duas sentenças, citação inteira e verificador como confiança | aceita | 2026-09-28 |

## Formato

- Nome do arquivo: `NNNN-titulo-em-ingles.md`, numerado em sequência.
- Cabeçalho: título `# ADR NNNN: ...`, seguido de uma lista com `Status` (`proposta`, `aceita` ou
  `substituída pela ADR NNNN`), `Data` e, quando se aplica, o código afetado, as medições e as
  decisões relacionadas.
- Seções: `Contexto`, `Decisão` e `Consequências`, com seções extras quando a decisão depende de
  medições (`Medições`, `Efeito medido`) ou descarta alternativas (`Alternativas consideradas`).
