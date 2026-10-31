# Ascend EPLB micro-test evidence — 2026-10-31

These device experiments validate weight migration and valid-route counting in the initial implementation of [PR #47574](https://github.com/sgl-project/sglang/pull/47574).

## Source versions

- HCCL migration used `f5f46fb1a6832490273307da5f9d670219d996df`. Its complete Git tree equals initial PR commit [`950433a6`](https://github.com/AkiYangDev/sglang/commit/950433a6fdb7c378c132f6cf38fba95d47e3f30e): `65c3219e9e44474d28e7cacdb01f39527c925ac6`.
- DeepEP counting used `950433a6fdb7c378c132f6cf38fba95d47e3f30e`.
- The archived device payloads were compared to that commit. The two migration source files and the statistics source file match after line-ending normalization. Their SHA-256 values are included below.

## HCCL migration

[All 16 rank results and configuration](hccl_migration.json)

Real HCCL default-WORLD transfers on Ascend 910C, with five layers, 32 physical slots, 30 logical experts, and two slots per rank. Each layer contains a BF16 matrix and strided FP32 auxiliary weights. Both ND and actual NZ format 29 pass exact-value and original-storage-address checks on every rank. Fixture layouts exercise unchanged layers, sending without local replacement, and shuffled placement. No model is loaded.

## DeepEP valid-route counts

[Both ranks, all per-expert vectors, configuration, and package versions](deepep_counts.json)

Two Ascend 910C ranks, 64 experts, TopK 8, hidden size 7168, and eight allocated rows per rank. The production sender histogram is captured once and replayed with valid rows 8, 4, 0, and 2. Actual DeepEP low-latency dispatch/combine runs eagerly; every case passes exact combine-output checks.

| Valid rows per rank | Global receive count | Global valid TopK count |
| ---: | ---: | ---: |
| 8 | 128 | 128 |
| 4 | 128 | 64 |
| 0 | 128 | 0 |
| 2 | 128 | 32 |

The controlled skew case routes real tokens to rank 0 and padded rows to rank 1. At four valid rows per rank, receive loads are 64/64 while valid-route loads are 64/0. The JSON includes every expert count, so these totals can be recomputed directly. Each rank receipt contains the same globally reduced vectors; do not sum the two copies.

These are device correctness experiments. Model accuracy and serving throughput have separate measurements in the PR. Public exports preserve the recorded numeric receipts and exclude internal runtime paths and process-owner metadata; hashes identify the original source artifacts.
