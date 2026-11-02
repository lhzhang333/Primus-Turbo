DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 05:58:51

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7619 | 1804 |  |  |  | 2212 | 1.685 | 1183 | 902.03 |  |
| f26a | 0.6483 | 2121 |  |  |  | 2212 | 1.434 | 1182 | 766.37 | f25b + sum tree partial sums in the leaf whose VGPR banks differ from the other operands (no accumulate-form bank conflict) |
| f27a | 0.6710 | 2049 |  |  |  | 2211 | 1.484 | 1183 | 795.55 | f26a with O *= corr on every tile (no RESC branches) |
| f28a | 0.6454 | 2130 | f26a | 1.005x | 1.005x | 2212 | 1.428 | 1181 | 762.14 | f26a with one RESC branch per q-tile pair (q tiles 2, 3 in the prefix; 0, 1 in the body) |
| f29x | 0.6500 | 2115 |  |  |  | 2211 | 1.437 | 1182 | 768.94 | timing ablation (wrong results): f27a with the O *= corr multiplier from an SGPR pair (no VGPR bank conflicts) |
| f29x#null | 0.6483 | 2120 |  |  |  | 2211 | 1.433 | 1182 | 767.56 | timing ablation (wrong results): f27a with the O *= corr multiplier from an SGPR pair (no VGPR bank conflicts) |
