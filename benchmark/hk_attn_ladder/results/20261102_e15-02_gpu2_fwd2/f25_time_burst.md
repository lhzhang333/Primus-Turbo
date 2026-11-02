DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 05:07:46

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7573 | 1815 |  |  |  | 2211 | 1.674 | 1183 | 897.31 |  |
| f22a | 0.6584 | 2088 |  |  |  | 2212 | 1.456 | 1180 | 776.18 | f19f + per-q-tile O resources with 32-bit bounds (no 64-bit clamp chains in the epilogue) |
| f24a | 0.6500 | 2115 | f22a | 1.011x | 1.011x | 2212 | 1.438 | 1180 | 767.85 | FINAL_A: one scheduled region -- the last prefix, then out / lse of all 4 q tiles stored to the staging by the generated code |
| f25a | 0.6513 | 2111 | f24a | 0.998x | 0.997x | 2212 | 1.441 | 1180 | 769.15 | f22a + K / V slot descriptors bounded at the tile's 64 rows (fewer SALU per TDM op) |
| f25b | 0.6470 | 2125 |  |  |  | 2212 | 1.431 | 1181 | 763.70 | f25a + FINAL_A |
| f25b#null | 0.6475 | 2123 |  |  |  | 2212 | 1.432 | 1180 | 764.15 | f25a + FINAL_A |
