DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 06:08:17

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7620 | 1804 |  |  |  | 2212 | 1.685 | 1183 | 901.99 |  |
| f28a | 0.6470 | 2125 |  |  |  | 2212 | 1.431 | 1181 | 764.55 | f26a with one RESC branch per q-tile pair (q tiles 2, 3 in the prefix; 0, 1 in the body) |
| f30a | 0.6434 | 2137 | f28a | 1.006x | 1.006x | 2212 | 1.423 | 1181 | 760.86 | f28a + TILE_C ring slots by inline asm from carried descriptors (no LLVM tensorcnt waits, fewer SALU) |
| f30a#null | 0.6422 | 2141 |  |  |  | 2212 | 1.421 | 1181 | 758.40 | f28a + TILE_C ring slots by inline asm from carried descriptors (no LLVM tensorcnt waits, fewer SALU) |
