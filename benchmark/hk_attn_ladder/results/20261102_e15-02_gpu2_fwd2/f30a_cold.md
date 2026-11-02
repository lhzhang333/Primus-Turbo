DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 06:07:24

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8346 | 1647 |  |  |  | 1486 | 1.240 | - | - |  |
| f28a | 0.7387 | 1861 |  |  |  | 1358 | 1.003 | - | - | f26a with one RESC branch per q-tile pair (q tiles 2, 3 in the prefix; 0, 1 in the body) |
| f30a | 0.7378 | 1863 | f28a | 1.001x | 1.001x | 1360 | 1.002 | - | - | f28a + TILE_C ring slots by inline asm from carried descriptors (no LLVM tensorcnt waits, fewer SALU) |
| f30a#null | 0.7365 | 1867 |  |  |  | 1357 | 0.999 | - | - | f28a + TILE_C ring slots by inline asm from carried descriptors (no LLVM tensorcnt waits, fewer SALU) |
