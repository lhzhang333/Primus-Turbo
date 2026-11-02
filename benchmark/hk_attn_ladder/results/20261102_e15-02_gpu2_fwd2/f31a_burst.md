DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 07:16:40

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7630 | 1802 |  |  |  | 2212 | 1.687 | 1184 | 903.21 |  |
| f30a | 0.6444 | 2133 |  |  |  | 2212 | 1.425 | 1181 | 761.17 | f28a + TILE_C ring slots by inline asm from carried descriptors (no LLVM tensorcnt waits, fewer SALU) |
| f31a | 0.6426 | 2139 | f30a | 1.004x | 1.004x | 2212 | 1.421 | 1182 | 760.53 | f30a + the diagonal prefixes skip the no-op PVs: TILE_D2 q tile 0, TILE_D3 q tiles 0-1, FINAL_A3 (after TILE_D3) q tiles 0-2 and q tile 2's softmax |
| f31a#null | 0.6425 | 2140 |  |  |  | 2212 | 1.421 | 1181 | 758.93 | f30a + the diagonal prefixes skip the no-op PVs: TILE_D2 q tile 0, TILE_D3 q tiles 0-1, FINAL_A3 (after TILE_D3) q tiles 0-2 and q tile 2's softmax |
