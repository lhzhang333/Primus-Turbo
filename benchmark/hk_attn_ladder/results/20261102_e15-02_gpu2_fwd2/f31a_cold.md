DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 07:15:47

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8366 | 1643 |  |  |  | 1486 | 1.241 | - | - |  |
| f30a | 0.7381 | 1862 |  |  |  | 1359 | 1.001 | - | - | f28a + TILE_C ring slots by inline asm from carried descriptors (no LLVM tensorcnt waits, fewer SALU) |
| f31a | 0.7329 | 1876 | f30a | 1.007x | 1.004x | 1363 | 0.998 | - | - | f30a + the diagonal prefixes skip the no-op PVs: TILE_D2 q tile 0, TILE_D3 q tiles 0-1, FINAL_A3 (after TILE_D3) q tiles 0-2 and q tile 2's softmax |
| f31a#null | 0.7331 | 1875 |  |  |  | 1362 | 0.999 | - | - | f30a + the diagonal prefixes skip the no-op PVs: TILE_D2 q tile 0, TILE_D3 q tiles 0-1, FINAL_A3 (after TILE_D3) q tiles 0-2 and q tile 2's softmax |
