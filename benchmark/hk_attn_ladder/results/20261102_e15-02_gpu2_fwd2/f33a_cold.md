DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 07:54:32

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8374 | 1642 |  |  |  | 1484 | 1.242 | - | - |  |
| f31a | 0.7349 | 1871 |  |  |  | 1360 | 0.999 | - | - | f30a + the diagonal prefixes skip the no-op PVs: TILE_D2 q tile 0, TILE_D3 q tiles 0-1, FINAL_A3 (after TILE_D3) q tiles 0-2 and q tile 2's softmax |
| f32a | 0.7291 | 1886 | f31a | 1.007x | 1.013x | 1352 | 0.986 | - | - | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads carried in SGPRs) |
| f33a | 0.7298 | 1884 | f32a | 1.001x | 1.000x | 1352 | 0.986 | - | - | f32a + FINAL_A3 laid out right after TILE_D3 (one ascending code range per block) |
| f33a#null | 0.7290 | 1886 |  |  |  | 1352 | 0.985 | - | - | f32a + FINAL_A3 laid out right after TILE_D3 (one ascending code range per block) |
