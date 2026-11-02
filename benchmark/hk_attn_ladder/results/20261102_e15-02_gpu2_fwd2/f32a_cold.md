DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 07:40:22

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8345 | 1647 |  |  |  | 1484 | 1.239 | - | - |  |
| f31a | 0.7332 | 1875 |  |  |  | 1358 | 0.996 | - | - | f30a + the diagonal prefixes skip the no-op PVs: TILE_D2 q tile 0, TILE_D3 q tiles 0-1, FINAL_A3 (after TILE_D3) q tiles 0-2 and q tile 2's softmax |
| f32a | 0.7305 | 1882 | f31a | 1.003x | 1.005x | 1354 | 0.990 | - | - | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads carried in SGPRs) |
| f32a#null | 0.7294 | 1885 |  |  |  | 1354 | 0.988 | - | - | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads carried in SGPRs) |
