DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 07:55:37

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7624 | 1803 |  |  |  | 2212 | 1.686 | 1182 | 900.62 |  |
| f31a | 0.6411 | 2144 |  |  |  | 2212 | 1.418 | 1180 | 756.73 | f30a + the diagonal prefixes skip the no-op PVs: TILE_D2 q tile 0, TILE_D3 q tiles 0-1, FINAL_A3 (after TILE_D3) q tiles 0-2 and q tile 2's softmax |
| f32a | 0.6397 | 2149 | f31a | 1.006x | 1.006x | 2212 | 1.415 | 1179 | 755.51 | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads carried in SGPRs) |
| f33a | 0.6373 | 2157 | f32a | 1.000x | 1.000x | 2212 | 1.410 | 1179 | 749.99 | f32a + FINAL_A3 laid out right after TILE_D3 (one ascending code range per block) |
| f33a#null | 0.6372 | 2157 |  |  |  | 2212 | 1.410 | 1179 | 751.34 | f32a + FINAL_A3 laid out right after TILE_D3 (one ascending code range per block) |
