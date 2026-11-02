DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 07:41:15

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7504 | 1832 |  |  |  | 2212 | 1.659 | 1183 | 889.37 |  |
| f31a | 0.6299 | 2183 |  |  |  | 2212 | 1.394 | 1181 | 744.76 | f30a + the diagonal prefixes skip the no-op PVs: TILE_D2 q tile 0, TILE_D3 q tiles 0-1, FINAL_A3 (after TILE_D3) q tiles 0-2 and q tile 2's softmax |
| f32a | 0.6249 | 2200 | f31a | 1.007x | 1.007x | 2213 | 1.383 | 1181 | 739.32 | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads carried in SGPRs) |
| f32a#null | 0.6242 | 2202 |  |  |  | 2212 | 1.381 | 1180 | 739.51 | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads carried in SGPRs) |
