DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 08:42:01

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8388 | 1639 |  |  |  | 1483 | 1.243 | - | - |  |
| f32a | 0.7297 | 1884 |  |  |  | 1350 | 0.985 | - | - | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads carried in SGPRs) |
| f37a | 0.7265 | 1892 | f32a | 1.005x | 0.989x | 1373 | 0.997 | - | - | f32a with all four Q fragments resident (Q(3) in block 2: softmax scratch in the leaves, m / d in the pairs' high halves): no Q LDS reads in the tiles |
| f37a#null | 0.7225 | 1903 |  |  |  | 1370 | 0.992 | - | - | f32a with all four Q fragments resident (Q(3) in block 2: softmax scratch in the leaves, m / d in the pairs' high halves): no Q LDS reads in the tiles |
