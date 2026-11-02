DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 08:42:53

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7615 | 1805 |  |  |  | 2212 | 1.684 | 1183 | 900.69 |  |
| f32a | 0.6370 | 2158 |  |  |  | 2212 | 1.409 | 1180 | 753.53 | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads carried in SGPRs) |
| f37a | 0.6393 | 2150 | f32a | 0.998x | 0.998x | 2213 | 1.415 | 1179 | 754.79 | f32a with all four Q fragments resident (Q(3) in block 2: softmax scratch in the leaves, m / d in the pairs' high halves): no Q LDS reads in the tiles |
| f37a#null | 0.6378 | 2155 |  |  |  | 2213 | 1.411 | 1179 | 752.10 | f32a with all four Q fragments resident (Q(3) in block 2: softmax scratch in the leaves, m / d in the pairs' high halves): no Q LDS reads in the tiles |
