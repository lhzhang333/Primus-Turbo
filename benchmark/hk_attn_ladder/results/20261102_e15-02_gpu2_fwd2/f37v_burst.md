DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 08:51:01

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7626 | 1803 |  |  |  | 2211 | 1.684 | 1180 | 900.32 |  |
| f32a | 0.6394 | 2150 |  |  |  | 2212 | 1.414 | 1178 | 753.15 | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads carried in SGPRs) |
| f37a | 0.6409 | 2145 | f32a | 0.999x | 0.998x | 2213 | 1.418 | 1177 | 754.15 | f32a with all four Q fragments resident (Q(3) in block 2: softmax scratch in the leaves, m / d in the pairs' high halves): no Q LDS reads in the tiles |
| f37b | 0.6374 | 2157 |  |  |  | 2213 | 1.410 | 1177 | 749.59 | f37a with --lat WMMA=28,VALU=8 |
| f37c | 0.6401 | 2148 |  |  |  | 2213 | 1.417 | 1177 | 753.99 | f37a with --lat WMMA=36,VALU=8 |
| f37d | 0.6372 | 2158 |  |  |  | 2213 | 1.410 | 1177 | 750.36 | f37a with --lat WMMA=32,VALU=6 |
| f37e | 0.6443 | 2134 |  |  |  | 2212 | 1.425 | 1177 | 758.12 | f37a with --lat WMMA=32,VALU=10 |
| f37f | 0.6396 | 2149 |  |  |  | 2212 | 1.415 | 1177 | 753.32 | f37a with --sgpr-lat 16 |
| f37g | 0.6427 | 2139 |  |  |  | 2212 | 1.422 | 1177 | 757.88 | f37a with --sgpr-lat 24 |
| f37i | 0.6406 | 2146 |  |  |  | 2212 | 1.417 | 1176 | 754.67 | f37a with --resc-cost 12 |
| f37i#null | 0.6404 | 2147 |  |  |  | 2213 | 1.417 | 1178 | 753.36 | f37a with --resc-cost 12 |
