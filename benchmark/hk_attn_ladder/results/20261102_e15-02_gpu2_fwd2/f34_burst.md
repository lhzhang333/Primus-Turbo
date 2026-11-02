DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 08:01:12

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7632 | 1801 |  |  |  | 2212 | 1.688 | 1182 | 902.91 |  |
| f32a | 0.6381 | 2155 |  |  |  | 2213 | 1.412 | 1180 | 753.62 | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads carried in SGPRs) |
| f34a | 0.6479 | 2122 | f32a | 0.985x | 0.985x | 2212 | 1.434 | 1179 | 764.32 | f32a scheduled on the m1ds issue model with LDS loads in back-to-back pairs, 12 cycles between pair starts |
| f34b | 0.6486 | 2120 | f34a | 0.998x | 0.998x | 2212 | 1.435 | 1180 | 765.64 | f32a scheduled on the m1ds issue model with LDS loads in back-to-back pairs, 10 cycles between pair starts |
| f34c | 0.6487 | 2119 | f34b | 1.000x | 1.000x | 2212 | 1.435 | 1179 | 765.72 | f32a scheduled on the m1ds issue model with LDS loads in back-to-back pairs, 16 cycles between pair starts |
| f34c#null | 0.6488 | 2119 |  |  |  | 2212 | 1.435 | 1181 | 765.97 | f32a scheduled on the m1ds issue model with LDS loads in back-to-back pairs, 16 cycles between pair starts |
