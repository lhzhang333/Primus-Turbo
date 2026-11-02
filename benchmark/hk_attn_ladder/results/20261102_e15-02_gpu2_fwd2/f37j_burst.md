DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 08:58:55

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7571 | 1816 |  |  |  | 2212 | 1.675 | 1182 | 895.51 |  |
| f37j | 0.6305 | 2180 |  |  |  | 2213 | 1.395 | 1179 | 743.61 | f37a with --lat WMMA=28,VALU=6 |
| f32a | 0.6322 | 2175 |  |  |  | 2212 | 1.399 | 1179 | 744.85 | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads carried in SGPRs) |
| f37d | 0.6331 | 2171 |  |  |  | 2213 | 1.401 | 1179 | 745.89 | f37a with --lat WMMA=32,VALU=6 |
| f37d#null | 0.6319 | 2176 |  |  |  | 2213 | 1.398 | 1179 | 744.60 | f37a with --lat WMMA=32,VALU=6 |
