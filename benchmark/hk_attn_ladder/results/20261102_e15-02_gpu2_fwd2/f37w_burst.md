DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 08:55:45

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7639 | 1800 |  |  |  | 2212 | 1.690 | 1182 | 901.79 |  |
| f32a | 0.6388 | 2152 |  |  |  | 2213 | 1.413 | 1180 | 753.17 | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads carried in SGPRs) |
| f37b | 0.6384 | 2153 |  |  |  | 2213 | 1.412 | 1180 | 751.83 | f37a with --lat WMMA=28,VALU=8 |
| f37d | 0.6369 | 2158 |  |  |  | 2213 | 1.409 | 1178 | 750.49 | f37a with --lat WMMA=32,VALU=6 |
| f37j | 0.6361 | 2161 |  |  |  | 2213 | 1.408 | 1178 | 749.96 | f37a with --lat WMMA=28,VALU=6 |
| f37k | 0.6378 | 2155 |  |  |  | 2213 | 1.411 | 1178 | 752.11 | f37a with --lat WMMA=24,VALU=8 |
| f37m | 0.6425 | 2140 |  |  |  | 2212 | 1.421 | 1180 | 761.13 | f32a (no --qres) with --lat WMMA=28,VALU=8 |
| f37n | 0.6392 | 2151 |  |  |  | 2212 | 1.414 | 1180 | 754.84 | f32a (no --qres) with --lat WMMA=32,VALU=6 |
| f37n#null | 0.6397 | 2149 |  |  |  | 2212 | 1.415 | 1180 | 755.03 | f32a (no --qres) with --lat WMMA=32,VALU=6 |
