DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 08:49:12

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8401 | 1636 |  |  |  | 1482 | 1.244 | - | - |  |
| f32a | 0.7316 | 1879 |  |  |  | 1351 | 0.987 | - | - | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads carried in SGPRs) |
| f37a | 0.7273 | 1890 | f32a | 1.007x | 0.997x | 1366 | 0.992 | - | - | f32a with all four Q fragments resident (Q(3) in block 2: softmax scratch in the leaves, m / d in the pairs' high halves): no Q LDS reads in the tiles |
| f37b | 0.7235 | 1900 |  |  |  | 1362 | 0.989 | - | - | f37a with --lat WMMA=28,VALU=8 |
| f37c | 0.7289 | 1886 |  |  |  | 1376 | 1.001 | - | - | f37a with --lat WMMA=36,VALU=8 |
| f37d | 0.7250 | 1896 |  |  |  | 1364 | 0.989 | - | - | f37a with --lat WMMA=32,VALU=6 |
| f37e | 0.7335 | 1874 |  |  |  | 1376 | 1.008 | - | - | f37a with --lat WMMA=32,VALU=10 |
| f37f | 0.7274 | 1890 |  |  |  | 1370 | 0.995 | - | - | f37a with --sgpr-lat 16 |
| f37g | 0.7294 | 1885 |  |  |  | 1376 | 1.002 | - | - | f37a with --sgpr-lat 24 |
| f37i | 0.7270 | 1891 |  |  |  | 1368 | 0.994 | - | - | f37a with --resc-cost 12 |
| f37i#null | 0.7260 | 1893 |  |  |  | 1368 | 0.993 | - | - | f37a with --resc-cost 12 |
