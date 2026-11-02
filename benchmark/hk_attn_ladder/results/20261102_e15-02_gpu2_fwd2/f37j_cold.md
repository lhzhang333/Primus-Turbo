DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 08:57:39

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8390 | 1638 |  |  |  | 1485 | 1.245 | - | - |  |
| f37j | 0.7230 | 1902 |  |  |  | 1366 | 0.989 | - | - | f37a with --lat WMMA=28,VALU=6 |
| f32a | 0.7301 | 1883 |  |  |  | 1353 | 0.989 | - | - | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads carried in SGPRs) |
| f37d | 0.7221 | 1904 |  |  |  | 1366 | 0.987 | - | - | f37a with --lat WMMA=32,VALU=6 |
| f37d#null | 0.7234 | 1900 |  |  |  | 1366 | 0.989 | - | - | f37a with --lat WMMA=32,VALU=6 |
