DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 08:54:03

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8455 | 1626 |  |  |  | 1484 | 1.256 | - | - |  |
| f32a | 0.7301 | 1883 |  |  |  | 1350 | 0.985 | - | - | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads carried in SGPRs) |
| f37b | 0.7264 | 1892 |  |  |  | 1366 | 0.991 | - | - | f37a with --lat WMMA=28,VALU=8 |
| f37d | 0.7233 | 1901 |  |  |  | 1364 | 0.985 | - | - | f37a with --lat WMMA=32,VALU=6 |
| f37j | 0.7195 | 1911 |  |  |  | 1364 | 0.983 | - | - | f37a with --lat WMMA=28,VALU=6 |
| f37k | 0.7271 | 1891 |  |  |  | 1364 | 0.991 | - | - | f37a with --lat WMMA=24,VALU=8 |
| f37m | 0.7342 | 1872 |  |  |  | 1359 | 0.996 | - | - | f32a (no --qres) with --lat WMMA=28,VALU=8 |
| f37n | 0.7315 | 1879 |  |  |  | 1351 | 0.988 | - | - | f32a (no --qres) with --lat WMMA=32,VALU=6 |
| f37n#null | 0.7323 | 1877 |  |  |  | 1348 | 0.989 | - | - | f32a (no --qres) with --lat WMMA=32,VALU=6 |
