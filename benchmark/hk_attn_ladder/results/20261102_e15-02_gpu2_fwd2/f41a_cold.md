DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 09:38:01

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8394 | 1638 |  |  |  | 1483 | 1.247 | - | - |  |
| f37j | 0.7243 | 1898 |  |  |  | 1366 | 0.990 | - | - | f37a with --lat WMMA=28,VALU=6 |
| f41a | 0.7240 | 1899 |  |  |  | 1370 | 0.991 | - | - | f37j + TILE_C advances its in-out K / V fragment bases (no VALU in the shell between TILE_C regions) |
| f41a#null | 0.7236 | 1900 |  |  |  | 1364 | 0.987 | - | - | f37j + TILE_C advances its in-out K / V fragment bases (no VALU in the shell between TILE_C regions) |
