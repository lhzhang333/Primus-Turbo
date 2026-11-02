DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 09:39:12

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7629 | 1802 |  |  |  | 2212 | 1.687 | 1183 | 901.39 |  |
| f37j | 0.6392 | 2151 |  |  |  | 2213 | 1.415 | 1179 | 752.52 | f37a with --lat WMMA=28,VALU=6 |
| f41a | 0.6363 | 2160 |  |  |  | 2213 | 1.408 | 1179 | 749.87 | f37j + TILE_C advances its in-out K / V fragment bases (no VALU in the shell between TILE_C regions) |
| f41a#null | 0.6359 | 2162 |  |  |  | 2213 | 1.407 | 1179 | 750.52 | f37j + TILE_C advances its in-out K / V fragment bases (no VALU in the shell between TILE_C regions) |
