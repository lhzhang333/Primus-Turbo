DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 09:31:36

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7664 | 1794 |  |  |  | 2212 | 1.695 | 1183 | 906.68 |  |
| f40a | 0.6407 | 2146 |  |  |  | 2213 | 1.418 | 1179 | 755.53 | f37j + O written by TDM stores from the staging slots (no read-back + buffer_store phase) |
| f37j | 0.6400 | 2148 |  |  |  | 2213 | 1.416 | 1178 | 753.75 | f37a with --lat WMMA=28,VALU=6 |
| f37j#null | 0.6382 | 2154 |  |  |  | 2213 | 1.412 | 1179 | 754.07 | f37a with --lat WMMA=28,VALU=6 |
