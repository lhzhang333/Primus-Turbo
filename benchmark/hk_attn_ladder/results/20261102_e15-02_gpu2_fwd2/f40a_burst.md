DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 09:28:31

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7448 | 1846 |  |  |  | 2212 | 1.647 | 1183 | 882.36 |  |
| f37j | 0.6169 | 2229 |  |  |  | 2213 | 1.365 | 1179 | 727.58 | f37a with --lat WMMA=28,VALU=6 |
| f40a | 0.6252 | 2199 |  |  |  | 2213 | 1.383 | 1179 | 738.94 | f37j + O written by TDM stores from the staging slots (no read-back + buffer_store phase) |
| f40a#null | 0.6203 | 2216 |  |  |  | 2213 | 1.373 | 1179 | 732.42 | f37j + O written by TDM stores from the staging slots (no read-back + buffer_store phase) |
