DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 09:21:02

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7587 | 1812 |  |  |  | 2212 | 1.678 | 1183 | 897.50 |  |
| f37j | 0.6329 | 2172 |  |  |  | 2213 | 1.400 | 1181 | 747.17 | f37a with --lat WMMA=28,VALU=6 |
| f39x | 0.6278 | 2190 |  |  |  | 2213 | 1.389 | 1178 | 738.29 | timing ablation (no output): f37j without the epilogue's global stores |
| f39x#null | 0.6271 | 2192 |  |  |  | 2213 | 1.388 | 1177 | 738.34 | timing ablation (no output): f37j without the epilogue's global stores |
