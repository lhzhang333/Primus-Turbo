DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 09:20:16

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8349 | 1647 |  |  |  | 1484 | 1.239 | - | - |  |
| f37j | 0.7227 | 1902 |  |  |  | 1366 | 0.987 | - | - | f37a with --lat WMMA=28,VALU=6 |
| f39x | 0.7107 | 1934 |  |  |  | 1378 | 0.979 | - | - | timing ablation (no output): f37j without the epilogue's global stores |
| f39x#null | 0.7088 | 1939 |  |  |  | 1379 | 0.975 | - | - | timing ablation (no output): f37j without the epilogue's global stores |
