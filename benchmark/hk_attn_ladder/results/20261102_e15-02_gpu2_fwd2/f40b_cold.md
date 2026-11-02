DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 09:30:26

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8372 | 1642 |  |  |  | 1483 | 1.242 | - | - |  |
| f40a | 0.7242 | 1898 |  |  |  | 1366 | 0.989 | - | - | f37j + O written by TDM stores from the staging slots (no read-back + buffer_store phase) |
| f37j | 0.7253 | 1895 |  |  |  | 1367 | 0.991 | - | - | f37a with --lat WMMA=28,VALU=6 |
| f37j#null | 0.7240 | 1899 |  |  |  | 1364 | 0.987 | - | - | f37a with --lat WMMA=28,VALU=6 |
