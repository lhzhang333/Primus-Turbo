DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 09:27:38

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8372 | 1642 |  |  |  | 1485 | 1.243 | - | - |  |
| f37j | 0.7231 | 1901 |  |  |  | 1366 | 0.990 | - | - | f37a with --lat WMMA=28,VALU=6 |
| f40a | 0.7168 | 1918 |  |  |  | 1368 | 0.983 | - | - | f37j + O written by TDM stores from the staging slots (no read-back + buffer_store phase) |
| f40a#null | 0.7234 | 1900 |  |  |  | 1370 | 0.988 | - | - | f37j + O written by TDM stores from the staging slots (no read-back + buffer_store phase) |
