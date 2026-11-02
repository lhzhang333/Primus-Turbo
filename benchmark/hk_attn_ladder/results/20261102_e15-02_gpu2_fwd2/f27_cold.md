DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 05:48:12

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8410 | 1635 |  |  |  | 1484 | 1.248 | - | - |  |
| f26a | 0.7398 | 1858 |  |  |  | 1366 | 1.009 | - | - | f25b + sum tree partial sums in the leaf whose VGPR banks differ from the other operands (no accumulate-form bank conflict) |
| f27a | 0.7660 | 1795 |  |  |  | 1374 | 1.051 | - | - | f26a with O *= corr on every tile (no RESC branches) |
| f27b | 0.7416 | 1854 |  |  |  | 1364 | 1.012 | - | - | f26a with at most one LDS load per 8 cycles in the model |
| f27c | 0.7658 | 1795 |  |  |  | 1370 | 1.047 | - | - | f27a + --ds-rate 8 |
| f27d | 0.7463 | 1842 |  |  |  | 1387 | 1.035 | - | - | f26a with the rescale decision and O *= corr out of line (rare path) |
| f27d#null | 0.7477 | 1839 |  |  |  | 1384 | 1.035 | - | - | f26a with the rescale decision and O *= corr out of line (rare path) |
