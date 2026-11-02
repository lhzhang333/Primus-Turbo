DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 05:42:28

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8424 | 1632 |  |  |  | 1485 | 1.250 | - | - |  |
| f25b | 0.7392 | 1860 |  |  |  | 1366 | 1.010 | - | - | f25a + FINAL_A |
| f26a | 0.7396 | 1859 |  |  |  | 1368 | 1.012 | - | - | f25b + sum tree partial sums in the leaf whose VGPR banks differ from the other operands (no accumulate-form bank conflict) |
| f26a#null | 0.7410 | 1855 |  |  |  | 1368 | 1.014 | - | - | f25b + sum tree partial sums in the leaf whose VGPR banks differ from the other operands (no accumulate-form bank conflict) |
