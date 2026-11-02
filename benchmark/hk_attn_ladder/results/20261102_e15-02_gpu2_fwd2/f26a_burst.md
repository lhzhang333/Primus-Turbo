DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 05:43:17

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7575 | 1815 |  |  |  | 2211 | 1.675 | 1184 | 896.35 |  |
| f25b | 0.6463 | 2127 |  |  |  | 2212 | 1.430 | 1179 | 762.99 | f25a + FINAL_A |
| f26a | 0.6475 | 2123 |  |  |  | 2212 | 1.432 | 1180 | 763.03 | f25b + sum tree partial sums in the leaf whose VGPR banks differ from the other operands (no accumulate-form bank conflict) |
| f26a#null | 0.6476 | 2123 |  |  |  | 2212 | 1.432 | 1182 | 764.93 | f25b + sum tree partial sums in the leaf whose VGPR banks differ from the other operands (no accumulate-form bank conflict) |
