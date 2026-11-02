DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 05:49:23

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7612 | 1806 |  |  |  | 2212 | 1.683 | 1182 | 902.48 |  |
| f26a | 0.6435 | 2136 |  |  |  | 2212 | 1.424 | 1182 | 766.19 | f25b + sum tree partial sums in the leaf whose VGPR banks differ from the other operands (no accumulate-form bank conflict) |
| f27a | 0.6650 | 2067 |  |  |  | 2211 | 1.470 | 1181 | 791.08 | f26a with O *= corr on every tile (no RESC branches) |
| f27b | 0.6487 | 2119 |  |  |  | 2212 | 1.435 | 1181 | 766.87 | f26a with at most one LDS load per 8 cycles in the model |
| f27c | 0.6625 | 2075 |  |  |  | 2211 | 1.465 | 1182 | 785.42 | f27a + --ds-rate 8 |
| f27d | 0.6500 | 2115 |  |  |  | 2212 | 1.438 | 1182 | 769.92 | f26a with the rescale decision and O *= corr out of line (rare path) |
| f27d#null | 0.6562 | 2095 |  |  |  | 2212 | 1.452 | 1180 | 775.38 | f26a with the rescale decision and O *= corr out of line (rare path) |
