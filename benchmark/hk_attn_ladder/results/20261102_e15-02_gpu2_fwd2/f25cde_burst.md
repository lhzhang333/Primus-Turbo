DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 05:24:57

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7636 | 1800 |  |  |  | 2212 | 1.689 | 1184 | 902.89 |  |
| f25b | 0.6509 | 2112 |  |  |  | 2212 | 1.440 | 1179 | 767.07 | f25a + FINAL_A |
| f25c | 0.6732 | 2042 | f25b | 0.966x | 0.966x | 2212 | 1.489 | 1180 | 795.37 | f25b + m1ds issue model with LDS bursts (DS_RATE 1; an LDS op holds non-LDS issue 4 cycles) |
| f25d | 0.6692 | 2054 | f25b | 0.972x | 0.973x | 2212 | 1.480 | 1180 | 789.27 | f25b + m1ds with DS_RATE 2 |
| f25e | 0.6776 | 2029 | f25b | 0.961x | 0.961x | 2212 | 1.499 | 1180 | 800.40 | f25b + m1ds with LDS block 3, DS_RATE 1 |
| f25e#null | 0.6786 | 2026 |  |  |  | 2212 | 1.501 | 1181 | 801.84 | f25b + m1ds with LDS block 3, DS_RATE 1 |
