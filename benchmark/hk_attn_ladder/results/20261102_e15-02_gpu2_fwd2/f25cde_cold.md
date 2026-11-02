DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 05:23:57

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8340 | 1648 |  |  |  | 1484 | 1.239 | - | - |  |
| f25b | 0.7386 | 1861 |  |  |  | 1366 | 1.008 | - | - | f25a + FINAL_A |
| f25c | 0.7582 | 1813 | f25b | 0.976x | 0.950x | 1398 | 1.060 | - | - | f25b + m1ds issue model with LDS bursts (DS_RATE 1; an LDS op holds non-LDS issue 4 cycles) |
| f25d | 0.7540 | 1823 | f25b | 0.980x | 0.959x | 1394 | 1.050 | - | - | f25b + m1ds with DS_RATE 2 |
| f25e | 0.7599 | 1809 | f25b | 0.971x | 0.937x | 1411 | 1.073 | - | - | f25b + m1ds with LDS block 3, DS_RATE 1 |
| f25e#null | 0.7618 | 1805 |  |  |  | 1409 | 1.075 | - | - | f25b + m1ds with LDS block 3, DS_RATE 1 |
