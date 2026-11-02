DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 05:06:46

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8400 | 1637 |  |  |  | 1485 | 1.249 | - | - |  |
| f22a | 0.7471 | 1840 |  |  |  | 1375 | 1.027 | - | - | f19f + per-q-tile O resources with 32-bit bounds (no 64-bit clamp chains in the epilogue) |
| f24a | 0.7421 | 1852 | f22a | 1.007x | 1.011x | 1369 | 1.015 | - | - | FINAL_A: one scheduled region -- the last prefix, then out / lse of all 4 q tiles stored to the staging by the generated code |
| f25a | 0.7429 | 1851 | f24a | 0.999x | 0.995x | 1374 | 1.021 | - | - | f22a + K / V slot descriptors bounded at the tile's 64 rows (fewer SALU per TDM op) |
| f25b | 0.7387 | 1861 |  |  |  | 1364 | 1.006 | - | - | f25a + FINAL_A |
| f25b#null | 0.7404 | 1857 |  |  |  | 1366 | 1.010 | - | - | f25a + FINAL_A |
