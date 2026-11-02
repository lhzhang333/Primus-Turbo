DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 08:00:03

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8343 | 1648 |  |  |  | 1485 | 1.239 | - | - |  |
| f32a | 0.7298 | 1884 |  |  |  | 1352 | 0.988 | - | - | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads carried in SGPRs) |
| f34a | 0.7370 | 1865 | f32a | 0.991x | 0.980x | 1369 | 1.007 | - | - | f32a scheduled on the m1ds issue model with LDS loads in back-to-back pairs, 12 cycles between pair starts |
| f34b | 0.7360 | 1868 | f34a | 1.001x | 0.999x | 1372 | 1.009 | - | - | f32a scheduled on the m1ds issue model with LDS loads in back-to-back pairs, 10 cycles between pair starts |
| f34c | 0.7353 | 1870 | f34b | 1.001x | 1.002x | 1368 | 1.007 | - | - | f32a scheduled on the m1ds issue model with LDS loads in back-to-back pairs, 16 cycles between pair starts |
| f34c#null | 0.7376 | 1864 |  |  |  | 1370 | 1.010 | - | - | f32a scheduled on the m1ds issue model with LDS loads in back-to-back pairs, 16 cycles between pair starts |
