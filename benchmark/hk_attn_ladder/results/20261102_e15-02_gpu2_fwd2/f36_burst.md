DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 08:26:45

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7619 | 1804 |  |  |  | 2212 | 1.685 | 1183 | 901.29 |  |
| f32a | 0.6376 | 2156 |  |  |  | 2212 | 1.410 | 1181 | 751.73 | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads carried in SGPRs) |
| f36x | 0.6219 | 2211 |  |  |  | 2214 | 1.377 | 1177 | 730.79 | timing ablation (wrong results): f32a without the per-tile Q(0) / Q(3) LDS reads (Q resident) |
| f36x#null | 0.6218 | 2211 |  |  |  | 2214 | 1.377 | 1176 | 731.13 | timing ablation (wrong results): f32a without the per-tile Q(0) / Q(3) LDS reads (Q resident) |
