DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 08:25:38

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8426 | 1631 |  |  |  | 1486 | 1.251 | - | - |  |
| f32a | 0.7273 | 1890 |  |  |  | 1352 | 0.986 | - | - | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads carried in SGPRs) |
| f36x | 0.6937 | 1982 |  |  |  | 1410 | 0.977 | - | - | timing ablation (wrong results): f32a without the per-tile Q(0) / Q(3) LDS reads (Q resident) |
| f36x#null | 0.6948 | 1979 |  |  |  | 1413 | 0.981 | - | - | timing ablation (wrong results): f32a without the per-tile Q(0) / Q(3) LDS reads (Q resident) |
