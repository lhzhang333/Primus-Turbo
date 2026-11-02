DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 08:07:30

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8392 | 1638 |  |  |  | 1483 | 1.248 | - | - |  |
| f32a | 0.7298 | 1884 |  |  |  | 1352 | 0.987 | - | - | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads carried in SGPRs) |
| f35x | 0.7273 | 1890 |  |  |  | 1358 | 0.986 | - | - | timing ablation (wrong results): f32a with no VGPR bank conflicts in the packed fma (m') and the RESC multiply (corr) |
| f35r8 | 0.7353 | 1869 | f32a | 0.993x | 0.987x | 1361 | 1.000 | - | - | f32a with at most one LDS load per 8 modelled cycles |
| f35r8#null | 0.7355 | 1869 |  |  |  | 1360 | 1.001 | - | - | f32a with at most one LDS load per 8 modelled cycles |
