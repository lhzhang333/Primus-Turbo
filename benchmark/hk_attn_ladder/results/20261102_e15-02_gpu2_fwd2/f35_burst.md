DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 08:08:28

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7599 | 1809 |  |  |  | 2212 | 1.681 | 1183 | 899.68 |  |
| f32a | 0.6360 | 2162 |  |  |  | 2212 | 1.407 | 1180 | 751.74 | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads carried in SGPRs) |
| f35x | 0.6366 | 2160 |  |  |  | 2212 | 1.408 | 1180 | 751.70 | timing ablation (wrong results): f32a with no VGPR bank conflicts in the packed fma (m') and the RESC multiply (corr) |
| f35r8 | 0.6411 | 2144 | f32a | 0.991x | 0.991x | 2212 | 1.418 | 1181 | 760.29 | f32a with at most one LDS load per 8 modelled cycles |
| f35r8#null | 0.6413 | 2144 |  |  |  | 2212 | 1.419 | 1181 | 760.39 | f32a with at most one LDS load per 8 modelled cycles |
