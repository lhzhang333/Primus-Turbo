DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 05:57:52

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8410 | 1635 |  |  |  | 1485 | 1.249 | - | - |  |
| f26a | 0.7422 | 1852 |  |  |  | 1366 | 1.012 | - | - | f25b + sum tree partial sums in the leaf whose VGPR banks differ from the other operands (no accumulate-form bank conflict) |
| f27a | 0.7624 | 1803 |  |  |  | 1369 | 1.045 | - | - | f26a with O *= corr on every tile (no RESC branches) |
| f28a | 0.7398 | 1858 | f26a | 1.003x | 1.009x | 1358 | 1.005 | - | - | f26a with one RESC branch per q-tile pair (q tiles 2, 3 in the prefix; 0, 1 in the body) |
| f29x | 0.7454 | 1844 |  |  |  | 1336 | 0.997 | - | - | timing ablation (wrong results): f27a with the O *= corr multiplier from an SGPR pair (no VGPR bank conflicts) |
| f29x#null | 0.7481 | 1838 |  |  |  | 1340 | 1.003 | - | - | timing ablation (wrong results): f27a with the O *= corr multiplier from an SGPR pair (no VGPR bank conflicts) |
