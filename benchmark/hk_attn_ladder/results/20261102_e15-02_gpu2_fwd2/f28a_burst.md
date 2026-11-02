DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 05:54:19

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7557 | 1819 |  |  |  | 2212 | 1.671 | 1183 | 893.24 |  |
| f26a | 0.6453 | 2131 |  |  |  | 2212 | 1.427 | 1180 | 762.61 | f25b + sum tree partial sums in the leaf whose VGPR banks differ from the other operands (no accumulate-form bank conflict) |
| f28a | 0.6422 | 2141 | f26a | 1.006x | 1.006x | 2212 | 1.420 | 1181 | 758.00 | f26a with one RESC branch per q-tile pair (q tiles 2, 3 in the prefix; 0, 1 in the body) |
| f28a#null | 0.6407 | 2146 |  |  |  | 2212 | 1.417 | 1180 | 756.45 | f26a with one RESC branch per q-tile pair (q tiles 2, 3 in the prefix; 0, 1 in the body) |
