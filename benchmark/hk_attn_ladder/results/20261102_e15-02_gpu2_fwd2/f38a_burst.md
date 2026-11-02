DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 09:10:02

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7659 | 1795 |  |  |  | 2212 | 1.693 | 1182 | 905.39 |  |
| f37j | 0.6388 | 2152 |  |  |  | 2213 | 1.413 | 1178 | 752.37 | f37a with --lat WMMA=28,VALU=6 |
| f38a | 0.6410 | 2145 |  |  |  | 2213 | 1.418 | 1179 | 755.97 | f37j + TILE_D3F: TILE_D3 and FINAL_A3's math (the last tile's own prefix, 1/d, out, lse) in one region, FINAL_ST stores after the barrier |
| f38a#null | 0.6399 | 2148 |  |  |  | 2213 | 1.416 | 1179 | 754.09 | f37j + TILE_D3F: TILE_D3 and FINAL_A3's math (the last tile's own prefix, 1/d, out, lse) in one region, FINAL_ST stores after the barrier |
