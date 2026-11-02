DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 09:07:24

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8411 | 1635 |  |  |  | 1486 | 1.249 | - | - |  |
| f37j | 0.7221 | 1904 |  |  |  | 1370 | 0.989 | - | - | f37a with --lat WMMA=28,VALU=6 |
| f38a | 0.7255 | 1895 |  |  |  | 1373 | 0.998 | - | - | f37j + TILE_D3F: TILE_D3 and FINAL_A3's math (the last tile's own prefix, 1/d, out, lse) in one region, FINAL_ST stores after the barrier |
| f38a#null | 0.7242 | 1898 |  |  |  | 1370 | 0.992 | - | - | f37j + TILE_D3F: TILE_D3 and FINAL_A3's math (the last tile's own prefix, 1/d, out, lse) in one region, FINAL_ST stores after the barrier |
