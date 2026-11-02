DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 09:48:15

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7651 | 1797 |  |  |  | 2212 | 1.692 | 1180 | 903.08 |  |
| f37j | 0.6379 | 2155 |  |  |  | 2213 | 1.412 | 1177 | 752.05 | f37a with --lat WMMA=28,VALU=6 |
| f41a | 0.6384 | 2153 |  |  |  | 2213 | 1.413 | 1177 | 751.81 | f37j + TILE_C advances its in-out K / V fragment bases (no VALU in the shell between TILE_C regions) |
| f42a | 0.6407 | 2146 | f41a | 0.998x | 0.998x | 2213 | 1.418 | 1178 | 754.35 | f41a + TILE_C emitted from a known entry state (drain before the tile loop; no full drain at every region entry) |
| f42a#null | 0.6402 | 2147 |  |  |  | 2213 | 1.417 | 1178 | 753.76 | f41a + TILE_C emitted from a known entry state (drain before the tile loop; no full drain at every region entry) |
