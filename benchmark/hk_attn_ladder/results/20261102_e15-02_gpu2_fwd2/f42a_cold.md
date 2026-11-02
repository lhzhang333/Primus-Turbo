DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 09:47:02

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8360 | 1644 |  |  |  | 1483 | 1.241 | - | - |  |
| f37j | 0.7224 | 1903 |  |  |  | 1366 | 0.985 | - | - | f37a with --lat WMMA=28,VALU=6 |
| f41a | 0.7247 | 1897 |  |  |  | 1362 | 0.988 | - | - | f37j + TILE_C advances its in-out K / V fragment bases (no VALU in the shell between TILE_C regions) |
| f42a | 0.7257 | 1894 | f41a | 0.999x | 0.998x | 1366 | 0.991 | - | - | f41a + TILE_C emitted from a known entry state (drain before the tile loop; no full drain at every region entry) |
| f42a#null | 0.7252 | 1896 |  |  |  | 1366 | 0.992 | - | - | f41a + TILE_C emitted from a known entry state (drain before the tile loop; no full drain at every region entry) |
