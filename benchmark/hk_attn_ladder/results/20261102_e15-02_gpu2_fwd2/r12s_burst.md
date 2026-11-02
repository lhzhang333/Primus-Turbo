DeepSeek-V3 MLA attention backward (bf16, qk 192 / v 128, MHA 128 heads, causal, SBHD) on gfx1250, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 09:17:01

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 3.2190 | 1110 |  |  |  | 2156 | 6.951 | 1327 | 4260.58 |  |
| r10x | 1.8888 | 1892 |  |  |  | 2156 | 4.071 | 1293 | 2440.35 | r10w + K / V by TDM (-DHKGEN_KV_TDM=1): INIT moves the wave's 16 NKV K / V rows into an LDS staging area with 3 TDM ops (as a ring stage's), clears the accumulators meanwhile, then reads the fragments with 60 ds_load_b128 (was 60 buffer_load_b128 of 16 lines each); LDS +64512 B |
| r12s | 1.8904 | 1891 | r10x | 0.998x | 0.999x | 2154 | 4.072 | 1290 | 2436.75 | r10x with the dS^T stores after the stage's last LDS reads (they overlap the ring barrier's dscnt wait) |
| r12s#null | 1.8841 | 1897 |  |  |  | 2154 | 4.054 | 1296 | 2440.70 | r10x with the dS^T stores after the stage's last LDS reads (they overlap the ring barrier's dscnt wait) |
