DeepSeek-V3 MLA attention backward (bf16, qk 192 / v 128, MHA 128 heads, causal, SBHD) on gfx1250, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 09:16:11

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 3.4823 | 1026 |  |  |  | 1566 | 5.448 | - | - |  |
| r10x | 2.2865 | 1563 |  |  |  | 1219 | 2.784 | - | - | r10w + K / V by TDM (-DHKGEN_KV_TDM=1): INIT moves the wave's 16 NKV K / V rows into an LDS staging area with 3 TDM ops (as a ring stage's), clears the accumulators meanwhile, then reads the fragments with 60 ds_load_b128 (was 60 buffer_load_b128 of 16 lines each); LDS +64512 B |
| r12s | 2.2799 | 1568 | r10x | 1.003x | 1.007x | 1215 | 2.771 | - | - | r10x with the dS^T stores after the stage's last LDS reads (they overlap the ring barrier's dscnt wait) |
| r12s#null | 2.2835 | 1565 |  |  |  | 1219 | 2.784 | - | - | r10x with the dS^T stores after the stage's last LDS reads (they overlap the ring barrier's dscnt wait) |
