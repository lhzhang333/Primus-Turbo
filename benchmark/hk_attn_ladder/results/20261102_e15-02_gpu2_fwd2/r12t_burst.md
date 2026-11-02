DeepSeek-V3 MLA attention backward (bf16, qk 192 / v 128, MHA 128 heads, causal, SBHD) on gfx1250, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-11-02 08:35:28

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 3.2107 | 1113 |  |  |  | 2155 | 6.921 | 1327 | 4262.28 |  |
| r10x | 1.8826 | 1899 |  |  |  | 2150 | 4.047 | 1292 | 2427.06 | r10w + K / V by TDM (-DHKGEN_KV_TDM=1): INIT moves the wave's 16 NKV K / V rows into an LDS staging area with 3 TDM ops (as a ring stage's), clears the accumulators meanwhile, then reads the fragments with 60 ds_load_b128 (was 60 buffer_load_b128 of 16 lines each); LDS +64512 B |
| r12t4 | 1.8878 | 1893 | r10x | 0.998x | 0.999x | 2150 | 4.060 | 1292 | 2428.33 | r10x with a ticket only for the last 1/4 of the dK/dV workgroups (the rest run their own linear id: no entry atomic) |
| r12t8 | 1.8731 | 1908 | r12t4 | 1.006x | 1.006x | 2154 | 4.034 | 1292 | 2421.59 | r10x with a ticket only for the last 1/8 of the dK/dV workgroups (the rest run their own linear id: no entry atomic) |
| r12t16 | 1.9011 | 1880 | r12t8 | 0.989x | 0.987x | 2152 | 4.087 | 1294 | 2445.69 | r10x with a ticket only for the last 1/16 of the dK/dV workgroups (the rest run their own linear id: no entry atomic) |
| r12t16#null | 1.8890 | 1892 |  |  |  | 2155 | 4.079 | 1292 | 2447.67 | r10x with a ticket only for the last 1/16 of the dK/dV workgroups (the rest run their own linear id: no entry atomic) |
