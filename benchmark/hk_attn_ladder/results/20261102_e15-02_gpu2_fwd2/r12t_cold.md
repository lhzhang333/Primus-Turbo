DeepSeek-V3 MLA attention backward (bf16, qk 192 / v 128, MHA 128 heads, causal, SBHD) on gfx1250, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-11-02 08:34:24

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 3.4900 | 1024 |  |  |  | 1560 | 5.444 | - | - |  |
| r10x | 2.2807 | 1567 |  |  |  | 1214 | 2.769 | - | - | r10w + K / V by TDM (-DHKGEN_KV_TDM=1): INIT moves the wave's 16 NKV K / V rows into an LDS staging area with 3 TDM ops (as a ring stage's), clears the accumulators meanwhile, then reads the fragments with 60 ds_load_b128 (was 60 buffer_load_b128 of 16 lines each); LDS +64512 B |
| r12t4 | 2.2798 | 1568 | r10x | 1.002x | 0.994x | 1224 | 2.786 | - | - | r10x with a ticket only for the last 1/4 of the dK/dV workgroups (the rest run their own linear id: no entry atomic) |
| r12t8 | 2.2757 | 1571 | r12t4 | 1.002x | 1.001x | 1223 | 2.781 | - | - | r10x with a ticket only for the last 1/8 of the dK/dV workgroups (the rest run their own linear id: no entry atomic) |
| r12t16 | 2.2996 | 1554 | r12t8 | 0.989x | 0.962x | 1258 | 2.890 | - | - | r10x with a ticket only for the last 1/16 of the dK/dV workgroups (the rest run their own linear id: no entry atomic) |
| r12t16#null | 2.3003 | 1554 |  |  |  | 1256 | 2.890 | - | - | r10x with a ticket only for the last 1/16 of the dK/dV workgroups (the rest run their own linear id: no entry atomic) |
