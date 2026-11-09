// One step of k_attn_bwd_main (k_attn_bwd.hip), included textually (a) by the `step` lambda (the
// diagonal steps) and (b) at kernel scope (the full steps, BWD_KERNEL_SCOPE 1), where the register
// holds (llvm.fake.use) below take effect: the inliner drops the fake uses of an inlined callee.
// The includer provides constexpr bool MASKED, MDEAD and constexpr int P4, P, NX; int s (the step);
// XSet &X (the operands of tile s on entry, of tile s + 1 on exit).
        constexpr bool live = !(MASKED && MDEAD); // false: a diagonal step's DEAD-wave body
        using Z0  = std::integral_constant<int, 0>;
        using SLN = std::integral_constant<int, NX>;
        const int qt = bwd::tile_qt(s, n, G, nqp);
#if BWD_KERNEL_SCOPE
        acc_t      S[2][2], dP[2][2];
        acc_t (&dq)[2][2] = dq_c; // loop-carried: dQ(s - 1)'s registers stay allocated through G0 of step s + 1
        u32x4      pu[2][2];
        u32x4 (&du)[2][2] = du_c; // loop-carried: E2's operands stay allocated through G0 of step s + 1
        opnd_t     dOT[8];
        opnd_t (&QT)[8] = qt_c;
#else
        acc_t      S[2][2], dP[2][2], dq[2][2];
        u32x4      pu[2][2], du[2][2];
        opnd_t     QT[8], dOT[8];
#endif
        auto sm = [&](auto hh, auto kh) { // softmax and dS of block (hh, kh)
            acc_t t = softmax_arg_rows(S[hh][kh], c1, X.ld[hh][0], X.ld[hh][2]);
            if constexpr (MASKED) t = mask_col_gt_row(t, l, bwd::mask_off(qt, hh, kvw0, w, kh, cshift), MASK_NEG);
            const acc_t p = exp2_acc(t), d = dsoftmax(p, dP[hh][kh], X.ld[hh][1], X.ld[hh][3]);
            pu[hh][kh] = cvt_pk_bf16(p);
            du[hh][kh] = cvt_pk_bf16(d);
        };
        auto sdp = [&](auto kh) { // S / dP of kv tile kh: 4 independent chains over dt
            static_for<4>([&](auto dt) {
                static_for<2>([&](auto hh) {
                    if constexpr (dt == 0) S[hh][kh] = wmma0(X.qA[hh][dt], kf[kh][dt]);
                    else S[hh][kh] = wmma(X.qA[hh][dt], kf[kh][dt], S[hh][kh]);
                });
                static_for<2>([&](auto hh) {
                    if constexpr (dt == 0) dP[hh][kh] = wmma0(X.dA[hh][dt], vf[kh][dt]);
                    else dP[hh][kh] = wmma(X.dA[hh][dt], vf[kh][dt], dP[hh][kh]);
                });
            });
        };
        auto load_dot = [&] { static_for<8>([&](auto dt) { dOT[dt] = load_tr<XP>(do_tr + P4 * bwd::QT_B, 0, 16 * dt); }); };
        auto load_qt  = [&] { static_for<8>([&](auto dt) { QT[dt] = load_tr<XP>(q_tr + P4 * bwd::QT_B, 0, 16 * dt); }); };
        // E WMMAs [LO, HI), kv-tile-major (an A operand dies only in the second kv-tile pass): e < 16
        // dV^T[dt][kh] += dOT[dt] P[kh] with kh = e / 8, dt = e % 8; e >= 16 dK^T[dt][kh] += QT[dt] dS[kh]
        auto e_wmma = [&](auto lo_c, auto hi_c) __attribute__((always_inline)) {
            constexpr int LO = decltype(lo_c)::value, HI = decltype(hi_c)::value;
            static_for<bwd::NE>([&](auto e) {
                if constexpr ((int) e >= LO && (int) e < HI) {
                    constexpr int kh = (e % 16) / 8, dt = e % 8;
                    if constexpr (e < 16) dVt[dt][kh] = wmma(dOT[dt], cat_u32(pu[0][kh], pu[1][kh]), dVt[dt][kh]);
                    else dKt[dt][kh] = wmma(QT[dt], cat_u32(du[0][kh], du[1][kh]), dKt[dt][kh]);
                }
            });
        };

        // T: ring op s + 3 (group 0 built under E2 of step s - 1); wave 3 issues its dummy op
        tdm_load(g0n, rg1);
        sched_barrier<0>();
        // G0: S/dP kv tile 0 (its A operands do not die here) || dSa(s - 1) (buffer P ^ 1), dOT(s)
        if constexpr (live) {
            sdp(I0);
            load_dsa(std::integral_constant<int, P ^ 1>{});
            load_dot();
            pattern<16, 0, 0, 16 + 8 * bwd::DQ_NQH>(); // 16 dOT + 8 dSa tr16
        } else {
            load_dsa(std::integral_constant<int, P ^ 1>{});
            sched_barrier<0>();
        }
        sched_barrier<0>();
#if BWD_KERNEL_SCOPE
        HOLD_2X2(dq); // dQ(s - 1)'s atomic data and voffset: not reused by the G0 loads (vm_vsrc WAR)
        hka_llvm_fake_use(avo_c);
        HOLD_8(QT); // step s - 1's E2 operands: not reused by the G0 loads (WAR on the last E2 WMMAs)
        HOLD_2X2(du);
#endif
        // G1: S/dP kv tile 1 || softmax kv tile 0
        if constexpr (live) {
            sdp(I1);
            sm(I0, I0);
            sm(I1, I0);
            pattern<16, 56 + (MASKED ? bwd::MASK_VALU : 0), 0, 0>();
        }
        sched_barrier<0>();
#if BWD_KERNEL_SCOPE
        HOLD_2X4(X.qA); // G1's WMMA A operands: the softmax temporaries go elsewhere (no WAR v_nop)
        HOLD_2X4(X.dA);
#endif
        // C: dQ(s - 1) || softmax kv tile 1; the atomic voffset of tile s - 1
        dq_wmma(dq);
        if constexpr (live) {
            sm(I0, I1);
            sm(I1, I1);
        } else {
            static_for<2>([&](auto hh) { static_for<2>([&](auto kh) { du[hh][kh] = zero_u32x4(); }); });
        }
#if BWD_KERNEL_SCOPE
        avo_c = voff_nonneg(voff_lane + bwd::dqa_voff(0, 0, bwd::atom_gh(s, n, G, rot), bwd::atom_qt(s, n, G, nqp), nqp));
        const int avo = avo_c;
#else
        const int avo = voff_nonneg(voff_lane + bwd::dqa_voff(0, 0, bwd::atom_gh(s, n, G, rot), bwd::atom_qt(s, n, G, nqp), nqp));
#endif
        // QT(s) (ring slot s, published since the step began) under the dQ WMMAs, 2 per gap in gaps 0..7
        if constexpr (live) {
            load_qt();
            constexpr int XV = 56 + (MASKED ? bwd::MASK_VALU : 0);
            static_for<16>([&](auto g) {
                sched_group_barrier<sg::WMMA, 1, 0>();
                if constexpr ((int) g >= 2) {
                    constexpr int nv = bwd::share(XV, 14, (int) g - 2);
                    if constexpr (nv > 0) sched_group_barrier<sg::VALU | sg::TRANS, nv, 0>();
                }
                if constexpr ((int) g < 8) sched_group_barrier<sg::DS_READ, 2, 0>();
            });
        }
        order_fence(); // every load above stays older than the stores
#if BWD_KERNEL_SCOPE
        hka_llvm_fake_use(dSa[0][0]);
        hka_llvm_fake_use(dSa[0][1]);
        hka_llvm_fake_use(dSa[0][2]);
        hka_llvm_fake_use(dSa[0][3]);
#endif
        // the LDS loads issued before the QT loads (dSa, dOT in G0) are complete here: one explicit wait
        // instead of a wait per E1 WMMA; the 16 QT loads (the youngest) may stay in flight
        if constexpr (live) wait_dscnt<16>();
        // ST: dS^T(s) -> buffer P
        static_for<2>([&](auto kh) {
            static_for<2>([&](auto hh) { ds_store_b128(ds_st, P * bwd::DS_B + bwd::ct_imm(SP, 16 * kh, 16 * hh), du[hh][kh]); });
        });
        order_fence(); // the stores stay older than every later LDS op
        // E1: E WMMAs 0..NE1-1 || dQ(s - 1) atomics + read-back items 0..RB1-1 (DEAD waves: the atomics as a burst)
        if constexpr (!live) {
            atomics(Z0{}, std::integral_constant<int, bwd::NATOM>{}, dq, avo);
            sched_barrier<0>();
        } else {
            constexpr int RB1 = bwd::RB1, AT1 = bwd::ATOM_E1;
            // source order == issue order per gap (atomics are ordered memory ops: the LDS loads cannot pass them)
            static_for<bwd::NE1>([&](auto g) {
                e_wmma(g, std::integral_constant<int, g + 1>{});
                constexpr int a0 = (AT1 * g) / bwd::NE1, a1 = (AT1 * (g + 1)) / bwd::NE1;
                atomics(std::integral_constant<int, a0>{}, std::integral_constant<int, a1>{}, dq, avo);
                if constexpr (g >= 8) {
                    constexpr int r0 = (RB1 * (g - 8)) / (bwd::NE1 - 8), r1 = (RB1 * (g - 7)) / (bwd::NE1 - 8);
                    readback(std::integral_constant<int, r0>{}, std::integral_constant<int, r1>{}, SLN{}, X);
                }
            });
            static_for<bwd::NE1>([&](auto g) {
                sched_group_barrier<sg::WMMA, 1, 0>();
                constexpr int na = (AT1 * (g + 1)) / bwd::NE1 - (AT1 * g) / bwd::NE1;
                if constexpr (na > 0) sched_group_barrier<sg::VMEM, na, 0>();
                if constexpr (g >= 8) {
                    constexpr int nr = bwd::rb_loads((RB1 * (g - 7)) / (bwd::NE1 - 8)) - bwd::rb_loads((RB1 * (g - 8)) / (bwd::NE1 - 8));
                    if constexpr (nr > 0) sched_group_barrier<sg::DS_READ, nr, 0>();
                }
            });
        }
        // SIG: QT(s) and the stores retire (the NUNC read-back loads may stay in flight)
        if constexpr (live) split_signal_tdm_dscnt<bwd::TCNT, bwd::NUNC>();
        else split_signal_tdm_lds<bwd::TCNT>();
        // E2: E WMMAs NE1..31 || read-back items RB1..23 + the other atomics; group 0 of op(s + 4)
        g0n = ring_g0(s + 1 + bwd::LEAD);
        asm volatile("" ::"s"(g0n)); // materialize under E2 (IR sinking would move it behind the wait)
        if constexpr (!live) {
            readback(Z0{}, std::integral_constant<int, bwd::RB_ITEMS>{}, SLN{}, X);
            sched_barrier<0>();
        } else {
            constexpr int RB1 = bwd::RB1, NR = bwd::RB_ITEMS - RB1, AT1 = bwd::ATOM_E1, AT2 = bwd::NATOM - AT1;
            static_for<bwd::NWIN>([&](auto g) {
                e_wmma(std::integral_constant<int, bwd::NE1 + g>{}, std::integral_constant<int, bwd::NE1 + g + 1>{});
                constexpr int r0 = RB1 + (NR * g) / bwd::NWIN, r1 = RB1 + (NR * (g + 1)) / bwd::NWIN;
                readback(std::integral_constant<int, r0>{}, std::integral_constant<int, r1>{}, SLN{}, X);
                constexpr int a0 = AT1 + (AT2 * g) / bwd::NWIN, a1 = AT1 + (AT2 * (g + 1)) / bwd::NWIN;
                atomics(std::integral_constant<int, a0>{}, std::integral_constant<int, a1>{}, dq, avo);
            });
            static_for<bwd::NWIN>([&](auto g) {
                sched_group_barrier<sg::WMMA, 1, 0>();
                constexpr int nr = bwd::rb_loads(RB1 + (NR * (g + 1)) / bwd::NWIN) - bwd::rb_loads(RB1 + (NR * g) / bwd::NWIN);
                if constexpr (nr > 0) sched_group_barrier<sg::DS_READ, nr, 0>();
                constexpr int na = (AT2 * (g + 1)) / bwd::NWIN - (AT2 * g) / bwd::NWIN;
                if constexpr (na > 0) sched_group_barrier<sg::VMEM, na, 0>();
            });
        }
#if BWD_KERNEL_SCOPE
        // the E WMMA operands stay allocated to the end of E2: the read-back targets cannot be the registers
        // the last E WMMAs read (expert-mode WAR drains in E2)
        sched_barrier<0>();
        if constexpr (live) {
            HOLD_8(QT);
            HOLD_8(dOT);
            HOLD_2X2(pu);
            HOLD_2X2(du);
        }
#endif
        // WAIT
        split_wait_lds();
