###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""fwd1gen -- the generated regions of k_attn_fwd_v1 (HK gfx1250 d128 GQA attention forward v1, Llama).

The marker shell (hip/src/k_attn_fwd_v1.hip, -DHKGEN=1) runs the persistent item loop, the schedule decode, the TDM
descriptors and the LDS burst bases; it marks per item

  FRONT       s_wait_tensorcnt 1 (own Q landed), the Q^T B operands, K(0), K(1), V(0); O^T = 0, row sums = 0; QK(0);
              SM1(0) (masked when n = 1): m2 = max(own half, other half) c2, m2th = (m2 + TH2) / c2
  STAGE_MAIN  one ring stage t < n - 2 (chunks c = 2t, 2t + 1), STAGE_PEN stage n - 2 (SM1 of its odd chunk masked),
  STAGE_LAST  stage n - 1 (Q(j+1) before its op, SM1 of its even chunk masked, no QK / SM1 in its odd chunk)
  EPI         l = sum, O^T / l to bf16 into the wave's O image, whole-row read-back and O stores; LSE = (m2 + log2 l) ln 2

A stage is two blocks, one per chunk, each ending in the lazy-rescale branch (out-of-line block R: the rescale of
chunk c + 1, taken when some q-tile's own-half max of chunk c + 1 exceeds its threshold). Chunk c (par = c & 1):
  QK(c+1) -> S[1 - par] (K operands kab[1 - par]) || exp / P^T pack / row sums of S[par] (the scores of chunk c)
  even: s_barrier_wait (WAIT(P_{T+1})), [LAST: Q(j+1)], the TDM op of stage T + 3; V(c+1) -> V[1] (stage T),
        K(c+2) -> kab[0], K(c+3) -> kab[1] (stage T + 1)
  odd:  s_wait_tensorcnt 1 (LAST: 2) + s_wait_dscnt N (V(c+1) read) + s_barrier_signal (SIGNAL(P_{T+2}));
        V(c+2) -> V[0] (stage T + 1, after PV(c - 1) released V[0])
  PV(c): O^T[h] += V[par] P^T[h] (8 d tiles x 4 q-tiles) || SM1(c+1): [mask], own-half max, the rescale decision
Every operation is the HIP kernel's (FWD1_LSUM 0, SM1L 1, OPT_RESC 1): o / lse are bitwise those of the production
build. The node lists are in program order; gencore.schedule_slot reorders them along the dependence graph."""

from gencore import (  # noqa: F401
    add_order,
    build_dag,
    fix_fence,
    mk,
    offset,
    schedule_slot,
    sreg,
    spair,
    vr,
    wmma,
)
import gencore as GC
import re

KP, VP = 272, 288  # K / V ring pitches (bytes)
NH, NDK, NDT = 4, 4, 8
NEG_INF = "0xff800000"
TH2 = "0x4138aa3b"  # 8 log2(e): the lazy-rescale threshold (log2 domain)
LN2 = "0x3f317218"
SEL_LO, SEL_HI = "0x76543210", "0xfedcba98"  # v_permlanex16_b32: lane l reads lane l of the other half
S_SEL, S_NEED, S_B, S_ANY, S_HI = "s96", "s97", "s98", "s99", "s100"
CLOBBERED = {f"s{i}" for i in range(96, 104)} | {"vcc", "vcc_lo", "vcc_hi", "scc"}
# the stage blocks keep their scalars in vcc (s96..s103 hold STAGE_LOOP's descriptor group 1): the rescale decision
# (vcc_lo, the branch tests it with s_cmp), the rescale's ballot / select (vcc_lo) and permlane select (vcc_hi)
V_NEED, V_B, V_SEL = "vcc_lo", "vcc_lo", "vcc_hi"
G1_LOOP = "s[96:103]"
K_SLOT_B, V_SLOT_B = 0x4400, 0x4800
K_W0, V_W0 = 0x07510000, 0x0F510000  # tdm_w0(2 D, K_PITCH / V_PITCH) (fwd1/idx.h)
INOUT = {"STAGE_LOOP": ("g0", "lp")}  # marker operands a region rewrites (in / out asm operands)


def k_imm(hc, kt, k, half):
    return (32 * hc + 16 * kt) * KP + 64 * k + 32 * half


def v_imm(hc, half, dt):
    return (32 * hc + 16 * half) * VP + 32 * dt


def q_imm(h, k, half):
    return 256 * h + 64 * k + 32 * half


class Layout:
    """VGPR map of the generated state (true register numbers >= base; no 8-tuple crosses a 256 block).
      qb[h][k]          Q^T B operands of the item
      kab[par][kt][k]   K A operands of chunk parity par (QK(c) reads kab[c & 1])
      V[par][dt]        V^T A operands (PV of an even chunk reads V[0], of an odd one V[1])
      S[s][h][kt]       S^T of chunk parity s (exp, row sums in place)
      P[h]              bf16 P^T B operands
      O[h][dt]          O^T accumulators
      lsum[h], m2[h]    pairs (lsum: the two lane partial sums; m2: element 0, the running max in the log2 domain)
      m2th[h], mxh[h]   the rescale threshold and chunk c + 1's own-half max
      cc                (c2, c2); dv the rescale decision terms; thr the mask thresholds; t scratch"""

    def __init__(self, base):
        used = [[] for _ in range(4)]

        def take(n, align):
            for blk in range(4):
                r = max([max(base, blk * 256)] + [a + m for a, m in used[blk]])
                r += (-r) % align
                if r + n <= (blk + 1) * 256:
                    used[blk].append((r, n))
                    return r
            raise ValueError(f"layout does not fit above base v{base}")

        t8 = lambda: take(8, 8)  # noqa: E731
        self.O = [[t8() for _ in range(NDT)] for _ in range(NH)]
        self.qb = [[t8() for _ in range(NDK)] for _ in range(NH)]
        self.kab = [[[t8() for _ in range(NDK)] for _ in range(2)] for _ in range(2)]
        self.V = [[t8() for _ in range(NDT)] for _ in range(2)]
        self.S = [[[t8() for _ in range(2)] for _ in range(NH)] for _ in range(2)]
        self.P = [t8() for _ in range(NH)]
        self.lsum = [take(2, 2) for _ in range(NH)]
        self.m2 = [take(2, 2) for _ in range(NH)]
        self.m2th = [take(1, 1) for _ in range(NH)]
        self.mxh = [take(1, 1) for _ in range(NH)]
        self.cc = take(2, 2)
        self.dv = [take(1, 1) for _ in range(NH)]
        self.thr = [take(1, 1) for _ in range(2)]
        self.t = [take(2, 2) for _ in range(8)]  # scratch pairs (SM1 FIRST, rescale)
        self.lb = [take(1, 1) for _ in range(4)]  # STAGE_LOOP's burst bases kb1, vb0, vb1 and a temporary
        self.ssave = take(1, 1)  # STAGE_LOOP: s96..s103 of the shell in lanes 0..7
        self.top = max(a + m for blk in used for a, m in blk)


# --------------------------------------------------------------------------------------------------------------
# instruction texts
# --------------------------------------------------------------------------------------------------------------
def ld_k(L, par, base, hc):
    return [
        f"ds_load_b128 {vr(L.kab[par][kt][k] + 4 * u, 4)}, {base}{offset(k_imm(hc, kt, k, u))}"
        for k in range(NDK)
        for kt in range(2)
        for u in range(2)
    ]


def ld_v(L, par, base, hc):
    return [
        f"ds_load_tr16_b128 {vr(L.V[par][dt] + 4 * u, 4)}, {base}{offset(v_imm(hc, u, dt))}"
        for dt in range(NDT)
        for u in range(2)
    ]


REUSE = {"a": False, "b": False}  # WMMA operand-reuse hints (build.py --wmma-reuse)


def qk_wmmas(L, s, kpar):
    """S^T(s) = K Q^T: k outer, kt, h inner, chains from +0 (the HIP qk). Reuse a: the 4 WMMAs of a (k, kt) group read
    one K tile (the scheduler keeps the WMMA order: the next WMMA of a hinted one is the group's next)."""
    return [
        wmma(L.S[s][h][kt], L.kab[kpar][kt][k], L.qb[h][k], None if k == 0 else L.S[s][h][kt])
        + (" matrix_a_reuse" if REUSE["a"] and h < NH - 1 else "")
        for k in range(NDK)
        for kt in range(2)
        for h in range(NH)
    ]


def pv_wmmas(L, vpar, h):
    """O^T(h) += V^T P^T(h), dt inner. Reuse b: the 8 WMMAs read one P tile."""
    return [wmma(L.O[h][dt], L.V[vpar][dt], L.P[h], L.O[h][dt]) + (" matrix_b_reuse" if REUSE["b"] and dt < NDT - 1 else "")
            for dt in range(NDT)]


def tail_texts(L, s, h):
    """p = exp2(s c2 - m2) in place, the P^T pack of q-tile h, then its row sums in place (lsum += (a + b))."""
    out = []
    for kt in range(2):
        a = L.S[s][h][kt]
        for j in range(4):
            out.append(f"v_pk_fma_f32 {vr(a + 2 * j, 2)}, {vr(a + 2 * j, 2)}, {vr(L.cc, 2)}, {vr(L.m2[h], 2)} "
                       "op_sel_hi:[1,1,0] neg_lo:[0,0,1] neg_hi:[0,0,1]")  # fmt: skip
            out.append(f"v_exp_f32 {vr(a + 2 * j)}, {vr(a + 2 * j)}")
            out.append(f"v_exp_f32 {vr(a + 2 * j + 1)}, {vr(a + 2 * j + 1)}")
    for kt in range(2):
        a = L.S[s][h][kt]
        for q in range(4):
            out.append(f"v_cvt_pk_bf16_f32 {vr(L.P[h] + 4 * kt + q)}, {vr(a + 2 * q)}, {vr(a + 2 * q + 1)}")
    a0, a1 = L.S[s][h][0], L.S[s][h][1]
    for a in (a0, a1):  # pk_add(pk_add(p0, p1), pk_add(p2, p3)) in place into pair 0
        out.append(f"v_pk_add_f32 {vr(a, 2)}, {vr(a, 2)}, {vr(a + 2, 2)}")
        out.append(f"v_pk_add_f32 {vr(a + 4, 2)}, {vr(a + 4, 2)}, {vr(a + 6, 2)}")
        out.append(f"v_pk_add_f32 {vr(a, 2)}, {vr(a, 2)}, {vr(a + 4, 2)}")
    out.append(f"v_pk_add_f32 {vr(a0, 2)}, {vr(a0, 2)}, {vr(a1, 2)}")
    out.append(f"v_pk_add_f32 {vr(L.lsum[h], 2)}, {vr(L.lsum[h], 2)}, {vr(a0, 2)}")
    return out


def mask_texts(L, s, h):
    """masked iff element i > dl + off (S^T: row kv > column q + off): keep where i <= thr."""
    out = []
    for kt in range(2):
        a = L.S[s][h][kt]
        for i in range(8):
            out.append(f"v_cmp_le_i32 vcc_lo, {i}, {vr(L.thr[kt])}")
            out.append(f"v_cndmask_b32 {vr(a + i)}, {NEG_INF}, {vr(a + i)}, vcc_lo")
    return out


def max16_texts(L, s, h, dst):
    """dst = max over the lane's 16 values of q-tile h (7 v_max3 + 1 v_max; exact in any order)."""
    a, b = L.S[s][h][0], L.S[s][h][1]
    t = L.t[h]
    x = [t, t + 1]
    return [
        f"v_max3_num_f32 {vr(x[0])}, {vr(a)}, {vr(a + 1)}, {vr(a + 2)}",
        f"v_max3_num_f32 {vr(x[1])}, {vr(a + 3)}, {vr(a + 4)}, {vr(a + 5)}",
        f"v_max3_num_f32 {vr(x[0])}, {vr(x[0])}, {vr(x[1])}, {vr(a + 6)}",
        f"v_max3_num_f32 {vr(x[1])}, {vr(a + 7)}, {vr(b)}, {vr(b + 1)}",
        f"v_max3_num_f32 {vr(x[0])}, {vr(x[0])}, {vr(x[1])}, {vr(b + 2)}",
        f"v_max3_num_f32 {vr(x[1])}, {vr(b + 3)}, {vr(b + 4)}, {vr(b + 5)}",
        f"v_max3_num_f32 {vr(x[1])}, {vr(x[1])}, {vr(b + 6)}, {vr(b + 7)}",
        f"v_max_num_f32 {vr(dst)}, {vr(x[0])}, {vr(x[1])}",
    ]


def thr_texts(L, ops):
    sc = ops["sc"]
    return [f"v_add_nc_u32 {vr(L.thr[kt])}, {sreg(sc, kt)}, {ops['dl']}" for kt in range(2)]


def sm1_first_texts(L, ops, h):
    """FRONT: m2 = max(mx, xhalf(mx)) c2, m2th = (m2 + TH2) c2inv (no fma: fp contract off)."""
    sc = ops["sc"]
    t = L.t[4 + h]
    mx = L.mxh[h]
    return max16_texts(L, 0, h, mx) + [
        f"v_permlanex16_b32 {vr(t)}, {vr(mx)}, {S_SEL}, {SEL_HI}",
        f"v_max_num_f32 {vr(t)}, {vr(mx)}, {vr(t)}",
        f"v_mul_f32 {vr(L.m2[h])}, {sreg(sc, 2)}, {vr(t)}",
        f"v_add_f32 {vr(t + 1)}, {TH2}, {vr(L.m2[h])}",
        f"v_mul_f32 {vr(L.m2th[h])}, {sreg(sc, 3)}, {vr(t + 1)}",
    ]


def sm1_texts(L, s, h):
    """SM1 (not first): mxh = own-half max; dv = mxh - m2th (the rescale decision term)."""
    return max16_texts(L, s, h, L.mxh[h]) + [f"v_sub_f32 {vr(L.dv[h])}, {vr(L.mxh[h])}, {vr(L.m2th[h])}"]


def need_texts(L):
    """need = ballot(max_h (mxh_h - m2th_h) > 0) (exact: denormals on)."""
    d = L.dv
    return [
        f"v_max3_num_f32 {vr(d[0])}, {vr(d[0])}, {vr(d[1])}, {vr(d[2])}",
        f"v_max_num_f32 {vr(d[0])}, {vr(d[0])}, {vr(d[3])}",
        f"v_cmp_gt_f32_e64 {V_NEED}, {vr(d[0])}, 0",
    ]


def rescale_texts(L, ops):
    """The out-of-line block: per q-tile b = ballot(mxh > m2th); mn = b ? max(m2, max(mxh, xhalf(mxh)) c2) : m2;
    corr = exp2(m2 - mn); O^T, lsum *= corr; m2 = mn; m2th = (mn + TH2) c2inv. Serial per q-tile (rare)."""
    sc = ops.get("sc")
    c2, ci = ops.get("c2") or sreg(sc, 2), ops.get("ci") or sreg(sc, 3)
    out = [f"s_mov_b32 {V_SEL}, {SEL_LO}"]
    for h in range(NH):
        t0, t1 = L.t[0], L.t[1]
        out += [
            f"v_cmp_gt_f32_e64 {V_B}, {vr(L.mxh[h])}, {vr(L.m2th[h])}",
            f"s_cmp_lg_u32 {V_B}, 0",
            f"s_cselect_b32 {V_B}, -1, 0",
            f"v_permlanex16_b32 {vr(t0)}, {vr(L.mxh[h])}, {V_SEL}, {SEL_HI}",
            f"v_max_num_f32 {vr(t0)}, {vr(L.mxh[h])}, {vr(t0)}",
            f"v_mul_f32 {vr(t0)}, {c2}, {vr(t0)}",
            f"v_max_num_f32 {vr(t0)}, {vr(L.m2[h])}, {vr(t0)}",
            f"v_cndmask_b32_e64 {vr(t0 + 1)}, {vr(L.m2[h])}, {vr(t0)}, {V_B}",
            f"v_sub_f32 {vr(t1)}, {vr(L.m2[h])}, {vr(t0 + 1)}",
            f"v_exp_f32 {vr(t1)}, {vr(t1)}",
        ]
        for dt in range(NDT):
            for r in range(0, 8, 2):
                o = vr(L.O[h][dt] + r, 2)
                out.append(f"v_pk_mul_f32 {o}, {o}, {vr(t1, 2)} op_sel_hi:[1,0]")
        out.append(f"v_pk_mul_f32 {vr(L.lsum[h], 2)}, {vr(L.lsum[h], 2)}, {vr(t1, 2)} op_sel_hi:[1,0]")
        out += [
            f"v_mov_b32 {vr(L.m2[h])}, {vr(t0 + 1)}",
            f"v_add_f32 {vr(t1 + 1)}, {TH2}, {vr(t0 + 1)}",
            f"v_mul_f32 {vr(L.m2th[h])}, {ci}, {vr(t1 + 1)}",
        ]
    return out


# --------------------------------------------------------------------------------------------------------------
# regions
# --------------------------------------------------------------------------------------------------------------
def schedule_texts(texts, edges_fn=None, wmma_pred=None):
    """Schedule a straight-line list (program order = semantics) with gencore's slot scheduler."""
    nodes = [mk(t) for t in texts]
    build_dag(nodes)
    if edges_fn:
        edges_fn(nodes)
    W = [i for i, n in enumerate(nodes) if n.kind == "WMMA"]
    order, cycles = schedule_slot(nodes, W)
    return [n.text for n in order], cycles


def region_front(L, ops):
    sc = ops["sc"]
    texts = [f"s_mov_b32 {S_SEL}, {SEL_LO}", "s_wait_tensorcnt 0x1"]
    qloads = [
        f"ds_load_b128 {vr(L.qb[h][k] + 4 * u, 4)}, {ops['qp']}{offset(q_imm(h, k, u))}"
        for h in range(NH)
        for k in range(NDK)
        for u in range(2)
    ]
    texts += qloads + ld_k(L, 0, ops["kb"], 0) + ld_k(L, 1, ops["kb"], 1) + ld_v(L, 0, ops["vb"], 0)
    for h in range(NH):
        for dt in range(NDT):
            for q in range(0, 8, 2):
                texts.append(f"v_mov_b64 {vr(L.O[h][dt] + q, 2)}, 0")
        texts.append(f"v_mov_b64 {vr(L.lsum[h], 2)}, 0")
    texts += [f"v_mov_b32 {vr(L.cc)}, {sreg(sc, 2)}", f"v_mov_b32 {vr(L.cc + 1)}, {sreg(sc, 2)}"]
    texts += qk_wmmas(L, 0, 0)
    texts += thr_texts(L, ops)
    for h in range(NH):
        texts += mask_texts(L, 0, h)
    for h in range(NH):
        texts += sm1_first_texts(L, ops, h)

    def edges(nodes):  # the Q loads behind the tensorcnt wait (own Q(j) landed)
        for i, n in enumerate(nodes):
            if n.text.startswith("ds_load") and i > 1:
                add_order(nodes, 1, i)

    t, cyc = schedule_texts(texts, edges)
    return [("F", t, ["X"])], {"FRONT": cyc}


# chunk-block parameters
SIG_AT, WAIT_AT = 16, 24  # the odd chunk's signal by its WMMA SIG_AT; the even chunk's wait by WMMA WAIT_AT
V2_AFTER = 13  # V(c+2) -> V[0] after this WMMA of the odd chunk (PV(c-1) released V[0])


def chunk_nodes(L, ops, par, kind, extra=None):
    """(nodes, WMMA order, protocol DS loads, fence) of one chunk block. kind: MAIN / PEN / LAST. The barrier point
    and the loads stand at fixed WMMA positions of the list (hooks), so every order edge points forward. extra:
    {"after_td": texts behind the TDM op (program order), "end": texts at the end of the list} (STAGE_LOOP)."""
    extra = extra or {}
    last = kind == "LAST"
    qkn = not (last and par == 1)
    sm = 0 if (last and par == 1) else (2 if (kind == "PEN" and par == 1) or (last and par == 0) else 1)
    nodes = []

    def add(text):
        nodes.append(mk(text))
        return len(nodes) - 1

    W, proto, marks = [], [], {}
    fence = None
    s, ns = par, 1 - par
    if sm == 2:
        for t in thr_texts(L, ops):
            add(t)
    wtexts = (qk_wmmas(L, ns, ns) if qkn else []) + [t for h in range(NH) for t in pv_wmmas(L, s, h)]
    nq = 32 if qkn else 0

    def tails():
        for h in range(NH):
            for t in tail_texts(L, s, h):
                add(t)

    def even_point():
        marks["bw"] = add("s_barrier_wait -1")
        if last:
            marks["tq"] = add(f"tensor_load_to_lds {ops['q0']}, {ops['q1']}")
        marks["td"] = add(f"tensor_load_to_lds {ops['g0']}, {ops['g1']}")
        for t in extra.get("after_td", []):
            add(t)
        proto.extend(add(t) for t in ld_v(L, 1, ops["vb0"], 1))  # V(c+1): stage T, half 1
        marks["kn"] = []
        if not last:  # K(c+2) -> kab[0]: stage T + 1 (published by the wait); kab[0] is not read in this chunk
            marks["kn"] = [add(t) for t in ld_k(L, 0, ops["kb1"], 0)]

    def k_next():  # K(c+3) -> kab[1]: behind every QK WMMA that reads kab[1] (program order = semantics)
        if not last:
            marks["kn"] += [add(t) for t in ld_k(L, 1, ops["kb1"], 1)]

    def odd_point():
        marks["tw"] = add(f"s_wait_tensorcnt {'0x2' if last else '0x1'}")
        marks["fence"] = add("s_wait_dscnt 0x0")
        marks["sig"] = add("s_barrier_signal -1")

    def v_next():
        marks["v2"] = [add(t) for t in ld_v(L, 0, ops["vb1"], 0)] if not last else []  # V(c+2) -> V[0]

    hooks = {nq: [tails]}
    if par == 0:
        hooks.setdefault(WAIT_AT, []).append(even_point)
        hooks.setdefault(nq, []).append(k_next)
    else:
        hooks.setdefault(SIG_AT, []).append(odd_point)
        hooks.setdefault(V2_AFTER + 1, []).append(v_next)
    for k, t in enumerate(wtexts):
        for f in hooks.get(k, []):
            f()
        W.append(add(t))
    for f in hooks.get(len(wtexts), []):
        f()
    if sm:
        for h in range(NH):
            if sm == 2:
                for t in mask_texts(L, ns, h):
                    add(t)
            for t in sm1_texts(L, ns, h):
                add(t)
        for t in need_texts(L):
            add(t)
    for t in extra.get("end", []):
        add(t)
    build_dag(nodes)
    if par == 0:
        add_order(nodes, marks["bw"], W[WAIT_AT])
        add_order(nodes, marks["bw"], marks["td"])
        for i in marks["kn"]:
            add_order(nodes, marks["bw"], i)
    else:
        fence = marks["fence"]
        add_order(nodes, marks["tw"], fence)
        add_order(nodes, fence, marks["sig"])
        add_order(nodes, marks["sig"], W[SIG_AT])
        for i in marks["v2"]:
            add_order(nodes, W[V2_AFTER], i)
    return nodes, W, proto, fence


def stage_blocks(L, ops, kind, extra=(None, None), latch=None):
    """Blocks C0, C1 and the rescale blocks R0, R1 (names R*: placed out of line by build.py, after the kernel's
    s_endpgm). C1 falls through to the region exit, or to the latch block (STAGE_LOOP)."""
    blocks, cyc = [], {}
    orders = []
    for par in range(2):
        nodes, W, proto, fence = chunk_nodes(L, ops, par, kind, extra[par])
        order, cycles = schedule_slot(nodes, W)
        orders.append((nodes, order, proto, fence))
        cyc[f"{kind}{par}"] = cycles
    # the odd chunk's fence: DS ops issued after the last V(c+1) load (even block) and before the fence (odd block)
    n0, o0, p0, _ = orders[0]
    n1, o1, _, f1 = orders[1]
    last_p = max(o0.index(n0[i]) for i in p0)
    after = sum(1 for x in o0[last_p + 1 :] if x.kind in ("DS_LD", "DS_ST"))
    pos = o1.index(n1[f1])
    before = sum(1 for x in o1[:pos] if x.kind in ("DS_LD", "DS_ST"))
    n1[f1].text = f"s_wait_dscnt 0x{min(after + before, 63):x}"
    has_r = [not (kind == "LAST" and par == 1) for par in range(2)]
    after_c1 = latch[0] if latch else "X"
    for par in range(2):
        t = [n.text for n in orders[par][1]]
        nxt = "C1" if par == 0 else after_c1
        if has_r[par]:
            t += [f"s_cmp_lg_u32 {V_NEED}, 0", f"s_cbranch_scc1 @R{par}"]
            blocks.append((f"C{par}", t, [nxt, f"R{par}"]))
        else:
            blocks.append((f"C{par}", t, [nxt]))
    if latch:
        blocks.append(latch)
    for par in range(2):
        if has_r[par]:
            back = "C1" if par == 0 else after_c1
            blocks.append((f"R{par}", rescale_texts(L, ops) + [f"s_branch @{back}"], [back]))
    return blocks, cyc


def region_stage(L, ops, kind):
    return stage_blocks(L, ops, kind)


SAVED = {}  # region kind -> the shell SGPRs it saves in VGPR lanes, uses, and restores (build.py allows them)


def sgprs_of(op):
    m = re.fullmatch(r"s\[(\d+):(\d+)\]|s(\d+)", op)
    if not m:
        return set()
    a = int(m.group(1) or m.group(3))
    return set(range(a, (int(m.group(2)) if m.group(1) else a) + 1))


def region_loop(L, ops):
    """STAGE_LOOP: stages T = T0 .. T_end - 1 of an item (MAIN stages whose TDM op is the item's own whole op of local
    stage T - T0 + 3). The marker clobbers no SGPR: the loop takes an 8-aligned block G of SGPRs no operand uses, saves
    it in VGPR lanes (entry block E) and restores it (exit block Z). E: descriptor group 1 of a whole K / V op into G
    (fwd1 g1_2d(W0, D, 32, D, 32, ss)), the burst bases into the layout's lb; the stage blocks; the latch L (T <
    T_end: back to C0). In / out operands: g0 (the op's group 0, advanced behind each TDM op: address + 128 ss bytes
    (G[6:7] are its temporaries, zero again before the next op), LDS address = b0 + slot(T + 4) sb) and lp = {T_end,
    T} (T + 1 at the end of the odd chunk). At the end of the odd chunk: kb1 = kbase(T + 2), vb0 = vb1, vb1 =
    vbase(T + 2). ss: the op's row stride (elements); b0: the LDS address of its slot 0; sb: the role's slot bytes;
    c2, ci: the softmax scale and its inverse (log2 domain)."""
    lp, g0 = ops["lp"], ops["g0"]
    used = set().union(*(sgprs_of(v) for v in ops.values()))
    gb = next(b for b in range(96, -1, -8) if not used & set(range(b, b + 8)))
    SAVED["STAGE_LOOP"] = {f"s{gb + i}" for i in range(8)}
    KB1, VB0, VB1, VT = (vr(r) for r in L.lb)
    T = sreg(lp, 1)
    lops = dict(ops, kb1=KB1, vb0=VB0, vb1=VB1, g1=f"s[{gb}:{gb + 7}]")
    g = lambda i: f"s{gb + i}"  # noqa: E731
    entry = [f"v_writelane_b32 {vr(L.ssave)}, {g(i)}, {i}" for i in range(8)] + [
        f"s_mov_b32 {g(0)}, {hex(K_W0)}",
        f"s_cmp_eq_u32 {ops['sb']}, {hex(V_SLOT_B)}",
        f"s_cselect_b32 {g(0)}, {hex(V_W0)}, {g(0)}",
        f"s_mov_b32 {g(1)}, 0x800000",
        f"s_mov_b32 {g(2)}, 0x200000",
        f"s_mov_b32 {g(3)}, 0x800000",
        f"s_mov_b32 {g(4)}, 32",
        f"s_mov_b32 {g(5)}, {ops['ss']}",
        f"s_mov_b64 s[{gb + 6}:{gb + 7}], 0",
        f"v_mov_b32 {KB1}, {ops['kb1']}",
        f"v_mov_b32 {VB0}, {ops['vb0']}",
        f"v_mov_b32 {VB1}, {ops['vb1']}",
    ]
    after_td = [
        f"s_mov_b32 {g(6)}, {ops['ss']}",
        f"s_lshl_b64 s[{gb + 6}:{gb + 7}], s[{gb + 6}:{gb + 7}], 7",
        f"s_add_nc_u64 {spair(g0, 2)}, {spair(g0, 2)}, s[{gb + 6}:{gb + 7}]",
        f"s_mov_b64 s[{gb + 6}:{gb + 7}], 0",
        f"s_and_b32 {sreg(g0, 1)}, {T}, 3",
        f"s_mul_i32 {sreg(g0, 1)}, {sreg(g0, 1)}, {ops['sb']}",
        f"s_add_co_i32 {sreg(g0, 1)}, {sreg(g0, 1)}, {ops['b0']}",
    ]
    end = [
        f"v_add_nc_u32 {VT}, {T}, 2",
        f"v_and_b32 {VT}, 3, {VT}",
        f"v_mad_u32 {KB1}, {hex(K_SLOT_B)}, {VT}, {ops['kl']}",
        f"v_mov_b32 {VB0}, {VB1}",
        f"v_mad_u32 {VB1}, {hex(V_SLOT_B)}, {VT}, {ops['vl']}",
        f"s_add_co_i32 {T}, {T}, 1",
    ]
    latch = ("L", [f"s_cmp_lt_i32 {T}, {sreg(lp, 0)}", "s_cbranch_scc1 @C0"], ["C0", "Z"])
    restore = ("Z", [f"v_readlane_b32 {g(i)}, {vr(L.ssave)}, {i}" for i in range(8)], ["X"])
    blocks, cyc = stage_blocks(L, lops, "MAIN", ({"after_td": after_td}, {"end": end}), latch)
    cyc = {k.replace("MAIN", "LOOP"): v for k, v in cyc.items()}
    k = next(i for i, b in enumerate(blocks) if b[0] == "L")
    return [("E", entry, ["C0"])] + blocks[: k + 1] + [restore] + blocks[k + 1 :], cyc


def region_epi(L, ops):
    """l = lsum.x + lsum.y + xhalf; inv = l > 0 ? 1 / l : 0 (LLVM's fp32 division); O^T inv -> bf16 -> O image;
    read-back + whole-row O stores; lse = (m2 + log2 l) ln 2 -> LSE (lanes 0..15 q-tiles 0 / 1, 16..31 2 / 3)."""
    texts = [f"s_mov_b32 {S_SEL}, {SEL_LO}", f"s_mov_b32 {S_HI}, 0xffff0000"]
    sc = ops["ev"]
    scr = [L.S[0][h][kt] for h in range(NH) for kt in range(2)] + [L.S[1][h][kt] for h in range(NH) for kt in range(2)]
    lse = []
    div_chains = []
    for h in range(NH):
        x = scr[h]  # 8 scratch VGPRs per q-tile
        ll, tt, lt, a, b, rc, e1, q = (x + k for k in range(8))
        y = scr[8 + h]
        e2, inv, lg, lv = y, y + 2, y + 3, y + 4  # inv at an even register: a pair operand
        lse.append(lv)
        texts += [
            f"v_add_f32 {vr(ll)}, {vr(L.lsum[h])}, {vr(L.lsum[h] + 1)}",
            f"v_permlanex16_b32 {vr(tt)}, {vr(ll)}, {S_SEL}, {SEL_HI}",
            f"v_add_f32 {vr(lt)}, {vr(ll)}, {vr(tt)}",
        ]
        chain = [
            f"v_div_scale_f32 {vr(a)}, null, {vr(lt)}, {vr(lt)}, 1.0",
            f"v_div_scale_f32 {vr(b)}, vcc_lo, 1.0, {vr(lt)}, 1.0",
            f"v_rcp_f32 {vr(rc)}, {vr(a)}",
            f"v_fma_f32 {vr(e1)}, -{vr(a)}, {vr(rc)}, 1.0",
            f"v_fmac_f32 {vr(rc)}, {vr(e1)}, {vr(rc)}",
            f"v_mul_f32 {vr(q)}, {vr(b)}, {vr(rc)}",
            f"v_fma_f32 {vr(e2)}, -{vr(a)}, {vr(q)}, {vr(b)}",
            f"v_fmac_f32 {vr(q)}, {vr(e2)}, {vr(rc)}",
            f"v_fma_f32 {vr(a)}, -{vr(a)}, {vr(q)}, {vr(b)}",
            f"v_div_fmas_f32 {vr(a)}, {vr(a)}, {vr(rc)}, {vr(q)}",
            f"v_div_fixup_f32 {vr(a)}, {vr(a)}, {vr(lt)}, 1.0",
            f"v_cmp_lt_f32_e64 {S_B}, 0, {vr(lt)}",
            f"v_cndmask_b32_e64 {vr(inv)}, 0, {vr(a)}, {S_B}",
        ]
        div_chains.append((len(texts), len(texts) + len(chain)))
        texts += chain
        for dt in range(NDT):
            o = L.O[h][dt]
            for r in range(0, 8, 2):
                texts.append(f"v_pk_mul_f32 {vr(o + r, 2)}, {vr(o + r, 2)}, {vr(inv, 2)} op_sel_hi:[1,0]")
            for qd in range(4):
                texts.append(f"v_cvt_pk_bf16_f32 {vr(o + qd)}, {vr(o + 2 * qd)}, {vr(o + 2 * qd + 1)}")
            texts.append(f"ds_store_b128 {ops['ow']}, {vr(o, 4)}{offset(256 * h + 32 * dt)}")
        texts += [
            f"v_log_f32 {vr(lg)}, {vr(lt)}",
            f"v_add_f32 {vr(lg)}, {vr(L.m2[h])}, {vr(lg)}",
            f"v_mul_f32 {vr(lv)}, {LN2}, {vr(lg)}",
        ]
    n_img = len(texts)
    # read-back of store s: position s / 2, bytes 512 (s % 2); global row (s / 2): ob + (s / 2) rs
    vo = [L.kab[0][0][0] + k for k in range(16)]
    # read-back targets: kab[1] and V[0] (128 VGPRs, dead after the last stage)
    rb = [L.kab[1][kt][k] + 4 * u for kt in range(2) for k in range(NDK) for u in range(2)]
    rb += [L.V[0][dt] + 4 * u for dt in range(NDT) for u in range(2)]
    texts.append(f"v_mov_b32 {vr(vo[0])}, {ops['ob']}")
    for k in range(1, 16):
        texts.append(f"v_add_nc_u32 {vr(vo[k])}, {sreg(sc, 0)}, {vr(vo[k - 1])}")
    loads = []
    for s_ in range(32):
        loads.append(len(texts))
        texts.append(f"ds_load_b128 {vr(rb[s_], 4)}, {ops['or']}{offset((s_ // 2) * 1040 + 512 * (s_ % 2))}")
        texts.append(f"buffer_store_b128 {vr(rb[s_], 4)}, {vr(vo[s_ // 2])}, {ops['ro']}, null offen{offset(512 * (s_ % 2))}")
    sel = [L.S[1][0][0] + k for k in range(2)]
    for k in range(2):
        texts += [
            f"v_cndmask_b32_e64 {vr(sel[k])}, {vr(lse[k])}, {vr(lse[2 + k])}, {S_HI}",
            f"buffer_store_b32 {vr(sel[k])}, {ops['l' + str(k)]}, {ops['rl']}, null offen",
        ]

    def edges(nodes):
        stores = [i for i in range(n_img) if nodes[i].text.startswith("ds_store")]
        for i in loads:  # the read-back after every image store (LDS: same wave, but no register dependence)
            for j in stores:
                add_order(nodes, j, i)
        for (a0, a1), (b0, _) in zip(div_chains, div_chains[1:]):  # one division at a time owns vcc
            add_order(nodes, a1 - 1, b0)

    t, cyc = schedule_texts(texts, edges)
    return [("E", t, ["X"])], {"EPI": cyc}
