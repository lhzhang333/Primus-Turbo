	.amdgcn_target "amdgpu12.50-amd-amdhsa-unknown-gfx1250"
	.amdhsa_code_object_version 6
	.text
	.protected	k_mla_fwd_mx            ; -- Begin function k_mla_fwd_mx
	.globl	k_mla_fwd_mx
	.p2align	8
	.type	k_mla_fwd_mx,@function
k_mla_fwd_mx:                           ; @k_mla_fwd_mx
	.cfi_startproc
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[64:65], 0
	v_nop
	global_prefetch_b8 v0, s[64:65] scope:SCOPE_SE
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b512 s[4:19], s[0:1], 0x0 nv
	s_load_b96 s[24:26], s[0:1], 0x40 nv
	s_bfe_u32 s2, ttmp6, 0x4000c
	s_bfe_u32 s21, ttmp6, 0x40010
	s_add_co_i32 s2, s2, 1
	s_and_b32 s3, ttmp6, 15
	s_mul_i32 s2, ttmp9, s2
	s_and_b32 s20, ttmp7, 0xffff
	s_add_co_i32 s21, s21, 1
	s_add_co_i32 s3, s3, s2
	s_mul_i32 s2, s20, s21
	s_bfe_u32 s21, ttmp6, 0x40004
	s_getreg_b32 s22, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s21, s21, s2
	s_cmp_eq_u32 s22, 0
	s_mov_b32 s71, 0
	s_cselect_b32 s68, s20, s21
	s_cselect_b32 s20, ttmp9, s3
	s_bfe_u32 s2, ttmp6, 0x40014
	s_lshr_b32 s3, ttmp7, 16
	s_add_co_i32 s2, s2, 1
	s_bfe_u32 s21, ttmp6, 0x40008
	s_mul_i32 s2, s3, s2
	s_add_co_i32 s21, s21, s2
	s_cmp_eq_u32 s22, 0
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
	v_readfirstlane_b32 s2, v0
	s_cselect_b32 s70, s3, s21
	s_mov_b32 s69, s71
	s_movk_i32 s40, 0x100
	s_mov_b32 s45, 0x2d800
	s_lshr_b32 s33, s2, 5
	s_cselect_b32 s28, 0, 1
	s_wait_kmcnt 0x0
	s_ashr_i32 s21, s25, 31
	s_ashr_i32 s27, s26, 31
	s_lshr_b32 s22, s21, 24
	s_not_b32 s23, s20
	s_add_co_i32 s22, s25, s22
	s_mul_u64 s[2:3], s[26:27], s[70:71]
	s_lshr_b32 s22, s22, 8
	s_mov_b32 s20, s25
	s_add_co_i32 s26, s22, s23
	s_add_nc_u64 s[22:23], s[2:3], s[68:69]
	s_lshl_b32 s2, s26, 8
	s_mul_u64 s[34:35], s[22:23], s[20:21]
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s20, s21, 25
	s_add_nc_u64 s[26:27], s[34:35], s[2:3]
	s_add_co_i32 s20, s25, s20
	s_mul_u64 s[30:31], s[26:27], 0xc0
	s_lshl_b64 s[26:27], s[26:27], 3
	s_add_co_i32 s3, s2, 0x100
	s_add_nc_u64 s[6:7], s[26:27], s[6:7]
	s_ashr_i32 s26, s20, 7
	s_ashr_i32 s3, s3, 7
	s_ashr_i32 s27, s26, 31
	s_lshl_b64 s[60:61], s[34:35], 3
	s_mul_u64 s[22:23], s[26:27], s[22:23]
	s_lshl_b64 s[26:27], s[34:35], 7
	s_lshl_b64 s[22:23], s[22:23], 9
	s_add_nc_u64 s[4:5], s[30:31], s[4:5]
	s_cmp_eq_u32 s33, 1
	s_mov_b32 s46, s4
	s_cselect_b32 s44, 1, 0
	s_or_b32 s47, s5, 0x80000000
	s_movk_i32 s41, 0xc0
	s_mov_b32 s39, 0x800000
	s_mov_b32 s38, 0x1007fff
	s_mov_b32 s37, 0xffff0000
	s_mov_b32 s36, 0x27100000
	s_mov_b32 s42, s71
	s_mov_b32 s43, s71
	s_cmp_eq_u32 s33, 2
	tensor_load_to_lds s[44:47], s[36:43]
	s_cselect_b32 s44, 1, 0
	s_cmp_eq_u32 s33, 3
	s_add_nc_u64 s[46:47], s[4:5], 0x80
	s_cselect_b32 s4, 1, 0
	s_add_co_i32 s69, s3, -1
	s_mul_u64 s[30:31], s[34:35], 0xc0
	s_min_i32 s62, s69, 0
	s_add_nc_u64 s[72:73], s[30:31], s[8:9]
	s_ashr_i32 s63, s62, 31
	s_bitset1_b32 s47, 31
	s_mul_u64 s[8:9], s[62:63], 0x6000
	s_mov_b32 s45, 0x2d880
	s_add_nc_u64 s[8:9], s[8:9], s[72:73]
	s_mov_b32 s55, 0x400000
	s_mov_b32 s52, 0x46d00000
	s_mov_b32 s53, s37
	s_mov_b32 s54, s38
	s_mov_b32 s56, s40
	s_mov_b32 s57, s41
	s_mov_b32 s58, s71
	s_mov_b32 s59, s71
	s_lshl_b64 s[64:65], s[62:63], 7
	s_or_b32 s31, s9, 0x80000000
	s_mov_b32 s49, 8
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[44:47], s[52:59]
	s_bitset1_b32 s7, 31
	s_mov_b32 s5, 0x3a800
	s_mov_b32 s47, 0x80000
	s_mov_b32 s44, s71
	s_mov_b32 s45, s37
	s_mov_b32 s46, s38
	s_mov_b32 s48, s40
	s_mov_b32 s50, s71
	s_mov_b32 s51, s71
	s_cmp_gt_i32 s3, 0
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[4:7], s[44:51]
	s_cselect_b32 s4, -1, 0
	s_mov_b32 s76, 0x807fff
	s_and_b32 vcc_lo, s4, exec_lo
	s_mov_b32 s29, s71
	s_mov_b32 s30, s8
	s_cselect_b32 s38, s76, 0x17fff
	s_movk_i32 s40, 0x80
	s_add_nc_u64 s[74:75], s[60:61], s[10:11]
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[28:31], s[36:43]
	s_add_nc_u64 s[30:31], s[8:9], 0x80
	s_mov_b32 s29, s40
	s_bitset1_b32 s31, 31
	s_mov_b32 s54, s38
	s_mov_b32 s56, s40
	s_lshl_b64 s[4:5], s[62:63], 10
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[28:31], s[52:59]
	s_add_nc_u64 s[30:31], s[4:5], s[74:75]
	s_add_nc_u64 s[12:13], s[26:27], s[12:13]
	s_bitset1_b32 s31, 31
	s_movk_i32 s29, 0x6800
	s_mov_b32 s46, s38
	s_mov_b32 s48, s40
	s_and_b32 s26, s21, 0xffff
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[28:31], s[44:51]
	s_add_nc_u64 s[30:31], s[64:65], s[12:13]
	s_add_nc_u64 s[14:15], s[22:23], s[14:15]
	s_bitset1_b32 s31, 31
	s_movk_i32 s29, 0x6c00
	s_mov_b32 s4, 0x7100000
	s_mov_b32 s5, s37
	s_mov_b32 s6, s38
	s_mov_b32 s7, s39
	s_mov_b32 s8, s40
	s_mov_b32 s9, s25
	s_mov_b32 s10, s26
	s_mov_b32 s11, s71
	s_lshl_b64 s[20:21], s[62:63], 9
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[28:31], s[4:11]
	s_add_nc_u64 s[30:31], s[20:21], s[14:15]
	s_min_i32 s20, s69, 1
	s_bitset1_b32 s31, 31
	s_ashr_i32 s21, s20, 31
	s_mov_b32 s29, 0xb400
	s_mul_u64 s[6:7], s[20:21], 0x6000
	s_mov_b32 s65, 4
	s_mov_b32 s63, 0x40000
	s_mov_b32 s60, s71
	s_mov_b32 s61, s37
	s_mov_b32 s62, s38
	s_mov_b32 s64, s40
	s_mov_b32 s66, s71
	s_mov_b32 s67, s71
	s_add_nc_u64 s[6:7], s[6:7], s[72:73]
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[28:31], s[60:67]
	s_lshl_b64 s[8:9], s[20:21], 7
	s_or_b32 s31, s7, 0x80000000
	s_cmp_gt_i32 s3, 1
	s_mov_b32 s29, 0xb600
	s_mov_b32 s30, s6
	s_cselect_b32 s38, s76, 0x17fff
	v_and_b32_e32 v25, 15, v0
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[28:31], s[36:43]
	s_add_nc_u64 s[30:31], s[6:7], 0x80
	s_mov_b32 s29, 0xb680
	s_bitset1_b32 s31, 31
	s_mov_b32 s54, s38
	s_lshl_b64 s[6:7], s[20:21], 10
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[28:31], s[52:59]
	s_add_nc_u64 s[30:31], s[6:7], s[74:75]
	s_mov_b32 s29, 0x11e00
	s_bitset1_b32 s31, 31
	s_mov_b32 s46, s38
	s_mov_b32 s6, s38
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[28:31], s[44:51]
	s_add_nc_u64 s[30:31], s[8:9], s[12:13]
	s_mov_b32 s29, 0x12200
	s_bitset1_b32 s31, 31
	s_mov_b32 s7, s39
	s_mov_b32 s8, s40
	s_mov_b32 s9, s25
	s_lshl_b64 s[20:21], s[20:21], 9
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[28:31], s[4:11]
	s_add_nc_u64 s[30:31], s[20:21], s[14:15]
	s_min_i32 s20, s69, 2
	s_bitset1_b32 s31, 31
	s_ashr_i32 s21, s20, 31
	s_mov_b32 s29, 0x16a00
	s_mul_u64 s[6:7], s[20:21], 0x6000
	s_mov_b32 s62, s38
	s_add_nc_u64 s[6:7], s[6:7], s[72:73]
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[28:31], s[60:67]
	s_lshl_b64 s[8:9], s[20:21], 7
	s_or_b32 s31, s7, 0x80000000
	s_cmp_gt_i32 s3, 2
	s_mov_b32 s29, 0x16c00
	s_mov_b32 s30, s6
	s_cselect_b32 s38, s76, 0x17fff
	v_bfe_u32 v37, v0, 4, 1
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[28:31], s[36:43]
	s_add_nc_u64 s[30:31], s[6:7], 0x80
	s_mov_b32 s29, 0x16c80
	s_bitset1_b32 s31, 31
	s_mov_b32 s54, s38
	s_lshl_b64 s[6:7], s[20:21], 10
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[28:31], s[52:59]
	s_add_nc_u64 s[30:31], s[6:7], s[74:75]
	s_mov_b32 s29, 0x1d400
	s_bitset1_b32 s31, 31
	s_mov_b32 s46, s38
	s_mov_b32 s6, s38
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[28:31], s[44:51]
	s_add_nc_u64 s[30:31], s[8:9], s[12:13]
	s_mov_b32 s29, 0x1d800
	s_bitset1_b32 s31, 31
	s_mov_b32 s7, s39
	s_mov_b32 s8, s40
	s_mov_b32 s9, s25
	s_lshl_b64 s[20:21], s[20:21], 9
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[28:31], s[4:11]
	s_add_nc_u64 s[30:31], s[20:21], s[14:15]
	s_mov_b32 s29, 0x22000
	s_bitset1_b32 s31, 31
	s_mov_b32 s36, s71
	s_mov_b32 s39, s63
	s_mov_b32 s41, s65
	s_mov_b32 s5, -1
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[28:31], s[36:43]
	s_wait_tensorcnt 0x0
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x0
	s_barrier_signal -1
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_barrier_wait -1
	; sched_barrier mask(0x00000000)
	v_lshlrev_b32_e32 v27, 3, v37
	s_lshl_b32 s4, s33, 4
	s_cbranch_vccnz .LBB0_3
; %bb.1:                                ; %.._crit_edge_crit_edge
	s_add_co_i32 s33, s4, s2
	v_lshlrev_b32_e32 v2, 3, v37
	v_or_b32_e32 v26, s33, v25
	s_mov_b32 s5, s71
	s_and_b32 s5, s5, exec_lo
	s_cselect_b32 s5, 1, 0
	s_cmp_lg_u32 s5, 1
	s_cbranch_scc0 .LBB0_4
.LBB0_2:
	s_set_vgpr_msb 0xc0                     ;  msbs: dst=3 src0=0 src1=0 src2=0
	v_dual_mov_b32 v15 /*v783*/, 0 :: v_dual_mov_b32 v248 /*v1016*/, 0xf149f2ca
	s_mov_b32 s2, 0xf149f2ca
	v_dual_mov_b32 v17 /*v785*/, s2 :: v_dual_mov_b32 v189 /*v957*/, s2
	s_mov_b32 s2, 0xf149f2ca
	s_set_vgpr_msb 0xc003                   ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_dual_mov_b32 v17, s2 :: v_dual_mov_b32 v19, v15 /*v783*/
	s_set_vgpr_msb 0x3c3                    ;  msbs: dst=3 src0=3 src1=0 src2=0
	v_dual_mov_b32 v14 /*v782*/, v15 /*v783*/ :: v_dual_mov_b32 v13 /*v781*/, v15 /*v783*/
	v_dual_mov_b32 v12 /*v780*/, v15 /*v783*/ :: v_dual_mov_b32 v11 /*v779*/, v15 /*v783*/
	v_dual_mov_b32 v10 /*v778*/, v15 /*v783*/ :: v_dual_mov_b32 v9 /*v777*/, v15 /*v783*/
	v_dual_mov_b32 v8 /*v776*/, v15 /*v783*/ :: v_dual_mov_b32 v7 /*v775*/, v15 /*v783*/
	v_dual_mov_b32 v6 /*v774*/, v15 /*v783*/ :: v_dual_mov_b32 v5 /*v773*/, v15 /*v783*/
	v_dual_mov_b32 v4 /*v772*/, v15 /*v783*/ :: v_dual_mov_b32 v3 /*v771*/, v15 /*v783*/
	v_dual_mov_b32 v2 /*v770*/, v15 /*v783*/ :: v_dual_mov_b32 v1 /*v769*/, v15 /*v783*/
	v_mov_b32_e32 v0 /*v768*/, v15 /*v783*/
	s_set_vgpr_msb 0xc383                   ;  msbs: dst=2 src0=3 src1=0 src2=0
	v_dual_mov_b32 v255 /*v767*/, v15 /*v783*/ :: v_dual_mov_b32 v254 /*v766*/, v15 /*v783*/
	v_dual_mov_b32 v253 /*v765*/, v15 /*v783*/ :: v_dual_mov_b32 v252 /*v764*/, v15 /*v783*/
	v_dual_mov_b32 v251 /*v763*/, v15 /*v783*/ :: v_dual_mov_b32 v250 /*v762*/, v15 /*v783*/
	v_dual_mov_b32 v249 /*v761*/, v15 /*v783*/ :: v_dual_mov_b32 v248 /*v760*/, v15 /*v783*/
	v_dual_mov_b32 v247 /*v759*/, v15 /*v783*/ :: v_dual_mov_b32 v246 /*v758*/, v15 /*v783*/
	v_dual_mov_b32 v245 /*v757*/, v15 /*v783*/ :: v_dual_mov_b32 v244 /*v756*/, v15 /*v783*/
	v_dual_mov_b32 v243 /*v755*/, v15 /*v783*/ :: v_dual_mov_b32 v242 /*v754*/, v15 /*v783*/
	v_dual_mov_b32 v241 /*v753*/, v15 /*v783*/ :: v_dual_mov_b32 v240 /*v752*/, v15 /*v783*/
	v_dual_mov_b32 v239 /*v751*/, v15 /*v783*/ :: v_dual_mov_b32 v238 /*v750*/, v15 /*v783*/
	v_dual_mov_b32 v237 /*v749*/, v15 /*v783*/ :: v_dual_mov_b32 v236 /*v748*/, v15 /*v783*/
	v_dual_mov_b32 v235 /*v747*/, v15 /*v783*/ :: v_dual_mov_b32 v234 /*v746*/, v15 /*v783*/
	v_dual_mov_b32 v233 /*v745*/, v15 /*v783*/ :: v_dual_mov_b32 v232 /*v744*/, v15 /*v783*/
	v_dual_mov_b32 v231 /*v743*/, v15 /*v783*/ :: v_dual_mov_b32 v230 /*v742*/, v15 /*v783*/
	v_dual_mov_b32 v229 /*v741*/, v15 /*v783*/ :: v_dual_mov_b32 v228 /*v740*/, v15 /*v783*/
	v_dual_mov_b32 v227 /*v739*/, v15 /*v783*/ :: v_dual_mov_b32 v226 /*v738*/, v15 /*v783*/
	v_dual_mov_b32 v225 /*v737*/, v15 /*v783*/ :: v_dual_mov_b32 v224 /*v736*/, v15 /*v783*/
	v_dual_mov_b32 v223 /*v735*/, v15 /*v783*/ :: v_dual_mov_b32 v222 /*v734*/, v15 /*v783*/
	v_dual_mov_b32 v221 /*v733*/, v15 /*v783*/ :: v_dual_mov_b32 v220 /*v732*/, v15 /*v783*/
	v_dual_mov_b32 v219 /*v731*/, v15 /*v783*/ :: v_dual_mov_b32 v218 /*v730*/, v15 /*v783*/
	v_dual_mov_b32 v217 /*v729*/, v15 /*v783*/ :: v_dual_mov_b32 v216 /*v728*/, v15 /*v783*/
	v_dual_mov_b32 v215 /*v727*/, v15 /*v783*/ :: v_dual_mov_b32 v214 /*v726*/, v15 /*v783*/
	v_dual_mov_b32 v213 /*v725*/, v15 /*v783*/ :: v_dual_mov_b32 v212 /*v724*/, v15 /*v783*/
	v_dual_mov_b32 v211 /*v723*/, v15 /*v783*/ :: v_dual_mov_b32 v210 /*v722*/, v15 /*v783*/
	v_dual_mov_b32 v209 /*v721*/, v15 /*v783*/ :: v_dual_mov_b32 v208 /*v720*/, v15 /*v783*/
	v_dual_mov_b32 v207 /*v719*/, v15 /*v783*/ :: v_dual_mov_b32 v206 /*v718*/, v15 /*v783*/
	v_dual_mov_b32 v205 /*v717*/, v15 /*v783*/ :: v_dual_mov_b32 v204 /*v716*/, v15 /*v783*/
	v_dual_mov_b32 v203 /*v715*/, v15 /*v783*/ :: v_dual_mov_b32 v202 /*v714*/, v15 /*v783*/
	v_dual_mov_b32 v201 /*v713*/, v15 /*v783*/ :: v_dual_mov_b32 v200 /*v712*/, v15 /*v783*/
	v_dual_mov_b32 v199 /*v711*/, v15 /*v783*/ :: v_dual_mov_b32 v198 /*v710*/, v15 /*v783*/
	v_dual_mov_b32 v197 /*v709*/, v15 /*v783*/ :: v_dual_mov_b32 v196 /*v708*/, v15 /*v783*/
	v_dual_mov_b32 v195 /*v707*/, v15 /*v783*/ :: v_dual_mov_b32 v194 /*v706*/, v15 /*v783*/
	v_dual_mov_b32 v193 /*v705*/, v15 /*v783*/ :: v_dual_mov_b32 v192 /*v704*/, v15 /*v783*/
	v_dual_mov_b32 v191 /*v703*/, v15 /*v783*/ :: v_dual_mov_b32 v190 /*v702*/, v15 /*v783*/
	v_dual_mov_b32 v189 /*v701*/, v15 /*v783*/ :: v_dual_mov_b32 v188 /*v700*/, v15 /*v783*/
	v_dual_mov_b32 v187 /*v699*/, v15 /*v783*/ :: v_dual_mov_b32 v186 /*v698*/, v15 /*v783*/
	v_dual_mov_b32 v185 /*v697*/, v15 /*v783*/ :: v_dual_mov_b32 v184 /*v696*/, v15 /*v783*/
	v_dual_mov_b32 v183 /*v695*/, v15 /*v783*/ :: v_dual_mov_b32 v182 /*v694*/, v15 /*v783*/
	v_dual_mov_b32 v181 /*v693*/, v15 /*v783*/ :: v_dual_mov_b32 v180 /*v692*/, v15 /*v783*/
	v_dual_mov_b32 v179 /*v691*/, v15 /*v783*/ :: v_dual_mov_b32 v178 /*v690*/, v15 /*v783*/
	v_dual_mov_b32 v177 /*v689*/, v15 /*v783*/ :: v_dual_mov_b32 v176 /*v688*/, v15 /*v783*/
	v_dual_mov_b32 v175 /*v687*/, v15 /*v783*/ :: v_dual_mov_b32 v174 /*v686*/, v15 /*v783*/
	v_dual_mov_b32 v173 /*v685*/, v15 /*v783*/ :: v_dual_mov_b32 v172 /*v684*/, v15 /*v783*/
	v_dual_mov_b32 v171 /*v683*/, v15 /*v783*/ :: v_dual_mov_b32 v170 /*v682*/, v15 /*v783*/
	v_dual_mov_b32 v169 /*v681*/, v15 /*v783*/ :: v_dual_mov_b32 v168 /*v680*/, v15 /*v783*/
	v_dual_mov_b32 v167 /*v679*/, v15 /*v783*/ :: v_dual_mov_b32 v166 /*v678*/, v15 /*v783*/
	v_dual_mov_b32 v165 /*v677*/, v15 /*v783*/ :: v_dual_mov_b32 v164 /*v676*/, v15 /*v783*/
	v_dual_mov_b32 v163 /*v675*/, v15 /*v783*/ :: v_dual_mov_b32 v162 /*v674*/, v15 /*v783*/
	v_dual_mov_b32 v161 /*v673*/, v15 /*v783*/ :: v_dual_mov_b32 v160 /*v672*/, v15 /*v783*/
	v_dual_mov_b32 v159 /*v671*/, v15 /*v783*/ :: v_dual_mov_b32 v158 /*v670*/, v15 /*v783*/
	v_dual_mov_b32 v157 /*v669*/, v15 /*v783*/ :: v_dual_mov_b32 v156 /*v668*/, v15 /*v783*/
	v_dual_mov_b32 v155 /*v667*/, v15 /*v783*/ :: v_dual_mov_b32 v154 /*v666*/, v15 /*v783*/
	v_dual_mov_b32 v153 /*v665*/, v15 /*v783*/ :: v_dual_mov_b32 v152 /*v664*/, v15 /*v783*/
	v_dual_mov_b32 v151 /*v663*/, v15 /*v783*/ :: v_dual_mov_b32 v150 /*v662*/, v15 /*v783*/
	v_dual_mov_b32 v149 /*v661*/, v15 /*v783*/ :: v_dual_mov_b32 v148 /*v660*/, v15 /*v783*/
	v_dual_mov_b32 v147 /*v659*/, v15 /*v783*/ :: v_dual_mov_b32 v146 /*v658*/, v15 /*v783*/
	v_dual_mov_b32 v145 /*v657*/, v15 /*v783*/ :: v_dual_mov_b32 v144 /*v656*/, v15 /*v783*/
	v_dual_mov_b32 v143 /*v655*/, v15 /*v783*/ :: v_dual_mov_b32 v142 /*v654*/, v15 /*v783*/
	v_dual_mov_b32 v141 /*v653*/, v15 /*v783*/ :: v_dual_mov_b32 v140 /*v652*/, v15 /*v783*/
	v_dual_mov_b32 v139 /*v651*/, v15 /*v783*/ :: v_dual_mov_b32 v138 /*v650*/, v15 /*v783*/
	v_dual_mov_b32 v137 /*v649*/, v15 /*v783*/ :: v_dual_mov_b32 v136 /*v648*/, v15 /*v783*/
	v_dual_mov_b32 v135 /*v647*/, v15 /*v783*/ :: v_dual_mov_b32 v134 /*v646*/, v15 /*v783*/
	v_dual_mov_b32 v133 /*v645*/, v15 /*v783*/ :: v_dual_mov_b32 v132 /*v644*/, v15 /*v783*/
	v_dual_mov_b32 v131 /*v643*/, v15 /*v783*/ :: v_dual_mov_b32 v130 /*v642*/, v15 /*v783*/
	v_dual_mov_b32 v129 /*v641*/, v15 /*v783*/ :: v_dual_mov_b32 v128 /*v640*/, v15 /*v783*/
	v_dual_mov_b32 v127 /*v639*/, v15 /*v783*/ :: v_dual_mov_b32 v126 /*v638*/, v15 /*v783*/
	v_dual_mov_b32 v125 /*v637*/, v15 /*v783*/ :: v_dual_mov_b32 v124 /*v636*/, v15 /*v783*/
	v_dual_mov_b32 v123 /*v635*/, v15 /*v783*/ :: v_dual_mov_b32 v122 /*v634*/, v15 /*v783*/
	v_dual_mov_b32 v121 /*v633*/, v15 /*v783*/ :: v_dual_mov_b32 v120 /*v632*/, v15 /*v783*/
	v_dual_mov_b32 v119 /*v631*/, v15 /*v783*/ :: v_dual_mov_b32 v118 /*v630*/, v15 /*v783*/
	v_dual_mov_b32 v117 /*v629*/, v15 /*v783*/ :: v_dual_mov_b32 v116 /*v628*/, v15 /*v783*/
	v_dual_mov_b32 v115 /*v627*/, v15 /*v783*/ :: v_dual_mov_b32 v114 /*v626*/, v15 /*v783*/
	v_dual_mov_b32 v113 /*v625*/, v15 /*v783*/ :: v_dual_mov_b32 v112 /*v624*/, v15 /*v783*/
	v_dual_mov_b32 v111 /*v623*/, v15 /*v783*/ :: v_dual_mov_b32 v110 /*v622*/, v15 /*v783*/
	v_dual_mov_b32 v109 /*v621*/, v15 /*v783*/ :: v_dual_mov_b32 v108 /*v620*/, v15 /*v783*/
	v_dual_mov_b32 v107 /*v619*/, v15 /*v783*/ :: v_dual_mov_b32 v106 /*v618*/, v15 /*v783*/
	v_dual_mov_b32 v105 /*v617*/, v15 /*v783*/ :: v_dual_mov_b32 v104 /*v616*/, v15 /*v783*/
	v_dual_mov_b32 v103 /*v615*/, v15 /*v783*/ :: v_dual_mov_b32 v102 /*v614*/, v15 /*v783*/
	v_dual_mov_b32 v101 /*v613*/, v15 /*v783*/ :: v_dual_mov_b32 v100 /*v612*/, v15 /*v783*/
	v_dual_mov_b32 v99 /*v611*/, v15 /*v783*/ :: v_dual_mov_b32 v98 /*v610*/, v15 /*v783*/
	v_dual_mov_b32 v97 /*v609*/, v15 /*v783*/ :: v_dual_mov_b32 v96 /*v608*/, v15 /*v783*/
	v_dual_mov_b32 v95 /*v607*/, v15 /*v783*/ :: v_dual_mov_b32 v94 /*v606*/, v15 /*v783*/
	v_dual_mov_b32 v93 /*v605*/, v15 /*v783*/ :: v_dual_mov_b32 v92 /*v604*/, v15 /*v783*/
	v_dual_mov_b32 v91 /*v603*/, v15 /*v783*/ :: v_dual_mov_b32 v90 /*v602*/, v15 /*v783*/
	v_dual_mov_b32 v89 /*v601*/, v15 /*v783*/ :: v_dual_mov_b32 v88 /*v600*/, v15 /*v783*/
	v_dual_mov_b32 v87 /*v599*/, v15 /*v783*/ :: v_dual_mov_b32 v86 /*v598*/, v15 /*v783*/
	v_dual_mov_b32 v85 /*v597*/, v15 /*v783*/ :: v_dual_mov_b32 v84 /*v596*/, v15 /*v783*/
	v_dual_mov_b32 v83 /*v595*/, v15 /*v783*/ :: v_dual_mov_b32 v82 /*v594*/, v15 /*v783*/
	v_dual_mov_b32 v81 /*v593*/, v15 /*v783*/ :: v_dual_mov_b32 v80 /*v592*/, v15 /*v783*/
	v_dual_mov_b32 v79 /*v591*/, v15 /*v783*/ :: v_dual_mov_b32 v78 /*v590*/, v15 /*v783*/
	v_dual_mov_b32 v77 /*v589*/, v15 /*v783*/ :: v_dual_mov_b32 v76 /*v588*/, v15 /*v783*/
	v_dual_mov_b32 v75 /*v587*/, v15 /*v783*/ :: v_dual_mov_b32 v74 /*v586*/, v15 /*v783*/
	v_dual_mov_b32 v73 /*v585*/, v15 /*v783*/ :: v_dual_mov_b32 v72 /*v584*/, v15 /*v783*/
	v_dual_mov_b32 v71 /*v583*/, v15 /*v783*/ :: v_dual_mov_b32 v70 /*v582*/, v15 /*v783*/
	v_dual_mov_b32 v69 /*v581*/, v15 /*v783*/ :: v_dual_mov_b32 v68 /*v580*/, v15 /*v783*/
	v_dual_mov_b32 v67 /*v579*/, v15 /*v783*/ :: v_dual_mov_b32 v66 /*v578*/, v15 /*v783*/
	v_dual_mov_b32 v65 /*v577*/, v15 /*v783*/ :: v_dual_mov_b32 v64 /*v576*/, v15 /*v783*/
	v_dual_mov_b32 v63 /*v575*/, v15 /*v783*/ :: v_dual_mov_b32 v62 /*v574*/, v15 /*v783*/
	v_dual_mov_b32 v61 /*v573*/, v15 /*v783*/ :: v_dual_mov_b32 v60 /*v572*/, v15 /*v783*/
	v_dual_mov_b32 v59 /*v571*/, v15 /*v783*/ :: v_dual_mov_b32 v58 /*v570*/, v15 /*v783*/
	v_dual_mov_b32 v57 /*v569*/, v15 /*v783*/ :: v_dual_mov_b32 v56 /*v568*/, v15 /*v783*/
	v_dual_mov_b32 v55 /*v567*/, v15 /*v783*/ :: v_dual_mov_b32 v54 /*v566*/, v15 /*v783*/
	v_dual_mov_b32 v53 /*v565*/, v15 /*v783*/ :: v_dual_mov_b32 v52 /*v564*/, v15 /*v783*/
	v_dual_mov_b32 v51 /*v563*/, v15 /*v783*/ :: v_dual_mov_b32 v50 /*v562*/, v15 /*v783*/
	v_dual_mov_b32 v49 /*v561*/, v15 /*v783*/ :: v_dual_mov_b32 v48 /*v560*/, v15 /*v783*/
	v_dual_mov_b32 v47 /*v559*/, v15 /*v783*/ :: v_dual_mov_b32 v46 /*v558*/, v15 /*v783*/
	v_dual_mov_b32 v45 /*v557*/, v15 /*v783*/ :: v_dual_mov_b32 v44 /*v556*/, v15 /*v783*/
	v_dual_mov_b32 v43 /*v555*/, v15 /*v783*/ :: v_dual_mov_b32 v42 /*v554*/, v15 /*v783*/
	v_dual_mov_b32 v41 /*v553*/, v15 /*v783*/ :: v_dual_mov_b32 v40 /*v552*/, v15 /*v783*/
	v_dual_mov_b32 v39 /*v551*/, v15 /*v783*/ :: v_dual_mov_b32 v38 /*v550*/, v15 /*v783*/
	v_dual_mov_b32 v37 /*v549*/, v15 /*v783*/ :: v_dual_mov_b32 v36 /*v548*/, v15 /*v783*/
	v_dual_mov_b32 v35 /*v547*/, v15 /*v783*/ :: v_dual_mov_b32 v34 /*v546*/, v15 /*v783*/
	v_dual_mov_b32 v33 /*v545*/, v15 /*v783*/ :: v_dual_mov_b32 v32 /*v544*/, v15 /*v783*/
	v_dual_mov_b32 v31 /*v543*/, v15 /*v783*/ :: v_dual_mov_b32 v30 /*v542*/, v15 /*v783*/
	v_dual_mov_b32 v29 /*v541*/, v15 /*v783*/ :: v_dual_mov_b32 v28 /*v540*/, v15 /*v783*/
	v_dual_mov_b32 v27 /*v539*/, v15 /*v783*/ :: v_dual_mov_b32 v26 /*v538*/, v15 /*v783*/
	v_dual_mov_b32 v25 /*v537*/, v15 /*v783*/ :: v_dual_mov_b32 v24 /*v536*/, v15 /*v783*/
	v_dual_mov_b32 v23 /*v535*/, v15 /*v783*/ :: v_dual_mov_b32 v22 /*v534*/, v15 /*v783*/
	v_dual_mov_b32 v21 /*v533*/, v15 /*v783*/ :: v_dual_mov_b32 v20 /*v532*/, v15 /*v783*/
	v_dual_mov_b32 v19 /*v531*/, v15 /*v783*/ :: v_dual_mov_b32 v18 /*v530*/, v15 /*v783*/
	v_dual_mov_b32 v17 /*v529*/, v15 /*v783*/ :: v_dual_mov_b32 v16 /*v528*/, v15 /*v783*/
	s_set_vgpr_msb 0x8303                   ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_dual_mov_b32 v20, v15 /*v783*/ :: v_dual_mov_b32 v21, v15 /*v783*/
	v_mov_b32_e32 v22, v15 /*v783*/
	s_set_vgpr_msb 0x300                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_branch .LBB0_23
.LBB0_3:
                                        ; implicit-def: $sgpr33
                                        ; implicit-def: $vgpr26
                                        ; implicit-def: $vgpr2
	s_and_b32 s5, s5, exec_lo
	s_cselect_b32 s5, 1, 0
	s_cmp_lg_u32 s5, 1
	s_cbranch_scc1 .LBB0_2
.LBB0_4:                                ; %.lr.ph
	s_mov_b32 s10, 0
	v_dual_mov_b32 v8, s10 :: v_dual_bitop2_b32 v1, s4, v25 bitop3:0x54
	v_dual_lshlrev_b32 v39, 6, v37 :: v_dual_bitop2_b32 v17, 31, v0 bitop3:0x40
	v_dual_lshlrev_b32 v41, 2, v37 :: v_dual_lshlrev_b32 v42, 5, v37
	v_mul_lo_u32 v16, 0xd0, v1
	v_dual_mov_b32 v56, s10 :: v_dual_mov_b32 v89, s10
	v_lshl_or_b32 v1, v1, 3, v41
	s_add_co_i32 s33, s4, s2
	v_dual_lshlrev_b32 v43, 3, v25 :: v_dual_bitop2_b32 v26, s33, v25 bitop3:0x54
	s_mov_b32 s2, 0x30c00
	v_dual_mov_b32 v13, s10 :: v_dual_add_nc_u32 v19, s2, v16
	v_dual_mov_b32 v15, s10 :: v_dual_add_nc_u32 v0, 0x2d800, v16
	v_dual_mov_b32 v9, s10 :: v_dual_mov_b32 v14, s10
	v_dual_add_nc_u32 v20, v19, v39 :: v_dual_add_nc_u32 v19, v19, v42
	v_dual_add_nc_u32 v2, v0, v39 :: v_dual_mov_b32 v12, s10
	s_set_vgpr_msb 0xc0                     ;  msbs: dst=3 src0=0 src1=0 src2=0
	ds_load_b128 v[98:101] /*v[866:869]*/, v20
	ds_load_b128 v[102:105] /*v[870:873]*/, v20 offset:16
	ds_load_b128 v[106:109] /*v[874:877]*/, v20 offset:32
	ds_load_b128 v[110:113] /*v[878:881]*/, v20 offset:48
	s_set_vgpr_msb 0xc000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51], v19 offset:128
	ds_load_b128 v[52:55], v19 offset:144
	v_add_nc_u32_e32 v19, 0x34000, v16
	v_dual_mov_b32 v61, s10 :: v_dual_add_nc_u32 v16, 0x37400, v16
	s_mov_b32 s2, 0x3a800
	v_dual_mov_b32 v88, s10 :: v_dual_add_nc_u32 v18, s2, v1
	v_dual_add_nc_u32 v4, v0, v42 :: v_dual_add_nc_u32 v20, v19, v39
	v_add_nc_u32_e32 v19, v19, v42
	ds_load_b128 v[64:67], v20
	ds_load_b128 v[68:71], v20 offset:16
	ds_load_b128 v[72:75], v20 offset:32
	ds_load_b128 v[76:79], v20 offset:48
	ds_load_b128 v[80:83], v19 offset:128
	ds_load_b128 v[84:87], v19 offset:144
	v_dual_mov_b32 v94, s10 :: v_dual_add_nc_u32 v19, v16, v39
	v_dual_mov_b32 v95, s10 :: v_dual_add_nc_u32 v16, v16, v42
	s_set_vgpr_msb 0xc0                     ;  msbs: dst=3 src0=0 src1=0 src2=0
	ds_load_b128 v[82:85] /*v[850:853]*/, v2
	ds_load_b128 v[86:89] /*v[854:857]*/, v2 offset:16
	ds_load_b128 v[90:93] /*v[858:861]*/, v2 offset:32
	ds_load_b128 v[94:97] /*v[862:865]*/, v2 offset:48
	s_set_vgpr_msb 0xc000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3], v4 offset:128
	ds_load_b128 v[4:7], v4 offset:144
	ds_load_b128 v[96:99], v19
	ds_load_b128 v[100:103], v19 offset:16
	ds_load_b128 v[104:107], v19 offset:32
	ds_load_b128 v[108:111], v19 offset:48
	ds_load_b128 v[112:115], v16 offset:128
	ds_load_b128 v[116:119], v16 offset:144
	ds_load_2addr_stride64_b32 v[28:29], v18 offset1:2
	ds_load_2addr_stride64_b32 v[30:31], v18 offset0:4 offset1:6
	s_movk_i32 s5, 0xd00
	s_mov_b32 s2, 0x180
	s_set_vgpr_msb 0xc0                     ;  msbs: dst=3 src0=0 src1=0 src2=0
	v_dual_mov_b32 v16 /*v784*/, 0 :: v_dual_bitop2_b32 v116 /*v884*/, s2, v43 bitop3:0x54
	s_set_vgpr_msb 0xc000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mad_u32_u24 v45, 0xd0, v25, s5
	s_movk_i32 s5, 0x1a00
	s_set_vgpr_msb 0xc0                     ;  msbs: dst=3 src0=0 src1=0 src2=0
	v_or_b32_e32 v118 /*v886*/, 0x200, v43
	s_set_vgpr_msb 0xc000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mad_u32_u24 v47, 0xd0, v25, s5
	s_movk_i32 s5, 0x2700
	s_mov_b32 s2, 0x280
	s_set_vgpr_msb 0xc3                     ;  msbs: dst=3 src0=3 src1=0 src2=0
	v_dual_mov_b32 v18 /*v786*/, v16 /*v784*/ :: v_dual_bitop2_b32 v120 /*v888*/, s2, v43 bitop3:0x54
	v_mad_u32_u24 v115 /*v883*/, 0xd0, v25, s5
	s_movk_i32 s5, 0x3400
	s_mov_b32 s2, 0x300
	v_dual_mov_b32 v19 /*v787*/, v16 /*v784*/ :: v_dual_bitop2_b32 v122 /*v890*/, s2, v43 bitop3:0x54
	v_mad_u32_u24 v117 /*v885*/, 0xd0, v25, s5
	s_movk_i32 s5, 0x4100
	v_or_b32_e32 v124 /*v892*/, 0x380, v43
	v_dual_mov_b32 v20 /*v788*/, v16 /*v784*/ :: v_dual_bitop2_b32 v127 /*v895*/, 2, v27 bitop3:0x54
	v_dual_mov_b32 v21 /*v789*/, v16 /*v784*/ :: v_dual_bitop2_b32 v128 /*v896*/, 3, v27 bitop3:0x54
	v_dual_mov_b32 v22 /*v790*/, v16 /*v784*/ :: v_dual_bitop2_b32 v129 /*v897*/, 4, v27 bitop3:0x54
	v_dual_mov_b32 v23 /*v791*/, v16 /*v784*/ :: v_dual_mov_b32 v17 /*v785*/, v16 /*v784*/
	v_mad_u32_u24 v119 /*v887*/, 0xd0, v25, s5
	s_movk_i32 s5, 0x4e00
	s_add_co_i32 s61, s33, 64
	v_mad_u32_u24 v121 /*v889*/, 0xd0, v25, s5
	s_movk_i32 s5, 0x5b00
	s_add_co_i32 s62, s33, 0x80
	s_add_co_i32 s63, s33, 0xc0
	s_set_vgpr_msb 0xc383                   ;  msbs: dst=2 src0=3 src1=0 src2=0
	v_mov_b64_e32 v[16:17] /*v[528:529]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[18:19] /*v[530:531]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[20:21] /*v[532:533]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[22:23] /*v[534:535]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[24:25] /*v[536:537]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[26:27] /*v[538:539]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[28:29] /*v[540:541]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[30:31] /*v[542:543]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[32:33] /*v[544:545]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[34:35] /*v[546:547]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[36:37] /*v[548:549]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[38:39] /*v[550:551]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[40:41] /*v[552:553]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[42:43] /*v[554:555]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[44:45] /*v[556:557]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[46:47] /*v[558:559]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[48:49] /*v[560:561]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[50:51] /*v[562:563]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[52:53] /*v[564:565]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[54:55] /*v[566:567]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[56:57] /*v[568:569]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[58:59] /*v[570:571]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[60:61] /*v[572:573]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[62:63] /*v[574:575]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[64:65] /*v[576:577]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[66:67] /*v[578:579]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[68:69] /*v[580:581]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[70:71] /*v[582:583]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[72:73] /*v[584:585]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[74:75] /*v[586:587]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[76:77] /*v[588:589]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[78:79] /*v[590:591]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[80:81] /*v[592:593]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[82:83] /*v[594:595]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[84:85] /*v[596:597]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[86:87] /*v[598:599]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[88:89] /*v[600:601]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[90:91] /*v[602:603]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[92:93] /*v[604:605]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[94:95] /*v[606:607]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[96:97] /*v[608:609]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[98:99] /*v[610:611]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[100:101] /*v[612:613]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[102:103] /*v[614:615]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[104:105] /*v[616:617]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[106:107] /*v[618:619]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[108:109] /*v[620:621]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[110:111] /*v[622:623]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[112:113] /*v[624:625]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[114:115] /*v[626:627]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[116:117] /*v[628:629]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[118:119] /*v[630:631]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[120:121] /*v[632:633]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[122:123] /*v[634:635]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[124:125] /*v[636:637]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[126:127] /*v[638:639]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[128:129] /*v[640:641]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[130:131] /*v[642:643]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[132:133] /*v[644:645]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[134:135] /*v[646:647]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[136:137] /*v[648:649]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[138:139] /*v[650:651]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[140:141] /*v[652:653]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[142:143] /*v[654:655]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[144:145] /*v[656:657]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[146:147] /*v[658:659]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[148:149] /*v[660:661]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[150:151] /*v[662:663]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[152:153] /*v[664:665]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[154:155] /*v[666:667]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[156:157] /*v[668:669]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[158:159] /*v[670:671]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[160:161] /*v[672:673]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[162:163] /*v[674:675]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[164:165] /*v[676:677]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[166:167] /*v[678:679]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[168:169] /*v[680:681]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[170:171] /*v[682:683]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[172:173] /*v[684:685]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[174:175] /*v[686:687]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[176:177] /*v[688:689]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[178:179] /*v[690:691]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[180:181] /*v[692:693]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[182:183] /*v[694:695]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[184:185] /*v[696:697]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[186:187] /*v[698:699]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[188:189] /*v[700:701]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[190:191] /*v[702:703]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[192:193] /*v[704:705]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[194:195] /*v[706:707]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[196:197] /*v[708:709]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[198:199] /*v[710:711]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[200:201] /*v[712:713]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[202:203] /*v[714:715]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[204:205] /*v[716:717]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[206:207] /*v[718:719]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[208:209] /*v[720:721]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[210:211] /*v[722:723]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[212:213] /*v[724:725]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[214:215] /*v[726:727]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[216:217] /*v[728:729]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[218:219] /*v[730:731]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[220:221] /*v[732:733]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[222:223] /*v[734:735]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[224:225] /*v[736:737]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[226:227] /*v[738:739]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[228:229] /*v[740:741]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[230:231] /*v[742:743]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[232:233] /*v[744:745]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[234:235] /*v[746:747]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[236:237] /*v[748:749]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[238:239] /*v[750:751]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[240:241] /*v[752:753]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[242:243] /*v[754:755]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[244:245] /*v[756:757]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[246:247] /*v[758:759]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[248:249] /*v[760:761]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[250:251] /*v[762:763]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[252:253] /*v[764:765]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[254:255] /*v[766:767]*/, v[22:23] /*v[790:791]*/
	s_set_vgpr_msb 0x83c3                   ;  msbs: dst=3 src0=3 src1=0 src2=0
	v_mov_b64_e32 v[0:1] /*v[768:769]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[2:3] /*v[770:771]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[4:5] /*v[772:773]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[6:7] /*v[774:775]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[8:9] /*v[776:777]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[10:11] /*v[778:779]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[12:13] /*v[780:781]*/, v[20:21] /*v[788:789]*/
	v_mov_b64_e32 v[14:15] /*v[782:783]*/, v[22:23] /*v[790:791]*/
	s_set_vgpr_msb 0xc300                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v10, s10 :: v_dual_mov_b32 v11, s10
	v_dual_mov_b32 v57, s10 :: v_dual_mov_b32 v58, s10
	v_dual_mov_b32 v59, s10 :: v_dual_mov_b32 v60, s10
	v_dual_mov_b32 v62, s10 :: v_dual_mov_b32 v63, s10
	v_dual_mov_b32 v90, s10 :: v_dual_mov_b32 v91, s10
	v_dual_mov_b32 v92, s10 :: v_dual_mov_b32 v93, s10
	v_dual_mov_b32 v120, s10 :: v_dual_mov_b32 v121, s10
	v_dual_mov_b32 v122, s10 :: v_dual_mov_b32 v123, s10
	v_dual_mov_b32 v124, s10 :: v_dual_mov_b32 v125, s10
	v_dual_mov_b32 v126, s10 :: v_dual_mov_b32 v127, s10
	v_mul_u32_u24_e32 v44, 0xd0, v25
	v_or_b32_e32 v46, 0x80, v43
	s_mov_b32 s2, 0x100
	s_set_vgpr_msb 0xc3                     ;  msbs: dst=3 src0=3 src1=0 src2=0
	v_dual_lshlrev_b32 v126 /*v894*/, 2, v17 :: v_dual_bitop2_b32 v114 /*v882*/, s2, v43 bitop3:0x54
	v_mad_u32_u24 v123 /*v891*/, 0xd0, v25, s5
	v_mul_u32_u24_e32 v125 /*v893*/, 0x90, v25
	s_mov_b32 s2, 0xf149f2ca
	v_dual_mov_b32 v188 /*v956*/, s2 :: v_dual_bitop2_b32 v130 /*v898*/, 5, v27 bitop3:0x54
	v_dual_mov_b32 v185 /*v953*/, v16 /*v784*/ :: v_dual_bitop2_b32 v131 /*v899*/, 6, v27 bitop3:0x54
	v_dual_mov_b32 v186 /*v954*/, v16 /*v784*/ :: v_dual_bitop2_b32 v132 /*v900*/, 7, v27 bitop3:0x54
	v_dual_mov_b32 v187 /*v955*/, v16 /*v784*/ :: v_dual_bitop2_b32 v133 /*v901*/, 16, v27 bitop3:0x54
	v_or_b32_e32 v134 /*v902*/, 18, v27
	v_or_b32_e32 v135 /*v903*/, 19, v27
	v_or_b32_e32 v136 /*v904*/, 20, v27
	v_or_b32_e32 v137 /*v905*/, 21, v27
	v_or_b32_e32 v138 /*v906*/, 22, v27
	v_or_b32_e32 v139 /*v907*/, 23, v27
	v_or_b32_e32 v140 /*v908*/, 32, v27
	v_or_b32_e32 v141 /*v909*/, 34, v27
	v_or_b32_e32 v142 /*v910*/, 35, v27
	v_or_b32_e32 v143 /*v911*/, 36, v27
	v_or_b32_e32 v144 /*v912*/, 37, v27
	v_or_b32_e32 v145 /*v913*/, 38, v27
	v_or_b32_e32 v146 /*v914*/, 39, v27
	v_or_b32_e32 v147 /*v915*/, 48, v27
	v_or_b32_e32 v148 /*v916*/, 50, v27
	v_or_b32_e32 v149 /*v917*/, 51, v27
	v_or_b32_e32 v150 /*v918*/, 52, v27
	v_or_b32_e32 v151 /*v919*/, 53, v27
	v_or_b32_e32 v152 /*v920*/, 54, v27
	v_or_b32_e32 v153 /*v921*/, 55, v27
	v_or_b32_e32 v154 /*v922*/, 64, v27
	v_or_b32_e32 v155 /*v923*/, 0x42, v27
	v_or_b32_e32 v156 /*v924*/, 0x43, v27
	v_or_b32_e32 v157 /*v925*/, 0x44, v27
	v_or_b32_e32 v158 /*v926*/, 0x45, v27
	v_or_b32_e32 v159 /*v927*/, 0x46, v27
	v_or_b32_e32 v160 /*v928*/, 0x47, v27
	v_or_b32_e32 v161 /*v929*/, 0x50, v27
	v_or_b32_e32 v162 /*v930*/, 0x52, v27
	v_or_b32_e32 v163 /*v931*/, 0x53, v27
	v_or_b32_e32 v164 /*v932*/, 0x54, v27
	v_or_b32_e32 v165 /*v933*/, 0x55, v27
	v_or_b32_e32 v166 /*v934*/, 0x56, v27
	v_or_b32_e32 v167 /*v935*/, 0x57, v27
	v_or_b32_e32 v168 /*v936*/, 0x60, v27
	v_or_b32_e32 v169 /*v937*/, 0x62, v27
	v_or_b32_e32 v170 /*v938*/, 0x63, v27
	v_or_b32_e32 v171 /*v939*/, 0x64, v27
	v_or_b32_e32 v172 /*v940*/, 0x65, v27
	v_or_b32_e32 v173 /*v941*/, 0x66, v27
	v_or_b32_e32 v174 /*v942*/, 0x67, v27
	v_or_b32_e32 v175 /*v943*/, 0x70, v27
	v_or_b32_e32 v176 /*v944*/, 0x72, v27
	v_or_b32_e32 v177 /*v945*/, 0x73, v27
	v_or_b32_e32 v178 /*v946*/, 0x74, v27
	v_or_b32_e32 v179 /*v947*/, 0x75, v27
	v_or_b32_e32 v180 /*v948*/, 0x76, v27
	v_or_b32_e32 v181 /*v949*/, 0x77, v27
	v_or_b32_e32 v182 /*v950*/, s61, v25
	v_or_b32_e32 v183 /*v951*/, s62, v25
	v_or_b32_e32 v184 /*v952*/, s63, v25
	s_set_vgpr_msb 0xc300                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mov_b32_e32 v16, 0x78787878
	s_mov_b32 s2, 0xf149f2ca
	v_dual_mov_b32 v40, s2 :: v_dual_mov_b32 v38, s2
	v_mov_b32_e32 v36, 0xf149f2ca
	s_mul_f32 s60, s24, 0x3fb8aa3b
	s_movk_i32 s8, 0x80
	s_mov_b32 s7, 0x800000
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s41, 4
	s_movk_i32 s9, 0xc0
	s_mov_b32 s20, 0x7100000
	s_mov_b32 s21, s5
	s_mov_b32 s23, s7
	s_mov_b32 s24, s8
	s_mov_b32 s27, s10
	s_mov_b32 s64, 0x807fff
	s_mov_b32 s4, 0x27100000
	s_mov_b32 s47, 0x400000
	s_mov_b32 s44, 0x46d00000
	s_mov_b32 s57, 8
	s_mov_b32 s55, 0x80000
	s_mov_b32 s39, 0x40000
	s_mov_b32 s65, 0x76543210
	s_mov_b32 s66, s10
	s_branch .LBB0_6
.LBB0_5:                                ; %_ZN2mx10static_forILi4EZ12k_mla_fwd_mxEUlT_E4_EEvOT0_.exit
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_cvt_scalef32_pk8_fp8_f32 v[0:1] /*v[256:257]*/, v[48:55] /*v[304:311]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[2:3] /*v[258:259]*/, v[56:63] /*v[312:319]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[4:5] /*v[260:261]*/, v[40:47] /*v[296:303]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[6:7] /*v[262:263]*/, v[32:39] /*v[288:295]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[8:9] /*v[264:265]*/, v[24:31] /*v[280:287]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[10:11] /*v[266:267]*/, v[16:23] /*v[272:279]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[12:13] /*v[268:269]*/, v[64:71] /*v[320:327]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[14:15] /*v[270:271]*/, v[72:79] /*v[328:335]*/, 0x3c000000
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_add_f32_e32 v19, v19, v20
	s_set_vgpr_msb 15                       ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_dual_add_f32 v20, v251 /*v1019*/, v252 /*v1020*/ :: v_dual_add_f32 v21, v249 /*v1017*/, v250 /*v1018*/
	v_add_f32_e32 v22, v190 /*v958*/, v191 /*v959*/
	s_set_vgpr_msb 0xfa4                    ;  msbs: dst=2 src0=0 src1=1 src2=2
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79] /*v[584:591]*/, v[128:143], v[0:15] /*v[256:271]*/, v[72:79] /*v[584:591]*/, v34, v16
	s_set_vgpr_msb 0xa40c                   ;  msbs: dst=0 src0=0 src1=3 src2=0
	v_dual_fmac_f32 v19, v18, v16 /*v784*/ :: v_dual_fmac_f32 v20, v40, v185 /*v953*/
	v_fmac_f32_e32 v21, v38, v186 /*v954*/
	s_set_vgpr_msb 0xca4                    ;  msbs: dst=2 src0=0 src1=1 src2=2
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71] /*v[576:583]*/, v[144:159], v[0:15] /*v[256:271]*/, v[64:71] /*v[576:583]*/, v34, v16 matrix_a_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa40c                   ;  msbs: dst=0 src0=0 src1=3 src2=0
	v_fmac_f32_e32 v22, v36, v187 /*v955*/
	s_set_vgpr_msb 0xca4                    ;  msbs: dst=2 src0=0 src1=1 src2=2
	v_wmma_scale_f32_16x16x128_f8f6f4 v[56:63] /*v[568:575]*/, v[160:175], v[0:15] /*v[256:271]*/, v[56:63] /*v[568:575]*/, v35, v16
	v_wmma_scale_f32_16x16x128_f8f6f4 v[48:55] /*v[560:567]*/, v[176:191], v[0:15] /*v[256:271]*/, v[48:55] /*v[560:567]*/, v35, v16 matrix_a_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47] /*v[552:559]*/, v[192:207], v[0:15] /*v[256:271]*/, v[40:47] /*v[552:559]*/, v32, v16
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39] /*v[544:551]*/, v[208:223], v[0:15] /*v[256:271]*/, v[32:39] /*v[544:551]*/, v32, v16 matrix_a_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31] /*v[536:543]*/, v[224:239], v[0:15] /*v[256:271]*/, v[24:31] /*v[536:543]*/, v33, v16
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23] /*v[528:535]*/, v[240:255], v[0:15] /*v[256:271]*/, v[16:23] /*v[528:535]*/, v33, v16 matrix_a_scale:MATRIX_SCALE_ROW1
	; sched_barrier mask(0x00000000)
	s_barrier_wait -1
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0xa4c0                   ;  msbs: dst=3 src0=0 src1=0 src2=0
	v_dual_mov_b32 v16 /*v784*/, v19 :: v_dual_mov_b32 v185 /*v953*/, v20
	v_dual_mov_b32 v186 /*v954*/, v21 :: v_dual_mov_b32 v187 /*v955*/, v22
	v_mov_b32_e32 v188 /*v956*/, v17
	s_set_vgpr_msb 0xc003                   ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_dual_mov_b32 v40, v248 /*v1016*/ :: v_dual_mov_b32 v38, v189 /*v957*/
	v_mov_b32_e32 v36, v17 /*v785*/
	s_add_co_i32 s66, s66, 1
	s_cmp_eq_u32 s66, s3
	s_set_vgpr_msb 0x300                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_scc1 .LBB0_22
.LBB0_6:                                ; =>This Inner Loop Header: Depth=1
	s_add_co_i32 s2, s66, 3
	s_mov_b32 s11, s10
	s_min_i32 s36, s2, s69
	s_and_b32 s6, s2, 3
	s_ashr_i32 s37, s36, 31
	s_mul_i32 s29, s6, 0xb600
	s_mul_u64 s[30:31], s[36:37], 0x6000
	s_lshl_b64 s[80:81], s[36:37], 7
	s_add_nc_u64 s[42:43], s[30:31], s[72:73]
	s_mov_b32 s76, s28
	s_or_b32 s31, s43, 0x80000000
	s_cmp_lt_i32 s2, s3
	s_mov_b32 s30, s42
	s_add_nc_u64 s[78:79], s[42:43], 0x80
	s_cselect_b32 s6, s64, 0x17fff
	s_and_b32 s2, s66, 3
	s_or_b32 s77, s29, 0x80
	s_bitset1_b32 s79, 31
	tensor_load_to_lds s[28:31], s[4:11]
	s_mov_b32 s45, s5
	s_mov_b32 s46, s6
	s_mov_b32 s48, s8
	s_mov_b32 s49, s9
	s_mov_b32 s50, s10
	s_mov_b32 s51, s10
	s_lshl_b64 s[30:31], s[36:37], 10
	s_mul_i32 s2, s2, 0xb600
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[76:79], s[44:51]
	s_add_nc_u64 s[50:51], s[30:31], s[74:75]
	s_add_co_i32 s11, s2, 0x6800
	v_add_nc_u32_e32 v17, s2, v44
	s_add_co_i32 s49, s29, 0x6800
	s_bitset1_b32 s51, 31
	s_mov_b32 s48, s28
	s_mov_b32 s52, s10
	s_mov_b32 s53, s5
	s_mov_b32 s54, s6
	s_mov_b32 s56, s8
	s_mov_b32 s58, s10
	s_mov_b32 s59, s10
	s_lshl_b64 s[30:31], s[36:37], 9
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[48:51], s[52:59]
	s_add_nc_u64 s[50:51], s[80:81], s[12:13]
	s_add_nc_u64 s[30:31], s[30:31], s[14:15]
	v_add_nc_u32_e32 v24, v17, v39
	s_add_co_i32 s49, s29, 0x6c00
	s_bitset1_b32 s51, 31
	s_mov_b32 s22, s6
	s_add_co_i32 s29, s29, 0xb400
	s_bitset1_b32 s31, 31
	s_mov_b32 s36, s10
	s_mov_b32 s37, s5
	s_mov_b32 s38, s6
	s_mov_b32 s40, s8
	s_mov_b32 s42, s10
	s_mov_b32 s43, s10
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[48:51], s[20:27]
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[28:31], s[36:43]
	v_add3_u32 v18, s2, v45, v42
	s_set_vgpr_msb 12                       ;  msbs: dst=0 src0=0 src1=3 src2=0
	v_add3_u32 v20, s2, v115 /*v883*/, v42
	s_set_vgpr_msb 0xc40                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[288:291]*/, v24 offset:3328
	ds_load_b128 v[36:39] /*v[292:295]*/, v24 offset:3344
	ds_load_b128 v[40:43] /*v[296:299]*/, v24 offset:3360
	ds_load_b128 v[44:47] /*v[300:303]*/, v24 offset:3376
	ds_load_b128 v[48:51] /*v[304:307]*/, v18 offset:128
	ds_load_b128 v[52:55] /*v[308:311]*/, v18 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_or_b32_e32 v18, s11, v46
	v_add_nc_u32_e32 v17, v17, v42
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[352:355]*/, v24 offset:9984
	ds_load_b128 v[100:103] /*v[356:359]*/, v24 offset:10000
	ds_load_b128 v[104:107] /*v[360:363]*/, v24 offset:10016
	ds_load_b128 v[108:111] /*v[364:367]*/, v24 offset:10032
	ds_load_b128 v[112:115] /*v[368:371]*/, v20 offset:128
	ds_load_b128 v[116:119] /*v[372:375]*/, v20 offset:144
	s_set_vgpr_msb 0x400c                   ;  msbs: dst=0 src0=0 src1=3 src2=0
	v_or_b32_e32 v20, s11, v116 /*v884*/
	v_add3_u32 v33, s11, v122 /*v890*/, v41
	v_add3_u32 v23, s2, v121 /*v889*/, v42
	s_add_co_i32 s6, s2, 0x6c00
	s_set_vgpr_msb 0xc40                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[200:203] /*v[456:459]*/, v24 offset:20000
	ds_load_b128 v[204:207] /*v[460:463]*/, v24 offset:20016
	ds_load_b128 v[208:211] /*v[464:467]*/, v23 offset:128
	ds_load_b128 v[212:215] /*v[468:471]*/, v23 offset:144
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v20, v20, v41
	ds_load_b32 v23, v33
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[224:227] /*v[480:483]*/, v24 offset:23296
	ds_load_b128 v[228:231] /*v[484:487]*/, v24 offset:23312
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_or_b32_e32 v33, s6, v39
	s_set_vgpr_msb 12                       ;  msbs: dst=0 src0=0 src1=3 src2=0
	v_add3_u32 v34, s2, v123 /*v891*/, v42
	v_or_b32_e32 v32, s2, v126 /*v894*/
	v_add3_u32 v35, s11, v124 /*v892*/, v41
	v_add3_u32 v19, v47, s2, v42
	v_add_nc_u32_e32 v33, v33, v125 /*v893*/
	v_add3_u32 v21, s2, v117 /*v885*/, v42
	v_add3_u32 v22, s2, v119 /*v887*/, v42
	s_set_vgpr_msb 0xc40                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[256:259]*/, v24
	ds_load_b128 v[4:7] /*v[260:263]*/, v24 offset:16
	ds_load_b128 v[8:11] /*v[264:267]*/, v24 offset:32
	ds_load_b128 v[12:15] /*v[268:271]*/, v24 offset:48
	ds_load_b128 v[16:19] /*v[272:275]*/, v17 offset:128
	ds_load_b128 v[20:23] /*v[276:279]*/, v17 offset:144
	ds_load_b128 v[64:67] /*v[320:323]*/, v24 offset:6656
	ds_load_b128 v[68:71] /*v[324:327]*/, v24 offset:6672
	ds_load_b128 v[72:75] /*v[328:331]*/, v24 offset:6688
	ds_load_b128 v[76:79] /*v[332:335]*/, v24 offset:6704
	ds_load_b128 v[80:83] /*v[336:339]*/, v19 offset:128
	ds_load_b128 v[84:87] /*v[340:343]*/, v19 offset:144
	ds_load_b128 v[128:131] /*v[384:387]*/, v24 offset:13312
	ds_load_b128 v[132:135] /*v[388:391]*/, v24 offset:13328
	ds_load_b128 v[136:139] /*v[392:395]*/, v24 offset:13344
	ds_load_b128 v[140:143] /*v[396:399]*/, v24 offset:13360
	ds_load_b128 v[144:147] /*v[400:403]*/, v21 offset:128
	ds_load_b128 v[148:151] /*v[404:407]*/, v21 offset:144
	ds_load_b128 v[160:163] /*v[416:419]*/, v24 offset:16640
	ds_load_b128 v[164:167] /*v[420:423]*/, v24 offset:16656
	ds_load_b128 v[168:171] /*v[424:427]*/, v24 offset:16672
	ds_load_b128 v[172:175] /*v[428:431]*/, v24 offset:16688
	ds_load_b128 v[176:179] /*v[432:435]*/, v22 offset:128
	ds_load_b128 v[180:183] /*v[436:439]*/, v22 offset:144
	ds_load_b128 v[192:195] /*v[448:451]*/, v24 offset:19968
	ds_load_b128 v[196:199] /*v[452:455]*/, v24 offset:19984
	ds_load_b128 v[232:235] /*v[488:491]*/, v24 offset:23328
	ds_load_b128 v[236:239] /*v[492:495]*/, v24 offset:23344
	ds_load_b128 v[240:243] /*v[496:499]*/, v34 offset:128
	ds_load_b128 v[244:247] /*v[500:503]*/, v34 offset:144
	s_set_vgpr_msb 0x400c                   ;  msbs: dst=0 src0=0 src1=3 src2=0
	ds_load_b32 v24, v35
	v_add3_u32 v34, s6, v125 /*v893*/, v39
	ds_load_b128 v[128:131], v33
	ds_load_b128 v[132:135], v33 offset:16
	ds_load_b128 v[136:139], v33 offset:32
	ds_load_b128 v[140:143], v33 offset:48
	ds_load_b128 v[144:147], v34 offset:2304
	ds_load_b128 v[148:151], v34 offset:2320
	ds_load_b128 v[152:155], v34 offset:2336
	ds_load_b128 v[156:159], v34 offset:2352
	ds_load_b128 v[160:163], v34 offset:4608
	ds_load_b128 v[164:167], v34 offset:4624
	ds_load_b128 v[168:171], v34 offset:4640
	ds_load_b128 v[172:175], v34 offset:4656
	ds_load_b128 v[176:179], v34 offset:6912
	ds_load_b128 v[180:183], v34 offset:6928
	ds_load_b128 v[184:187], v34 offset:6944
	ds_load_b128 v[188:191], v34 offset:6960
	ds_load_b128 v[192:195], v34 offset:9216
	ds_load_b128 v[196:199], v34 offset:9232
	ds_load_b128 v[200:203], v34 offset:9248
	ds_load_b128 v[204:207], v34 offset:9264
	ds_load_b128 v[208:211], v34 offset:11520
	ds_load_b128 v[212:215], v34 offset:11536
	ds_load_b128 v[216:219], v34 offset:11552
	ds_load_b128 v[220:223], v34 offset:11568
	ds_load_b128 v[224:227], v34 offset:13824
	ds_load_b128 v[228:231], v34 offset:13840
	ds_load_b128 v[232:235], v34 offset:13856
	ds_load_b128 v[236:239], v34 offset:13872
	s_set_vgpr_msb 0xc00                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v18, v18, v41 :: v_dual_bitop2_b32 v17, s11, v43 bitop3:0x54
	s_set_vgpr_msb 12                       ;  msbs: dst=0 src0=0 src1=3 src2=0
	v_or_b32_e32 v19, s11, v114 /*v882*/
	v_add3_u32 v21, s11, v118 /*v886*/, v41
	v_add3_u32 v22, s11, v120 /*v888*/, v41
	s_mov_b32 s2, 0xb400
	s_set_vgpr_msb 0xc00                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_add_nc_u32 v17, v17, v41 :: v_dual_add_nc_u32 v32, s2, v32
	v_add_nc_u32_e32 v19, v19, v41
	ds_load_b128 v[240:243], v34 offset:16128
	ds_load_b128 v[244:247], v34 offset:16144
	ds_load_b128 v[248:251], v34 offset:16160
	ds_load_b128 v[252:255], v34 offset:16176
	ds_load_2addr_b32 v[34:35], v32 offset1:32
	ds_load_2addr_b32 v[32:33], v32 offset0:64 offset1:96
	ds_load_b32 v18, v18
	ds_load_b32 v17, v17
	ds_load_b32 v20, v20
	ds_load_b32 v19, v19
	ds_load_b32 v21, v21
	ds_load_b32 v22, v22
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_mov_b32 v24 /*v280*/, s10 :: v_dual_mov_b32 v25 /*v281*/, s10
	v_dual_mov_b32 v26 /*v282*/, s10 :: v_dual_mov_b32 v27 /*v283*/, s10
	v_dual_mov_b32 v28 /*v284*/, s10 :: v_dual_mov_b32 v29 /*v285*/, s10
	v_dual_mov_b32 v30 /*v286*/, s10 :: v_dual_mov_b32 v31 /*v287*/, s10
	v_dual_mov_b32 v56 /*v312*/, s10 :: v_dual_mov_b32 v57 /*v313*/, s10
	v_dual_mov_b32 v58 /*v314*/, s10 :: v_dual_mov_b32 v59 /*v315*/, s10
	v_dual_mov_b32 v60 /*v316*/, s10 :: v_dual_mov_b32 v61 /*v317*/, s10
	v_dual_mov_b32 v62 /*v318*/, s10 :: v_dual_mov_b32 v63 /*v319*/, s10
	v_dual_mov_b32 v88 /*v344*/, s10 :: v_dual_mov_b32 v89 /*v345*/, s10
	v_dual_mov_b32 v90 /*v346*/, s10 :: v_dual_mov_b32 v91 /*v347*/, s10
	v_dual_mov_b32 v92 /*v348*/, s10 :: v_dual_mov_b32 v93 /*v349*/, s10
	v_dual_mov_b32 v94 /*v350*/, s10 :: v_dual_mov_b32 v95 /*v351*/, s10
	v_dual_mov_b32 v120 /*v376*/, s10 :: v_dual_mov_b32 v121 /*v377*/, s10
	v_dual_mov_b32 v122 /*v378*/, s10 :: v_dual_mov_b32 v123 /*v379*/, s10
	v_dual_mov_b32 v124 /*v380*/, s10 :: v_dual_mov_b32 v125 /*v381*/, s10
	v_dual_mov_b32 v126 /*v382*/, s10 :: v_dual_mov_b32 v127 /*v383*/, s10
	v_dual_mov_b32 v152 /*v408*/, s10 :: v_dual_mov_b32 v153 /*v409*/, s10
	v_dual_mov_b32 v154 /*v410*/, s10 :: v_dual_mov_b32 v155 /*v411*/, s10
	v_dual_mov_b32 v156 /*v412*/, s10 :: v_dual_mov_b32 v157 /*v413*/, s10
	v_dual_mov_b32 v158 /*v414*/, s10 :: v_dual_mov_b32 v159 /*v415*/, s10
	v_dual_mov_b32 v184 /*v440*/, s10 :: v_dual_mov_b32 v185 /*v441*/, s10
	v_dual_mov_b32 v186 /*v442*/, s10 :: v_dual_mov_b32 v187 /*v443*/, s10
	v_dual_mov_b32 v188 /*v444*/, s10 :: v_dual_mov_b32 v189 /*v445*/, s10
	v_dual_mov_b32 v190 /*v446*/, s10 :: v_dual_mov_b32 v191 /*v447*/, s10
	v_dual_mov_b32 v216 /*v472*/, s10 :: v_dual_mov_b32 v217 /*v473*/, s10
	v_dual_mov_b32 v218 /*v474*/, s10 :: v_dual_mov_b32 v219 /*v475*/, s10
	v_dual_mov_b32 v220 /*v476*/, s10 :: v_dual_mov_b32 v221 /*v477*/, s10
	v_dual_mov_b32 v222 /*v478*/, s10 :: v_dual_mov_b32 v223 /*v479*/, s10
	v_dual_mov_b32 v248 /*v504*/, s10 :: v_dual_mov_b32 v249 /*v505*/, s10
	v_dual_mov_b32 v250 /*v506*/, s10 :: v_dual_mov_b32 v251 /*v507*/, s10
	v_dual_mov_b32 v252 /*v508*/, s10 :: v_dual_mov_b32 v253 /*v509*/, s10
	v_dual_mov_b32 v254 /*v510*/, s10 :: v_dual_mov_b32 v255 /*v511*/, s10
	s_wait_tensorcnt 0xa
	s_lshl_b32 s2, s66, 7
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x0
	s_barrier_signal -1
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x40cd                   ;  msbs: dst=3 src0=1 src1=3 src2=0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[50:57] /*v[818:825]*/, v[0:15] /*v[256:271]*/, v[82:97] /*v[850:865]*/, 0, v17, v28
	s_or_b32 s6, s2, 0x7f
	s_cmp_le_i32 s6, s33
	s_set_vgpr_msb 0xcdc0                   ;  msbs: dst=3 src0=0 src1=0 src2=0
	v_or_b32_e32 v247 /*v1015*/, s2, v27
	s_set_vgpr_msb 0xc0cd                   ;  msbs: dst=3 src0=1 src1=3 src2=0
	v_or_b32_e32 v246 /*v1014*/, s2, v127 /*v895*/
	v_wmma_scale_f32_16x16x128_f8f6f4 v[58:65] /*v[826:833]*/, v[32:47] /*v[288:303]*/, v[82:97] /*v[850:865]*/, 0, v18, v28
	v_or_b32_e32 v245 /*v1013*/, s2, v128 /*v896*/
	v_or_b32_e32 v244 /*v1012*/, s2, v129 /*v897*/
	v_or_b32_e32 v243 /*v1011*/, s2, v130 /*v898*/
	v_wmma_scale_f32_16x16x128_f8f6f4 v[34:41] /*v[802:809]*/, v[64:79] /*v[320:335]*/, v[82:97] /*v[850:865]*/, 0, v19, v28
	v_or_b32_e32 v241 /*v1009*/, s2, v131 /*v899*/
	v_or_b32_e32 v242 /*v1010*/, s2, v132 /*v900*/
	v_or_b32_e32 v240 /*v1008*/, s2, v133 /*v901*/
	s_set_vgpr_msb 0xcd8d                   ;  msbs: dst=2 src0=1 src1=3 src2=0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15] /*v[520:527]*/, v[96:111] /*v[352:367]*/, v[82:97] /*v[850:865]*/, 0, v20, v28
	s_set_vgpr_msb 0x8dcd                   ;  msbs: dst=3 src0=1 src1=3 src2=0
	v_or_b32_e32 v239 /*v1007*/, s2, v134 /*v902*/
	v_or_b32_e32 v238 /*v1006*/, s2, v135 /*v903*/
	v_or_b32_e32 v237 /*v1005*/, s2, v136 /*v904*/
	v_wmma_scale_f32_16x16x128_f8f6f4 v[42:49] /*v[810:817]*/, v[128:143] /*v[384:399]*/, v[82:97] /*v[850:865]*/, 0, v21, v28
	v_or_b32_e32 v236 /*v1004*/, s2, v137 /*v905*/
	v_or_b32_e32 v234 /*v1002*/, s2, v138 /*v906*/
	v_or_b32_e32 v235 /*v1003*/, s2, v139 /*v907*/
	v_wmma_scale_f32_16x16x128_f8f6f4 v[26:33] /*v[794:801]*/, v[160:175] /*v[416:431]*/, v[82:97] /*v[850:865]*/, 0, v22, v28
	v_or_b32_e32 v233 /*v1001*/, s2, v140 /*v908*/
	v_or_b32_e32 v232 /*v1000*/, s2, v141 /*v909*/
	v_or_b32_e32 v231 /*v999*/, s2, v142 /*v910*/
	v_wmma_scale_f32_16x16x128_f8f6f4 v[18:25] /*v[786:793]*/, v[192:207] /*v[448:463]*/, v[82:97] /*v[850:865]*/, 0, v23, v28
	v_or_b32_e32 v230 /*v998*/, s2, v143 /*v911*/
	v_or_b32_e32 v229 /*v997*/, s2, v144 /*v912*/
	v_or_b32_e32 v227 /*v995*/, s2, v145 /*v913*/
	s_set_vgpr_msb 0xcd8d                   ;  msbs: dst=2 src0=1 src1=3 src2=0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7] /*v[512:519]*/, v[224:239] /*v[480:495]*/, v[82:97] /*v[850:865]*/, 0, v24, v28
	s_set_vgpr_msb 0x8dcc                   ;  msbs: dst=3 src0=0 src1=3 src2=0
	v_or_b32_e32 v228 /*v996*/, s2, v146 /*v914*/
	v_or_b32_e32 v226 /*v994*/, s2, v147 /*v915*/
	v_or_b32_e32 v225 /*v993*/, s2, v148 /*v916*/
	s_set_vgpr_msb 0xccf1                   ;  msbs: dst=3 src0=1 src1=0 src2=3
	v_wmma_scale_f32_16x16x128_f8f6f4 v[50:57] /*v[818:825]*/, v[16:31] /*v[272:287]*/, v[0:15], v[50:57] /*v[818:825]*/, v17, v28 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xf1cc                   ;  msbs: dst=3 src0=0 src1=3 src2=0
	v_or_b32_e32 v224 /*v992*/, s2, v149 /*v917*/
	v_or_b32_e32 v223 /*v991*/, s2, v150 /*v918*/
	v_or_b32_e32 v222 /*v990*/, s2, v151 /*v919*/
	s_set_vgpr_msb 0xccf1                   ;  msbs: dst=3 src0=1 src1=0 src2=3
	v_wmma_scale_f32_16x16x128_f8f6f4 v[58:65] /*v[826:833]*/, v[48:63] /*v[304:319]*/, v[0:15], v[58:65] /*v[826:833]*/, v18, v28 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xf1cc                   ;  msbs: dst=3 src0=0 src1=3 src2=0
	v_or_b32_e32 v220 /*v988*/, s2, v152 /*v920*/
	v_or_b32_e32 v221 /*v989*/, s2, v153 /*v921*/
	v_or_b32_e32 v219 /*v987*/, s2, v154 /*v922*/
	s_set_vgpr_msb 0xccf1                   ;  msbs: dst=3 src0=1 src1=0 src2=3
	v_wmma_scale_f32_16x16x128_f8f6f4 v[34:41] /*v[802:809]*/, v[80:95] /*v[336:351]*/, v[0:15], v[34:41] /*v[802:809]*/, v19, v28 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xf1cc                   ;  msbs: dst=3 src0=0 src1=3 src2=0
	v_or_b32_e32 v218 /*v986*/, s2, v155 /*v923*/
	v_or_b32_e32 v217 /*v985*/, s2, v156 /*v924*/
	v_or_b32_e32 v216 /*v984*/, s2, v157 /*v925*/
	s_set_vgpr_msb 0xcca1                   ;  msbs: dst=2 src0=1 src1=0 src2=2
	v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15] /*v[520:527]*/, v[112:127] /*v[368:383]*/, v[0:15], v[8:15] /*v[520:527]*/, v20, v28 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa1cc                   ;  msbs: dst=3 src0=0 src1=3 src2=0
	v_or_b32_e32 v215 /*v983*/, s2, v158 /*v926*/
	v_or_b32_e32 v213 /*v981*/, s2, v159 /*v927*/
	v_or_b32_e32 v214 /*v982*/, s2, v160 /*v928*/
	s_set_vgpr_msb 0xccf1                   ;  msbs: dst=3 src0=1 src1=0 src2=3
	v_wmma_scale_f32_16x16x128_f8f6f4 v[42:49] /*v[810:817]*/, v[144:159] /*v[400:415]*/, v[0:15], v[42:49] /*v[810:817]*/, v21, v28 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xf1cc                   ;  msbs: dst=3 src0=0 src1=3 src2=0
	v_or_b32_e32 v212 /*v980*/, s2, v161 /*v929*/
	v_or_b32_e32 v211 /*v979*/, s2, v162 /*v930*/
	v_or_b32_e32 v210 /*v978*/, s2, v163 /*v931*/
	s_set_vgpr_msb 0xccf1                   ;  msbs: dst=3 src0=1 src1=0 src2=3
	v_wmma_scale_f32_16x16x128_f8f6f4 v[26:33] /*v[794:801]*/, v[176:191] /*v[432:447]*/, v[0:15], v[26:33] /*v[794:801]*/, v22, v28 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xf1cc                   ;  msbs: dst=3 src0=0 src1=3 src2=0
	v_or_b32_e32 v209 /*v977*/, s2, v164 /*v932*/
	v_or_b32_e32 v208 /*v976*/, s2, v165 /*v933*/
	v_or_b32_e32 v206 /*v974*/, s2, v166 /*v934*/
	s_set_vgpr_msb 0xccf1                   ;  msbs: dst=3 src0=1 src1=0 src2=3
	v_wmma_scale_f32_16x16x128_f8f6f4 v[18:25] /*v[786:793]*/, v[208:223] /*v[464:479]*/, v[0:15], v[18:25] /*v[786:793]*/, v23, v28 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xf1cc                   ;  msbs: dst=3 src0=0 src1=3 src2=0
	v_or_b32_e32 v207 /*v975*/, s2, v167 /*v935*/
	v_or_b32_e32 v205 /*v973*/, s2, v168 /*v936*/
	v_or_b32_e32 v204 /*v972*/, s2, v169 /*v937*/
	s_set_vgpr_msb 0xcca1                   ;  msbs: dst=2 src0=1 src1=0 src2=2
	v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7] /*v[512:519]*/, v[240:255] /*v[496:511]*/, v[0:15], v[0:7] /*v[512:519]*/, v24, v28 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa1cc                   ;  msbs: dst=3 src0=0 src1=3 src2=0
	v_or_b32_e32 v203 /*v971*/, s2, v170 /*v938*/
	v_or_b32_e32 v202 /*v970*/, s2, v171 /*v939*/
	v_or_b32_e32 v201 /*v969*/, s2, v172 /*v940*/
	v_or_b32_e32 v199 /*v967*/, s2, v173 /*v941*/
	v_or_b32_e32 v200 /*v968*/, s2, v174 /*v942*/
	v_or_b32_e32 v198 /*v966*/, s2, v175 /*v943*/
	v_or_b32_e32 v197 /*v965*/, s2, v176 /*v944*/
	v_or_b32_e32 v196 /*v964*/, s2, v177 /*v945*/
	v_or_b32_e32 v195 /*v963*/, s2, v178 /*v946*/
	v_or_b32_e32 v194 /*v962*/, s2, v179 /*v947*/
	v_or_b32_e32 v193 /*v961*/, s2, v180 /*v948*/
	v_or_b32_e32 v192 /*v960*/, s2, v181 /*v949*/
	s_set_vgpr_msb 0xcc00                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_scc1 .LBB0_8
; %bb.7:                                ; %_ZZZZ12k_mla_fwd_mxENKUlT_E4_clISt17integral_constantIiLi0EEEEDaS_ENKUlS_E0_clIS3_EEDaS_ENKUlS_E_clIS3_EEDaS_.exit.i.i.i.i.i.i.i.i
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_set_vgpr_msb 3                        ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_cmp_le_i32_e32 vcc_lo, v247 /*v1015*/, v26
	s_set_vgpr_msb 0x3cc                    ;  msbs: dst=3 src0=0 src1=3 src2=0
	v_cndmask_b32_e32 v50 /*v818*/, 0xff800000, v50 /*v818*/, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v26, v247 /*v1015*/
	v_cndmask_b32_e32 v51 /*v819*/, 0xff800000, v51 /*v819*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v246 /*v1014*/
	v_cndmask_b32_e32 v52 /*v820*/, 0xff800000, v52 /*v820*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v245 /*v1013*/
	v_cndmask_b32_e32 v53 /*v821*/, 0xff800000, v53 /*v821*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v244 /*v1012*/
	v_cndmask_b32_e32 v54 /*v822*/, 0xff800000, v54 /*v822*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v243 /*v1011*/
	v_cndmask_b32_e32 v55 /*v823*/, 0xff800000, v55 /*v823*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v241 /*v1009*/
	v_cndmask_b32_e32 v56 /*v824*/, 0xff800000, v56 /*v824*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v242 /*v1010*/
	v_cndmask_b32_e32 v57 /*v825*/, 0xff800000, v57 /*v825*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v240 /*v1008*/
	v_cndmask_b32_e32 v58 /*v826*/, 0xff800000, v58 /*v826*/, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v26, v240 /*v1008*/
	v_cndmask_b32_e32 v59 /*v827*/, 0xff800000, v59 /*v827*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v239 /*v1007*/
	v_cndmask_b32_e32 v60 /*v828*/, 0xff800000, v60 /*v828*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v238 /*v1006*/
	v_cndmask_b32_e32 v61 /*v829*/, 0xff800000, v61 /*v829*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v237 /*v1005*/
	v_cndmask_b32_e32 v62 /*v830*/, 0xff800000, v62 /*v830*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v236 /*v1004*/
	v_cndmask_b32_e32 v63 /*v831*/, 0xff800000, v63 /*v831*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v234 /*v1002*/
	v_cndmask_b32_e32 v64 /*v832*/, 0xff800000, v64 /*v832*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v235 /*v1003*/
	v_cndmask_b32_e32 v65 /*v833*/, 0xff800000, v65 /*v833*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v233 /*v1001*/
	v_cndmask_b32_e32 v34 /*v802*/, 0xff800000, v34 /*v802*/, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v26, v233 /*v1001*/
	v_cndmask_b32_e32 v35 /*v803*/, 0xff800000, v35 /*v803*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v232 /*v1000*/
	v_cndmask_b32_e32 v36 /*v804*/, 0xff800000, v36 /*v804*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v231 /*v999*/
	v_cndmask_b32_e32 v37 /*v805*/, 0xff800000, v37 /*v805*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v230 /*v998*/
	v_cndmask_b32_e32 v38 /*v806*/, 0xff800000, v38 /*v806*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v229 /*v997*/
	v_cndmask_b32_e32 v39 /*v807*/, 0xff800000, v39 /*v807*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v227 /*v995*/
	v_cndmask_b32_e32 v40 /*v808*/, 0xff800000, v40 /*v808*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v228 /*v996*/
	v_cndmask_b32_e32 v41 /*v809*/, 0xff800000, v41 /*v809*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v226 /*v994*/
	s_set_vgpr_msb 0xcc88                   ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v8 /*v520*/, 0xff800000, v8 /*v520*/, vcc_lo
	s_set_vgpr_msb 0x8803                   ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_cmp_lt_i32_e32 vcc_lo, v226 /*v994*/, v26
	s_set_vgpr_msb 0x388                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v9 /*v521*/, 0xff800000, v9 /*v521*/, vcc_lo
	s_set_vgpr_msb 0x8803                   ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_cmp_le_i32_e32 vcc_lo, v225 /*v993*/, v26
	s_set_vgpr_msb 0x388                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v10 /*v522*/, 0xff800000, v10 /*v522*/, vcc_lo
	s_set_vgpr_msb 0x8803                   ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_cmp_le_i32_e32 vcc_lo, v224 /*v992*/, v26
	s_set_vgpr_msb 0x388                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v11 /*v523*/, 0xff800000, v11 /*v523*/, vcc_lo
	s_set_vgpr_msb 0x8803                   ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_cmp_le_i32_e32 vcc_lo, v223 /*v991*/, v26
	s_set_vgpr_msb 0x388                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v12 /*v524*/, 0xff800000, v12 /*v524*/, vcc_lo
	s_set_vgpr_msb 0x8803                   ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_cmp_le_i32_e32 vcc_lo, v222 /*v990*/, v26
	s_set_vgpr_msb 0x388                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v13 /*v525*/, 0xff800000, v13 /*v525*/, vcc_lo
	s_set_vgpr_msb 0x8803                   ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_cmp_le_i32_e32 vcc_lo, v220 /*v988*/, v26
	s_set_vgpr_msb 0x388                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v14 /*v526*/, 0xff800000, v14 /*v526*/, vcc_lo
	s_set_vgpr_msb 0x8803                   ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_cmp_le_i32_e32 vcc_lo, v221 /*v989*/, v26
	s_set_vgpr_msb 0x388                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v15 /*v527*/, 0xff800000, v15 /*v527*/, vcc_lo
	s_set_vgpr_msb 0x8803                   ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_cmp_le_i32_e32 vcc_lo, v219 /*v987*/, v26
	s_set_vgpr_msb 0x3cc                    ;  msbs: dst=3 src0=0 src1=3 src2=0
	v_cndmask_b32_e32 v42 /*v810*/, 0xff800000, v42 /*v810*/, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v26, v219 /*v987*/
	v_cndmask_b32_e32 v43 /*v811*/, 0xff800000, v43 /*v811*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v218 /*v986*/
	v_cndmask_b32_e32 v44 /*v812*/, 0xff800000, v44 /*v812*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v217 /*v985*/
	v_cndmask_b32_e32 v45 /*v813*/, 0xff800000, v45 /*v813*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v216 /*v984*/
	v_cndmask_b32_e32 v46 /*v814*/, 0xff800000, v46 /*v814*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v215 /*v983*/
	v_cndmask_b32_e32 v47 /*v815*/, 0xff800000, v47 /*v815*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v213 /*v981*/
	v_cndmask_b32_e32 v48 /*v816*/, 0xff800000, v48 /*v816*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v214 /*v982*/
	v_cndmask_b32_e32 v49 /*v817*/, 0xff800000, v49 /*v817*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v212 /*v980*/
	v_cndmask_b32_e32 v26 /*v794*/, 0xff800000, v26 /*v794*/, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v26, v212 /*v980*/
	v_cndmask_b32_e32 v27 /*v795*/, 0xff800000, v27 /*v795*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v211 /*v979*/
	v_cndmask_b32_e32 v28 /*v796*/, 0xff800000, v28 /*v796*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v210 /*v978*/
	v_cndmask_b32_e32 v29 /*v797*/, 0xff800000, v29 /*v797*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v209 /*v977*/
	v_cndmask_b32_e32 v30 /*v798*/, 0xff800000, v30 /*v798*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v208 /*v976*/
	v_cndmask_b32_e32 v31 /*v799*/, 0xff800000, v31 /*v799*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v206 /*v974*/
	v_cndmask_b32_e32 v32 /*v800*/, 0xff800000, v32 /*v800*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v207 /*v975*/
	v_cndmask_b32_e32 v33 /*v801*/, 0xff800000, v33 /*v801*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v205 /*v973*/
	v_cndmask_b32_e32 v18 /*v786*/, 0xff800000, v18 /*v786*/, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v26, v205 /*v973*/
	v_cndmask_b32_e32 v19 /*v787*/, 0xff800000, v19 /*v787*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v204 /*v972*/
	v_cndmask_b32_e32 v20 /*v788*/, 0xff800000, v20 /*v788*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v203 /*v971*/
	v_cndmask_b32_e32 v21 /*v789*/, 0xff800000, v21 /*v789*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v202 /*v970*/
	v_cndmask_b32_e32 v22 /*v790*/, 0xff800000, v22 /*v790*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v201 /*v969*/
	v_cndmask_b32_e32 v23 /*v791*/, 0xff800000, v23 /*v791*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v199 /*v967*/
	v_cndmask_b32_e32 v24 /*v792*/, 0xff800000, v24 /*v792*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v200 /*v968*/
	v_cndmask_b32_e32 v25 /*v793*/, 0xff800000, v25 /*v793*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v26, v198 /*v966*/
	s_set_vgpr_msb 0xcc88                   ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v0 /*v512*/, 0xff800000, v0 /*v512*/, vcc_lo
	s_set_vgpr_msb 0x8803                   ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_cmp_lt_i32_e32 vcc_lo, v198 /*v966*/, v26
	s_set_vgpr_msb 0x388                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v1 /*v513*/, 0xff800000, v1 /*v513*/, vcc_lo
	s_set_vgpr_msb 0x8803                   ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_cmp_le_i32_e32 vcc_lo, v197 /*v965*/, v26
	s_set_vgpr_msb 0x388                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v2 /*v514*/, 0xff800000, v2 /*v514*/, vcc_lo
	s_set_vgpr_msb 0x8803                   ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_cmp_le_i32_e32 vcc_lo, v196 /*v964*/, v26
	s_set_vgpr_msb 0x388                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v3 /*v515*/, 0xff800000, v3 /*v515*/, vcc_lo
	s_set_vgpr_msb 0x8803                   ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_cmp_le_i32_e32 vcc_lo, v195 /*v963*/, v26
	s_set_vgpr_msb 0x388                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v4 /*v516*/, 0xff800000, v4 /*v516*/, vcc_lo
	s_set_vgpr_msb 0x8803                   ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_cmp_le_i32_e32 vcc_lo, v194 /*v962*/, v26
	s_set_vgpr_msb 0x388                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v5 /*v517*/, 0xff800000, v5 /*v517*/, vcc_lo
	s_set_vgpr_msb 0x8803                   ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_cmp_le_i32_e32 vcc_lo, v193 /*v961*/, v26
	s_set_vgpr_msb 0x388                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v6 /*v518*/, 0xff800000, v6 /*v518*/, vcc_lo
	s_set_vgpr_msb 0x8803                   ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_cmp_le_i32_e32 vcc_lo, v192 /*v960*/, v26
	s_set_vgpr_msb 0x388                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v7 /*v519*/, 0xff800000, v7 /*v519*/, vcc_lo
	s_set_vgpr_msb 0x8800                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_8:                                ;   in Loop: Header=BB0_6 Depth=1
	s_set_vgpr_msb 0xff                     ;  msbs: dst=3 src0=3 src1=3 src2=3
	v_max3_num_f32 v17 /*v785*/, v50 /*v818*/, 0xff800000, v51 /*v819*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v52 /*v820*/, v53 /*v821*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v54 /*v822*/, v55 /*v823*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v56 /*v824*/, v57 /*v825*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v58 /*v826*/, v59 /*v827*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v60 /*v828*/, v61 /*v829*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v62 /*v830*/, v63 /*v831*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v64 /*v832*/, v65 /*v833*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v34 /*v802*/, v35 /*v803*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v36 /*v804*/, v37 /*v805*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v38 /*v806*/, v39 /*v807*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v40 /*v808*/, v41 /*v809*/
	s_set_vgpr_msb 0xffeb                   ;  msbs: dst=3 src0=3 src1=2 src2=2
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v8 /*v520*/, v9 /*v521*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v10 /*v522*/, v11 /*v523*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v12 /*v524*/, v13 /*v525*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v14 /*v526*/, v15 /*v527*/
	s_set_vgpr_msb 0xebff                   ;  msbs: dst=3 src0=3 src1=3 src2=3
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v42 /*v810*/, v43 /*v811*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v44 /*v812*/, v45 /*v813*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v46 /*v814*/, v47 /*v815*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v48 /*v816*/, v49 /*v817*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v26 /*v794*/, v27 /*v795*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v28 /*v796*/, v29 /*v797*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v30 /*v798*/, v31 /*v799*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v32 /*v800*/, v33 /*v801*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v18 /*v786*/, v19 /*v787*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v20 /*v788*/, v21 /*v789*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v22 /*v790*/, v23 /*v791*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v24 /*v792*/, v25 /*v793*/
	s_set_vgpr_msb 0xffeb                   ;  msbs: dst=3 src0=3 src1=2 src2=2
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v0 /*v512*/, v1 /*v513*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v2 /*v514*/, v3 /*v515*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v4 /*v516*/, v5 /*v517*/
	v_max3_num_f32 v17 /*v785*/, v17 /*v785*/, v6 /*v518*/, v7 /*v519*/
	v_mov_b32_e32 v66 /*v834*/, v17 /*v785*/
	v_permlanex16_b32 v66 /*v834*/, v66 /*v834*/, s65, 0xfedcba98
	s_set_vgpr_msb 0xebcf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_max_num_f32_e32 v17 /*v785*/, v17 /*v785*/, v66 /*v834*/
	v_fma_f32 v66 /*v834*/, v17 /*v785*/, s60, -v36
	v_mul_f32_e32 v17 /*v785*/, s60, v17 /*v785*/
	v_cmp_lt_f32_e32 vcc_lo, 0x3fb8aa3b, v66 /*v834*/
	s_set_vgpr_msb 0xcfcc                   ;  msbs: dst=3 src0=0 src1=3 src2=0
	v_max_num_f32_e32 v17 /*v785*/, v36, v17 /*v785*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0xccf3                   ;  msbs: dst=3 src0=3 src1=0 src2=3
	v_cndmask_b32_e64 v17 /*v785*/, v17 /*v785*/, v36, s2
	v_fma_f32 v50 /*v818*/, v50 /*v818*/, s60, -v17 /*v785*/
	v_fma_f32 v51 /*v819*/, v51 /*v819*/, s60, -v17 /*v785*/
	v_fma_f32 v52 /*v820*/, v52 /*v820*/, s60, -v17 /*v785*/
	v_fma_f32 v53 /*v821*/, v53 /*v821*/, s60, -v17 /*v785*/
	v_fma_f32 v54 /*v822*/, v54 /*v822*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v50 /*v818*/, v50 /*v818*/
	v_fma_f32 v55 /*v823*/, v55 /*v823*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v51 /*v819*/, v51 /*v819*/
	v_fma_f32 v56 /*v824*/, v56 /*v824*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v52 /*v820*/, v52 /*v820*/
	v_fma_f32 v57 /*v825*/, v57 /*v825*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v53 /*v821*/, v53 /*v821*/
	v_fma_f32 v58 /*v826*/, v58 /*v826*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v54 /*v822*/, v54 /*v822*/
	v_fma_f32 v59 /*v827*/, v59 /*v827*/, s60, -v17 /*v785*/
	s_set_vgpr_msb 0xf3ff                   ;  msbs: dst=3 src0=3 src1=3 src2=3
	v_add_f32_e32 v66 /*v834*/, v50 /*v818*/, v51 /*v819*/
	v_exp_f32_e32 v55 /*v823*/, v55 /*v823*/
	v_fma_f32 v60 /*v828*/, v60 /*v828*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v56 /*v824*/, v56 /*v824*/
	v_fma_f32 v61 /*v829*/, v61 /*v829*/, s60, -v17 /*v785*/
	v_add_f32_e32 v66 /*v834*/, v52 /*v820*/, v66 /*v834*/
	v_exp_f32_e32 v57 /*v825*/, v57 /*v825*/
	v_fma_f32 v62 /*v830*/, v62 /*v830*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v58 /*v826*/, v58 /*v826*/
	v_fma_f32 v63 /*v831*/, v63 /*v831*/, s60, -v17 /*v785*/
	v_add_f32_e32 v66 /*v834*/, v53 /*v821*/, v66 /*v834*/
	v_exp_f32_e32 v59 /*v827*/, v59 /*v827*/
	v_fma_f32 v64 /*v832*/, v64 /*v832*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v60 /*v828*/, v60 /*v828*/
	v_fma_f32 v65 /*v833*/, v65 /*v833*/, s60, -v17 /*v785*/
	v_add_f32_e32 v66 /*v834*/, v54 /*v822*/, v66 /*v834*/
	v_exp_f32_e32 v61 /*v829*/, v61 /*v829*/
	v_fma_f32 v34 /*v802*/, v34 /*v802*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v62 /*v830*/, v62 /*v830*/
	v_fma_f32 v35 /*v803*/, v35 /*v803*/, s60, -v17 /*v785*/
	v_add_f32_e32 v66 /*v834*/, v55 /*v823*/, v66 /*v834*/
	v_exp_f32_e32 v63 /*v831*/, v63 /*v831*/
	v_fma_f32 v36 /*v804*/, v36 /*v804*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v64 /*v832*/, v64 /*v832*/
	v_fma_f32 v37 /*v805*/, v37 /*v805*/, s60, -v17 /*v785*/
	v_add_f32_e32 v66 /*v834*/, v56 /*v824*/, v66 /*v834*/
	v_exp_f32_e32 v65 /*v833*/, v65 /*v833*/
	v_fma_f32 v38 /*v806*/, v38 /*v806*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v34 /*v802*/, v34 /*v802*/
	v_fma_f32 v39 /*v807*/, v39 /*v807*/, s60, -v17 /*v785*/
	v_add_f32_e32 v66 /*v834*/, v57 /*v825*/, v66 /*v834*/
	v_exp_f32_e32 v35 /*v803*/, v35 /*v803*/
	v_fma_f32 v40 /*v808*/, v40 /*v808*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v36 /*v804*/, v36 /*v804*/
	v_fma_f32 v41 /*v809*/, v41 /*v809*/, s60, -v17 /*v785*/
	v_add_f32_e32 v66 /*v834*/, v58 /*v826*/, v66 /*v834*/
	v_exp_f32_e32 v37 /*v805*/, v37 /*v805*/
	s_set_vgpr_msb 0xffb2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v8 /*v520*/, v8 /*v520*/, s60, -v17 /*v785*/
	s_set_vgpr_msb 0xb2c3                   ;  msbs: dst=3 src0=3 src1=0 src2=0
	v_exp_f32_e32 v38 /*v806*/, v38 /*v806*/
	s_set_vgpr_msb 0xc3b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v9 /*v521*/, v9 /*v521*/, s60, -v17 /*v785*/
	s_set_vgpr_msb 0xb2cf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v59 /*v827*/, v66 /*v834*/
	v_exp_f32_e32 v39 /*v807*/, v39 /*v807*/
	s_set_vgpr_msb 0xcfb2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v10 /*v522*/, v10 /*v522*/, s60, -v17 /*v785*/
	s_set_vgpr_msb 0xb2c3                   ;  msbs: dst=3 src0=3 src1=0 src2=0
	v_exp_f32_e32 v40 /*v808*/, v40 /*v808*/
	s_set_vgpr_msb 0xc3b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v11 /*v523*/, v11 /*v523*/, s60, -v17 /*v785*/
	s_set_vgpr_msb 0xb2cf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v60 /*v828*/, v66 /*v834*/
	v_exp_f32_e32 v41 /*v809*/, v41 /*v809*/
	s_set_vgpr_msb 0xcfb2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v12 /*v524*/, v12 /*v524*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v8 /*v520*/, v8 /*v520*/
	v_fma_f32 v13 /*v525*/, v13 /*v525*/, s60, -v17 /*v785*/
	s_set_vgpr_msb 0xb2cf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v61 /*v829*/, v66 /*v834*/
	s_set_vgpr_msb 0xcfb2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_exp_f32_e32 v9 /*v521*/, v9 /*v521*/
	v_fma_f32 v14 /*v526*/, v14 /*v526*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v10 /*v522*/, v10 /*v522*/
	v_fma_f32 v15 /*v527*/, v15 /*v527*/, s60, -v17 /*v785*/
	s_set_vgpr_msb 0xb2cf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v62 /*v830*/, v66 /*v834*/
	s_set_vgpr_msb 0xcf82                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	v_exp_f32_e32 v11 /*v523*/, v11 /*v523*/
	s_set_vgpr_msb 0x82f3                   ;  msbs: dst=3 src0=3 src1=0 src2=3
	v_fma_f32 v42 /*v810*/, v42 /*v810*/, s60, -v17 /*v785*/
	s_set_vgpr_msb 0xf382                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	v_exp_f32_e32 v12 /*v524*/, v12 /*v524*/
	s_set_vgpr_msb 0x82ff                   ;  msbs: dst=3 src0=3 src1=3 src2=3
	v_fma_f32 v43 /*v811*/, v43 /*v811*/, s60, -v17 /*v785*/
	v_add_f32_e32 v66 /*v834*/, v63 /*v831*/, v66 /*v834*/
	s_set_vgpr_msb 0xff82                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	v_exp_f32_e32 v13 /*v525*/, v13 /*v525*/
	s_set_vgpr_msb 0x82f3                   ;  msbs: dst=3 src0=3 src1=0 src2=3
	v_fma_f32 v44 /*v812*/, v44 /*v812*/, s60, -v17 /*v785*/
	s_set_vgpr_msb 0xf382                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	v_exp_f32_e32 v14 /*v526*/, v14 /*v526*/
	s_set_vgpr_msb 0x82ff                   ;  msbs: dst=3 src0=3 src1=3 src2=3
	v_fma_f32 v45 /*v813*/, v45 /*v813*/, s60, -v17 /*v785*/
	v_add_f32_e32 v66 /*v834*/, v64 /*v832*/, v66 /*v834*/
	s_set_vgpr_msb 0xff82                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	v_exp_f32_e32 v15 /*v527*/, v15 /*v527*/
	s_set_vgpr_msb 0x82ff                   ;  msbs: dst=3 src0=3 src1=3 src2=3
	v_fma_f32 v46 /*v814*/, v46 /*v814*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v42 /*v810*/, v42 /*v810*/
	v_fma_f32 v47 /*v815*/, v47 /*v815*/, s60, -v17 /*v785*/
	v_add_f32_e32 v66 /*v834*/, v65 /*v833*/, v66 /*v834*/
	v_exp_f32_e32 v43 /*v811*/, v43 /*v811*/
	v_fma_f32 v48 /*v816*/, v48 /*v816*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v44 /*v812*/, v44 /*v812*/
	v_fma_f32 v49 /*v817*/, v49 /*v817*/, s60, -v17 /*v785*/
	v_add_f32_e32 v66 /*v834*/, v34 /*v802*/, v66 /*v834*/
	v_exp_f32_e32 v45 /*v813*/, v45 /*v813*/
	v_fma_f32 v26 /*v794*/, v26 /*v794*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v46 /*v814*/, v46 /*v814*/
	v_fma_f32 v27 /*v795*/, v27 /*v795*/, s60, -v17 /*v785*/
	v_add_f32_e32 v66 /*v834*/, v35 /*v803*/, v66 /*v834*/
	v_exp_f32_e32 v47 /*v815*/, v47 /*v815*/
	v_fma_f32 v28 /*v796*/, v28 /*v796*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v48 /*v816*/, v48 /*v816*/
	v_fma_f32 v29 /*v797*/, v29 /*v797*/, s60, -v17 /*v785*/
	v_add_f32_e32 v66 /*v834*/, v36 /*v804*/, v66 /*v834*/
	v_exp_f32_e32 v49 /*v817*/, v49 /*v817*/
	v_fma_f32 v30 /*v798*/, v30 /*v798*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v26 /*v794*/, v26 /*v794*/
	v_fma_f32 v31 /*v799*/, v31 /*v799*/, s60, -v17 /*v785*/
	v_add_f32_e32 v66 /*v834*/, v37 /*v805*/, v66 /*v834*/
	v_exp_f32_e32 v27 /*v795*/, v27 /*v795*/
	v_fma_f32 v32 /*v800*/, v32 /*v800*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v28 /*v796*/, v28 /*v796*/
	v_fma_f32 v33 /*v801*/, v33 /*v801*/, s60, -v17 /*v785*/
	v_add_f32_e32 v66 /*v834*/, v38 /*v806*/, v66 /*v834*/
	v_exp_f32_e32 v29 /*v797*/, v29 /*v797*/
	v_fma_f32 v18 /*v786*/, v18 /*v786*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v30 /*v798*/, v30 /*v798*/
	v_fma_f32 v19 /*v787*/, v19 /*v787*/, s60, -v17 /*v785*/
	v_add_f32_e32 v66 /*v834*/, v39 /*v807*/, v66 /*v834*/
	v_exp_f32_e32 v31 /*v799*/, v31 /*v799*/
	v_fma_f32 v20 /*v788*/, v20 /*v788*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v32 /*v800*/, v32 /*v800*/
	v_fma_f32 v21 /*v789*/, v21 /*v789*/, s60, -v17 /*v785*/
	v_add_f32_e32 v66 /*v834*/, v40 /*v808*/, v66 /*v834*/
	v_exp_f32_e32 v33 /*v801*/, v33 /*v801*/
	v_fma_f32 v22 /*v790*/, v22 /*v790*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v18 /*v786*/, v18 /*v786*/
	v_fma_f32 v23 /*v791*/, v23 /*v791*/, s60, -v17 /*v785*/
	v_add_f32_e32 v66 /*v834*/, v41 /*v809*/, v66 /*v834*/
	v_exp_f32_e32 v19 /*v787*/, v19 /*v787*/
	v_fma_f32 v24 /*v792*/, v24 /*v792*/, s60, -v17 /*v785*/
	v_exp_f32_e32 v20 /*v788*/, v20 /*v788*/
	v_fma_f32 v25 /*v793*/, v25 /*v793*/, s60, -v17 /*v785*/
	s_set_vgpr_msb 0xffce                   ;  msbs: dst=3 src0=2 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v8 /*v520*/, v66 /*v834*/
	s_set_vgpr_msb 0xcec3                   ;  msbs: dst=3 src0=3 src1=0 src2=0
	v_exp_f32_e32 v21 /*v789*/, v21 /*v789*/
	s_set_vgpr_msb 0xc3b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v0 /*v512*/, v0 /*v512*/, s60, -v17 /*v785*/
	s_set_vgpr_msb 0xb2c3                   ;  msbs: dst=3 src0=3 src1=0 src2=0
	v_exp_f32_e32 v22 /*v790*/, v22 /*v790*/
	s_set_vgpr_msb 0xc3b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v2 /*v514*/, v2 /*v514*/, s60, -v17 /*v785*/
	s_set_vgpr_msb 0xb2ce                   ;  msbs: dst=3 src0=2 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v9 /*v521*/, v66 /*v834*/
	s_set_vgpr_msb 0xcec3                   ;  msbs: dst=3 src0=3 src1=0 src2=0
	v_exp_f32_e32 v23 /*v791*/, v23 /*v791*/
	s_set_vgpr_msb 0xc3ce                   ;  msbs: dst=3 src0=2 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v10 /*v522*/, v66 /*v834*/
	s_set_vgpr_msb 0xcec3                   ;  msbs: dst=3 src0=3 src1=0 src2=0
	v_exp_f32_e32 v24 /*v792*/, v24 /*v792*/
	s_set_vgpr_msb 0xc3ce                   ;  msbs: dst=3 src0=2 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v11 /*v523*/, v66 /*v834*/
	s_set_vgpr_msb 0xcec3                   ;  msbs: dst=3 src0=3 src1=0 src2=0
	v_exp_f32_e32 v25 /*v793*/, v25 /*v793*/
	s_set_vgpr_msb 0xc3ce                   ;  msbs: dst=3 src0=2 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v12 /*v524*/, v66 /*v834*/
	v_exp_f32_e32 v68 /*v836*/, v2 /*v514*/
	v_nop
	s_set_vgpr_msb 0xceb2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v2 /*v514*/, v4 /*v516*/, s60, -v17 /*v785*/
	s_set_vgpr_msb 0xb2ce                   ;  msbs: dst=3 src0=2 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v13 /*v525*/, v66 /*v834*/
	v_exp_f32_e32 v70 /*v838*/, v2 /*v514*/
	v_nop
	s_set_vgpr_msb 0xceb2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v2 /*v514*/, v6 /*v518*/, s60, -v17 /*v785*/
	s_set_vgpr_msb 0xb2ce                   ;  msbs: dst=3 src0=2 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v14 /*v526*/, v66 /*v834*/
	v_exp_f32_e32 v72 /*v840*/, v2 /*v514*/
	v_add_f32_e32 v66 /*v834*/, v15 /*v527*/, v66 /*v834*/
	s_set_vgpr_msb 0xcecf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v42 /*v810*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v43 /*v811*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v44 /*v812*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v45 /*v813*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v46 /*v814*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v47 /*v815*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v48 /*v816*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v49 /*v817*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v26 /*v794*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v27 /*v795*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v28 /*v796*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v29 /*v797*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v30 /*v798*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v31 /*v799*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v32 /*v800*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v33 /*v801*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v18 /*v786*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v19 /*v787*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v20 /*v788*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v21 /*v789*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v22 /*v790*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v23 /*v791*/, v66 /*v834*/
	v_add_f32_e32 v67 /*v835*/, v24 /*v792*/, v66 /*v834*/
	s_set_vgpr_msb 0xcfc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v66 /*v834*/, v0 /*v512*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v0 /*v512*/, v1 /*v513*/, s60, -v17 /*v785*/
	s_set_vgpr_msb 0xb28f                   ;  msbs: dst=2 src0=3 src1=3 src2=0
	v_add_f32_e32 v1 /*v513*/, v25 /*v793*/, v67 /*v835*/
	s_set_vgpr_msb 0x8fc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v67 /*v835*/, v0 /*v512*/
	v_nop
	s_set_vgpr_msb 0xc28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v0 /*v512*/, v66 /*v834*/, v1 /*v513*/
	s_set_vgpr_msb 0x8bb2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v1 /*v513*/, v3 /*v515*/, s60, -v17 /*v785*/
	s_set_vgpr_msb 0xb28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v0 /*v512*/, v67 /*v835*/, v0 /*v512*/
	s_set_vgpr_msb 0x8bc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v69 /*v837*/, v1 /*v513*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v1 /*v513*/, v5 /*v517*/, s60, -v17 /*v785*/
	s_set_vgpr_msb 0xb28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v0 /*v512*/, v68 /*v836*/, v0 /*v512*/
	s_set_vgpr_msb 0x8bc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v71 /*v839*/, v1 /*v513*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v1 /*v513*/, v7 /*v519*/, s60, -v17 /*v785*/
	s_set_vgpr_msb 0xb28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v0 /*v512*/, v69 /*v837*/, v0 /*v512*/
	s_set_vgpr_msb 0x8bc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v73 /*v841*/, v1 /*v513*/
	s_set_vgpr_msb 0xc28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v0 /*v512*/, v70 /*v838*/, v0 /*v512*/
	v_add_f32_e32 v0 /*v512*/, v71 /*v839*/, v0 /*v512*/
	v_add_f32_e32 v0 /*v512*/, v72 /*v840*/, v0 /*v512*/
	s_set_vgpr_msb 0x8bcb                   ;  msbs: dst=3 src0=3 src1=2 src2=0
	v_add_f32_e32 v190 /*v958*/, v73 /*v841*/, v0 /*v512*/
	v_mov_b32_e32 v191 /*v959*/, v190 /*v958*/
	s_set_vgpr_msb 0xcb0c                   ;  msbs: dst=0 src0=0 src1=3 src2=0
	v_sub_f32_e32 v36, v36, v17 /*v785*/
	s_set_vgpr_msb 0xcc3                    ;  msbs: dst=3 src0=3 src1=0 src2=0
	v_permlanex16_b32 v191 /*v959*/, v191 /*v959*/, s65, 0xfedcba98
	s_set_vgpr_msb 0xc300                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_exp_f32_e32 v36, v36
	s_cbranch_vccz .LBB0_10
; %bb.9:                                ;   in Loop: Header=BB0_6 Depth=1
	v_nop
	s_set_vgpr_msb 0xc3                     ;  msbs: dst=3 src0=3 src1=0 src2=0
	v_pk_mul_f32 v[14:15] /*v[782:783]*/, v[14:15] /*v[782:783]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13] /*v[780:781]*/, v[12:13] /*v[780:781]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11] /*v[778:779]*/, v[10:11] /*v[778:779]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9] /*v[776:777]*/, v[8:9] /*v[776:777]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7] /*v[774:775]*/, v[6:7] /*v[774:775]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5] /*v[772:773]*/, v[4:5] /*v[772:773]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3] /*v[770:771]*/, v[2:3] /*v[770:771]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1] /*v[768:769]*/, v[0:1] /*v[768:769]*/, v[36:37] op_sel_hi:[1,0]
	s_set_vgpr_msb 0xc382                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[254:255] /*v[766:767]*/, v[254:255] /*v[766:767]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[252:253] /*v[764:765]*/, v[252:253] /*v[764:765]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[250:251] /*v[762:763]*/, v[250:251] /*v[762:763]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[248:249] /*v[760:761]*/, v[248:249] /*v[760:761]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[246:247] /*v[758:759]*/, v[246:247] /*v[758:759]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[244:245] /*v[756:757]*/, v[244:245] /*v[756:757]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[242:243] /*v[754:755]*/, v[242:243] /*v[754:755]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[240:241] /*v[752:753]*/, v[240:241] /*v[752:753]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[238:239] /*v[750:751]*/, v[238:239] /*v[750:751]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[236:237] /*v[748:749]*/, v[236:237] /*v[748:749]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[234:235] /*v[746:747]*/, v[234:235] /*v[746:747]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[232:233] /*v[744:745]*/, v[232:233] /*v[744:745]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[230:231] /*v[742:743]*/, v[230:231] /*v[742:743]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[228:229] /*v[740:741]*/, v[228:229] /*v[740:741]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[226:227] /*v[738:739]*/, v[226:227] /*v[738:739]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[224:225] /*v[736:737]*/, v[224:225] /*v[736:737]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[222:223] /*v[734:735]*/, v[222:223] /*v[734:735]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[220:221] /*v[732:733]*/, v[220:221] /*v[732:733]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[218:219] /*v[730:731]*/, v[218:219] /*v[730:731]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[216:217] /*v[728:729]*/, v[216:217] /*v[728:729]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[214:215] /*v[726:727]*/, v[214:215] /*v[726:727]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[212:213] /*v[724:725]*/, v[212:213] /*v[724:725]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[210:211] /*v[722:723]*/, v[210:211] /*v[722:723]*/, v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[208:209] /*v[720:721]*/, v[208:209] /*v[720:721]*/, v[36:37] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_10:                               ; %_ZZ12k_mla_fwd_mxENKUlT_E4_clISt17integral_constantIiLi0EEEEDaS_.exit.i.i
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_set_vgpr_msb 0x83                     ;  msbs: dst=2 src0=3 src1=0 src2=0
	v_cvt_scalef32_pk8_fp8_f32 v[0:1] /*v[512:513]*/, v[50:57] /*v[818:825]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[2:3] /*v[514:515]*/, v[58:65] /*v[826:833]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[4:5] /*v[516:517]*/, v[34:41] /*v[802:809]*/, 0x3c000000
	s_set_vgpr_msb 0x8382                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	v_cvt_scalef32_pk8_fp8_f32 v[6:7] /*v[518:519]*/, v[8:15] /*v[520:527]*/, 0x3c000000
	s_set_vgpr_msb 0x8283                   ;  msbs: dst=2 src0=3 src1=0 src2=0
	v_cvt_scalef32_pk8_fp8_f32 v[8:9] /*v[520:521]*/, v[42:49] /*v[810:817]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[10:11] /*v[522:523]*/, v[26:33] /*v[794:801]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[12:13] /*v[524:525]*/, v[18:25] /*v[786:793]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[14:15] /*v[526:527]*/, v[66:73] /*v[834:841]*/, 0x3c000000
	s_set_vgpr_msb 0x83f8                   ;  msbs: dst=3 src0=0 src1=2 src2=3
	v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15] /*v[776:783]*/, v[128:143], v[0:15] /*v[512:527]*/, v[8:15] /*v[776:783]*/, v34, v16
	s_cmp_le_i32 s6, s61
	v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7] /*v[768:775]*/, v[144:159], v[0:15] /*v[512:527]*/, v[0:7] /*v[768:775]*/, v34, v16 matrix_a_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xf8a8                   ;  msbs: dst=2 src0=0 src1=2 src2=2
	v_wmma_scale_f32_16x16x128_f8f6f4 v[248:255] /*v[760:767]*/, v[160:175], v[0:15] /*v[512:527]*/, v[248:255] /*v[760:767]*/, v35, v16
	v_wmma_scale_f32_16x16x128_f8f6f4 v[240:247] /*v[752:759]*/, v[176:191], v[0:15] /*v[512:527]*/, v[240:247] /*v[752:759]*/, v35, v16 matrix_a_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[232:239] /*v[744:751]*/, v[192:207], v[0:15] /*v[512:527]*/, v[232:239] /*v[744:751]*/, v32, v16
	v_wmma_scale_f32_16x16x128_f8f6f4 v[224:231] /*v[736:743]*/, v[208:223], v[0:15] /*v[512:527]*/, v[224:231] /*v[736:743]*/, v32, v16 matrix_a_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[216:223] /*v[728:735]*/, v[224:239], v[0:15] /*v[512:527]*/, v[216:223] /*v[728:735]*/, v33, v16
	v_wmma_scale_f32_16x16x128_f8f6f4 v[208:215] /*v[720:727]*/, v[240:255], v[0:15] /*v[512:527]*/, v[208:215] /*v[720:727]*/, v33, v16 matrix_a_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa8cd                   ;  msbs: dst=3 src0=1 src1=3 src2=0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[58:65] /*v[826:833]*/, v[0:15] /*v[256:271]*/, v[98:113] /*v[866:881]*/, 0, v17, v29
	v_wmma_scale_f32_16x16x128_f8f6f4 v[50:57] /*v[818:825]*/, v[32:47] /*v[288:303]*/, v[98:113] /*v[866:881]*/, 0, v18, v29
	v_wmma_scale_f32_16x16x128_f8f6f4 v[42:49] /*v[810:817]*/, v[64:79] /*v[320:335]*/, v[98:113] /*v[866:881]*/, 0, v19, v29
	v_wmma_scale_f32_16x16x128_f8f6f4 v[34:41] /*v[802:809]*/, v[96:111] /*v[352:367]*/, v[98:113] /*v[866:881]*/, 0, v20, v29
	v_wmma_scale_f32_16x16x128_f8f6f4 v[26:33] /*v[794:801]*/, v[128:143] /*v[384:399]*/, v[98:113] /*v[866:881]*/, 0, v21, v29
	v_wmma_scale_f32_16x16x128_f8f6f4 v[18:25] /*v[786:793]*/, v[160:175] /*v[416:431]*/, v[98:113] /*v[866:881]*/, 0, v22, v29
	s_set_vgpr_msb 0xcd8d                   ;  msbs: dst=2 src0=1 src1=3 src2=0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15] /*v[520:527]*/, v[192:207] /*v[448:463]*/, v[98:113] /*v[866:881]*/, 0, v23, v29
	v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7] /*v[512:519]*/, v[224:239] /*v[480:495]*/, v[98:113] /*v[866:881]*/, 0, v24, v29
	s_set_vgpr_msb 0x8df1                   ;  msbs: dst=3 src0=1 src1=0 src2=3
	v_wmma_scale_f32_16x16x128_f8f6f4 v[58:65] /*v[826:833]*/, v[16:31] /*v[272:287]*/, v[48:63], v[58:65] /*v[826:833]*/, v17, v29 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[50:57] /*v[818:825]*/, v[48:63] /*v[304:319]*/, v[48:63], v[50:57] /*v[818:825]*/, v18, v29 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[42:49] /*v[810:817]*/, v[80:95] /*v[336:351]*/, v[48:63], v[42:49] /*v[810:817]*/, v19, v29 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[34:41] /*v[802:809]*/, v[112:127] /*v[368:383]*/, v[48:63], v[34:41] /*v[802:809]*/, v20, v29 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[26:33] /*v[794:801]*/, v[144:159] /*v[400:415]*/, v[48:63], v[26:33] /*v[794:801]*/, v21, v29 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[18:25] /*v[786:793]*/, v[176:191] /*v[432:447]*/, v[48:63], v[18:25] /*v[786:793]*/, v22, v29 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xf1a1                   ;  msbs: dst=2 src0=1 src1=0 src2=2
	v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15] /*v[520:527]*/, v[208:223] /*v[464:479]*/, v[48:63], v[8:15] /*v[520:527]*/, v23, v29 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7] /*v[512:519]*/, v[240:255] /*v[496:511]*/, v[48:63], v[0:7] /*v[512:519]*/, v24, v29 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_scc1 .LBB0_12
; %bb.11:                               ; %_ZZZZ12k_mla_fwd_mxENKUlT_E4_clISt17integral_constantIiLi1EEEEDaS_ENKUlS_E0_clIS2_IiLi0EEEEDaS_ENKUlS_E_clIS6_EEDaS_.exit.i.i.i.i.i.i.i.i
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_set_vgpr_msb 0xcf                     ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v247 /*v1015*/, v182 /*v950*/
	v_cndmask_b32_e32 v58 /*v826*/, 0xff800000, v58 /*v826*/, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v247 /*v1015*/, v182 /*v950*/
	v_cndmask_b32_e32 v59 /*v827*/, 0xff800000, v59 /*v827*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v246 /*v1014*/, v182 /*v950*/
	v_cndmask_b32_e32 v60 /*v828*/, 0xff800000, v60 /*v828*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v245 /*v1013*/, v182 /*v950*/
	v_cndmask_b32_e32 v61 /*v829*/, 0xff800000, v61 /*v829*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v244 /*v1012*/, v182 /*v950*/
	v_cndmask_b32_e32 v62 /*v830*/, 0xff800000, v62 /*v830*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v243 /*v1011*/, v182 /*v950*/
	v_cndmask_b32_e32 v63 /*v831*/, 0xff800000, v63 /*v831*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v241 /*v1009*/, v182 /*v950*/
	v_cndmask_b32_e32 v64 /*v832*/, 0xff800000, v64 /*v832*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v242 /*v1010*/, v182 /*v950*/
	v_cndmask_b32_e32 v65 /*v833*/, 0xff800000, v65 /*v833*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v240 /*v1008*/, v182 /*v950*/
	v_cndmask_b32_e32 v50 /*v818*/, 0xff800000, v50 /*v818*/, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v240 /*v1008*/, v182 /*v950*/
	v_cndmask_b32_e32 v51 /*v819*/, 0xff800000, v51 /*v819*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v239 /*v1007*/, v182 /*v950*/
	v_cndmask_b32_e32 v52 /*v820*/, 0xff800000, v52 /*v820*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v238 /*v1006*/, v182 /*v950*/
	v_cndmask_b32_e32 v53 /*v821*/, 0xff800000, v53 /*v821*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v237 /*v1005*/, v182 /*v950*/
	v_cndmask_b32_e32 v54 /*v822*/, 0xff800000, v54 /*v822*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v236 /*v1004*/, v182 /*v950*/
	v_cndmask_b32_e32 v55 /*v823*/, 0xff800000, v55 /*v823*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v234 /*v1002*/, v182 /*v950*/
	v_cndmask_b32_e32 v56 /*v824*/, 0xff800000, v56 /*v824*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v235 /*v1003*/, v182 /*v950*/
	v_cndmask_b32_e32 v57 /*v825*/, 0xff800000, v57 /*v825*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v233 /*v1001*/, v182 /*v950*/
	v_cndmask_b32_e32 v42 /*v810*/, 0xff800000, v42 /*v810*/, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v233 /*v1001*/, v182 /*v950*/
	v_cndmask_b32_e32 v43 /*v811*/, 0xff800000, v43 /*v811*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v232 /*v1000*/, v182 /*v950*/
	v_cndmask_b32_e32 v44 /*v812*/, 0xff800000, v44 /*v812*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v231 /*v999*/, v182 /*v950*/
	v_cndmask_b32_e32 v45 /*v813*/, 0xff800000, v45 /*v813*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v230 /*v998*/, v182 /*v950*/
	v_cndmask_b32_e32 v46 /*v814*/, 0xff800000, v46 /*v814*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v229 /*v997*/, v182 /*v950*/
	v_cndmask_b32_e32 v47 /*v815*/, 0xff800000, v47 /*v815*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v227 /*v995*/, v182 /*v950*/
	v_cndmask_b32_e32 v48 /*v816*/, 0xff800000, v48 /*v816*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v228 /*v996*/, v182 /*v950*/
	v_cndmask_b32_e32 v49 /*v817*/, 0xff800000, v49 /*v817*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v226 /*v994*/, v182 /*v950*/
	v_cndmask_b32_e32 v34 /*v802*/, 0xff800000, v34 /*v802*/, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v226 /*v994*/, v182 /*v950*/
	v_cndmask_b32_e32 v35 /*v803*/, 0xff800000, v35 /*v803*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v225 /*v993*/, v182 /*v950*/
	v_cndmask_b32_e32 v36 /*v804*/, 0xff800000, v36 /*v804*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v224 /*v992*/, v182 /*v950*/
	v_cndmask_b32_e32 v37 /*v805*/, 0xff800000, v37 /*v805*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v223 /*v991*/, v182 /*v950*/
	v_cndmask_b32_e32 v38 /*v806*/, 0xff800000, v38 /*v806*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v222 /*v990*/, v182 /*v950*/
	v_cndmask_b32_e32 v39 /*v807*/, 0xff800000, v39 /*v807*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v220 /*v988*/, v182 /*v950*/
	v_cndmask_b32_e32 v40 /*v808*/, 0xff800000, v40 /*v808*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v221 /*v989*/, v182 /*v950*/
	v_cndmask_b32_e32 v41 /*v809*/, 0xff800000, v41 /*v809*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v219 /*v987*/, v182 /*v950*/
	v_cndmask_b32_e32 v26 /*v794*/, 0xff800000, v26 /*v794*/, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v219 /*v987*/, v182 /*v950*/
	v_cndmask_b32_e32 v27 /*v795*/, 0xff800000, v27 /*v795*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v218 /*v986*/, v182 /*v950*/
	v_cndmask_b32_e32 v28 /*v796*/, 0xff800000, v28 /*v796*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v217 /*v985*/, v182 /*v950*/
	v_cndmask_b32_e32 v29 /*v797*/, 0xff800000, v29 /*v797*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v216 /*v984*/, v182 /*v950*/
	v_cndmask_b32_e32 v30 /*v798*/, 0xff800000, v30 /*v798*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v215 /*v983*/, v182 /*v950*/
	v_cndmask_b32_e32 v31 /*v799*/, 0xff800000, v31 /*v799*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v213 /*v981*/, v182 /*v950*/
	v_cndmask_b32_e32 v32 /*v800*/, 0xff800000, v32 /*v800*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v214 /*v982*/, v182 /*v950*/
	v_cndmask_b32_e32 v33 /*v801*/, 0xff800000, v33 /*v801*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v212 /*v980*/, v182 /*v950*/
	v_cndmask_b32_e32 v18 /*v786*/, 0xff800000, v18 /*v786*/, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v212 /*v980*/, v182 /*v950*/
	v_cndmask_b32_e32 v19 /*v787*/, 0xff800000, v19 /*v787*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v211 /*v979*/, v182 /*v950*/
	v_cndmask_b32_e32 v20 /*v788*/, 0xff800000, v20 /*v788*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v210 /*v978*/, v182 /*v950*/
	v_cndmask_b32_e32 v21 /*v789*/, 0xff800000, v21 /*v789*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v209 /*v977*/, v182 /*v950*/
	v_cndmask_b32_e32 v22 /*v790*/, 0xff800000, v22 /*v790*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v208 /*v976*/, v182 /*v950*/
	v_cndmask_b32_e32 v23 /*v791*/, 0xff800000, v23 /*v791*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v206 /*v974*/, v182 /*v950*/
	v_cndmask_b32_e32 v24 /*v792*/, 0xff800000, v24 /*v792*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v207 /*v975*/, v182 /*v950*/
	v_cndmask_b32_e32 v25 /*v793*/, 0xff800000, v25 /*v793*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v205 /*v973*/, v182 /*v950*/
	s_set_vgpr_msb 0xcf88                   ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v8 /*v520*/, 0xff800000, v8 /*v520*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_lt_i32_e32 vcc_lo, v205 /*v973*/, v182 /*v950*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v9 /*v521*/, 0xff800000, v9 /*v521*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v204 /*v972*/, v182 /*v950*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v10 /*v522*/, 0xff800000, v10 /*v522*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v203 /*v971*/, v182 /*v950*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v11 /*v523*/, 0xff800000, v11 /*v523*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v202 /*v970*/, v182 /*v950*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v12 /*v524*/, 0xff800000, v12 /*v524*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v201 /*v969*/, v182 /*v950*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v13 /*v525*/, 0xff800000, v13 /*v525*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v199 /*v967*/, v182 /*v950*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v14 /*v526*/, 0xff800000, v14 /*v526*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v200 /*v968*/, v182 /*v950*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v15 /*v527*/, 0xff800000, v15 /*v527*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v198 /*v966*/, v182 /*v950*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v0 /*v512*/, 0xff800000, v0 /*v512*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_lt_i32_e32 vcc_lo, v198 /*v966*/, v182 /*v950*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v1 /*v513*/, 0xff800000, v1 /*v513*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v197 /*v965*/, v182 /*v950*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v2 /*v514*/, 0xff800000, v2 /*v514*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v196 /*v964*/, v182 /*v950*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v3 /*v515*/, 0xff800000, v3 /*v515*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v195 /*v963*/, v182 /*v950*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v4 /*v516*/, 0xff800000, v4 /*v516*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v194 /*v962*/, v182 /*v950*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v5 /*v517*/, 0xff800000, v5 /*v517*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v193 /*v961*/, v182 /*v950*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v6 /*v518*/, 0xff800000, v6 /*v518*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v192 /*v960*/, v182 /*v950*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v7 /*v519*/, 0xff800000, v7 /*v519*/, vcc_lo
	s_set_vgpr_msb 0x8800                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_12:                               ;   in Loop: Header=BB0_6 Depth=1
	s_set_vgpr_msb 0xff                     ;  msbs: dst=3 src0=3 src1=3 src2=3
	v_max3_num_f32 v66 /*v834*/, v58 /*v826*/, 0xff800000, v59 /*v827*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v60 /*v828*/, v61 /*v829*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v62 /*v830*/, v63 /*v831*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v64 /*v832*/, v65 /*v833*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v50 /*v818*/, v51 /*v819*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v52 /*v820*/, v53 /*v821*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v54 /*v822*/, v55 /*v823*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v56 /*v824*/, v57 /*v825*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v42 /*v810*/, v43 /*v811*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v44 /*v812*/, v45 /*v813*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v46 /*v814*/, v47 /*v815*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v48 /*v816*/, v49 /*v817*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v34 /*v802*/, v35 /*v803*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v36 /*v804*/, v37 /*v805*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v38 /*v806*/, v39 /*v807*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v40 /*v808*/, v41 /*v809*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v26 /*v794*/, v27 /*v795*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v28 /*v796*/, v29 /*v797*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v30 /*v798*/, v31 /*v799*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v32 /*v800*/, v33 /*v801*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v18 /*v786*/, v19 /*v787*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v20 /*v788*/, v21 /*v789*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v22 /*v790*/, v23 /*v791*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v24 /*v792*/, v25 /*v793*/
	s_set_vgpr_msb 0xffeb                   ;  msbs: dst=3 src0=3 src1=2 src2=2
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v8 /*v520*/, v9 /*v521*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v10 /*v522*/, v11 /*v523*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v12 /*v524*/, v13 /*v525*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v14 /*v526*/, v15 /*v527*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v0 /*v512*/, v1 /*v513*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v2 /*v514*/, v3 /*v515*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v4 /*v516*/, v5 /*v517*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v6 /*v518*/, v7 /*v519*/
	v_mov_b32_e32 v67 /*v835*/, v66 /*v834*/
	v_permlanex16_b32 v67 /*v835*/, v67 /*v835*/, s65, 0xfedcba98
	s_set_vgpr_msb 0xebcf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_max_num_f32_e32 v66 /*v834*/, v66 /*v834*/, v67 /*v835*/
	v_fma_f32 v67 /*v835*/, v66 /*v834*/, s60, -v38
	v_mul_f32_e32 v66 /*v834*/, s60, v66 /*v834*/
	v_cmp_lt_f32_e32 vcc_lo, 0x3fb8aa3b, v67 /*v835*/
	s_set_vgpr_msb 0xcfcc                   ;  msbs: dst=3 src0=0 src1=3 src2=0
	v_max_num_f32_e32 v66 /*v834*/, v38, v66 /*v834*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0xccf3                   ;  msbs: dst=3 src0=3 src1=0 src2=3
	v_cndmask_b32_e64 v189 /*v957*/, v66 /*v834*/, v38, s2
	v_fma_f32 v58 /*v826*/, v58 /*v826*/, s60, -v189 /*v957*/
	v_fma_f32 v59 /*v827*/, v59 /*v827*/, s60, -v189 /*v957*/
	v_fma_f32 v60 /*v828*/, v60 /*v828*/, s60, -v189 /*v957*/
	v_fma_f32 v61 /*v829*/, v61 /*v829*/, s60, -v189 /*v957*/
	v_fma_f32 v62 /*v830*/, v62 /*v830*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v58 /*v826*/, v58 /*v826*/
	v_fma_f32 v63 /*v831*/, v63 /*v831*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v59 /*v827*/, v59 /*v827*/
	v_fma_f32 v64 /*v832*/, v64 /*v832*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v60 /*v828*/, v60 /*v828*/
	v_fma_f32 v65 /*v833*/, v65 /*v833*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v61 /*v829*/, v61 /*v829*/
	v_fma_f32 v50 /*v818*/, v50 /*v818*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v62 /*v830*/, v62 /*v830*/
	v_fma_f32 v51 /*v819*/, v51 /*v819*/, s60, -v189 /*v957*/
	s_set_vgpr_msb 0xf3ff                   ;  msbs: dst=3 src0=3 src1=3 src2=3
	v_add_f32_e32 v66 /*v834*/, v58 /*v826*/, v59 /*v827*/
	v_exp_f32_e32 v63 /*v831*/, v63 /*v831*/
	v_fma_f32 v52 /*v820*/, v52 /*v820*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v64 /*v832*/, v64 /*v832*/
	v_fma_f32 v53 /*v821*/, v53 /*v821*/, s60, -v189 /*v957*/
	v_add_f32_e32 v66 /*v834*/, v60 /*v828*/, v66 /*v834*/
	v_exp_f32_e32 v65 /*v833*/, v65 /*v833*/
	v_fma_f32 v54 /*v822*/, v54 /*v822*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v50 /*v818*/, v50 /*v818*/
	v_fma_f32 v55 /*v823*/, v55 /*v823*/, s60, -v189 /*v957*/
	v_add_f32_e32 v66 /*v834*/, v61 /*v829*/, v66 /*v834*/
	v_exp_f32_e32 v51 /*v819*/, v51 /*v819*/
	v_fma_f32 v56 /*v824*/, v56 /*v824*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v52 /*v820*/, v52 /*v820*/
	v_fma_f32 v57 /*v825*/, v57 /*v825*/, s60, -v189 /*v957*/
	v_add_f32_e32 v66 /*v834*/, v62 /*v830*/, v66 /*v834*/
	v_exp_f32_e32 v53 /*v821*/, v53 /*v821*/
	v_fma_f32 v42 /*v810*/, v42 /*v810*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v54 /*v822*/, v54 /*v822*/
	v_fma_f32 v43 /*v811*/, v43 /*v811*/, s60, -v189 /*v957*/
	v_add_f32_e32 v66 /*v834*/, v63 /*v831*/, v66 /*v834*/
	v_exp_f32_e32 v55 /*v823*/, v55 /*v823*/
	v_fma_f32 v44 /*v812*/, v44 /*v812*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v56 /*v824*/, v56 /*v824*/
	v_fma_f32 v45 /*v813*/, v45 /*v813*/, s60, -v189 /*v957*/
	v_add_f32_e32 v66 /*v834*/, v64 /*v832*/, v66 /*v834*/
	v_exp_f32_e32 v57 /*v825*/, v57 /*v825*/
	v_fma_f32 v46 /*v814*/, v46 /*v814*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v42 /*v810*/, v42 /*v810*/
	v_fma_f32 v47 /*v815*/, v47 /*v815*/, s60, -v189 /*v957*/
	v_add_f32_e32 v66 /*v834*/, v65 /*v833*/, v66 /*v834*/
	v_exp_f32_e32 v43 /*v811*/, v43 /*v811*/
	v_fma_f32 v48 /*v816*/, v48 /*v816*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v44 /*v812*/, v44 /*v812*/
	v_fma_f32 v49 /*v817*/, v49 /*v817*/, s60, -v189 /*v957*/
	v_add_f32_e32 v66 /*v834*/, v50 /*v818*/, v66 /*v834*/
	v_exp_f32_e32 v45 /*v813*/, v45 /*v813*/
	v_fma_f32 v34 /*v802*/, v34 /*v802*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v46 /*v814*/, v46 /*v814*/
	v_fma_f32 v35 /*v803*/, v35 /*v803*/, s60, -v189 /*v957*/
	v_add_f32_e32 v66 /*v834*/, v51 /*v819*/, v66 /*v834*/
	v_exp_f32_e32 v47 /*v815*/, v47 /*v815*/
	v_fma_f32 v36 /*v804*/, v36 /*v804*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v48 /*v816*/, v48 /*v816*/
	v_fma_f32 v37 /*v805*/, v37 /*v805*/, s60, -v189 /*v957*/
	v_add_f32_e32 v66 /*v834*/, v52 /*v820*/, v66 /*v834*/
	v_exp_f32_e32 v49 /*v817*/, v49 /*v817*/
	v_fma_f32 v38 /*v806*/, v38 /*v806*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v34 /*v802*/, v34 /*v802*/
	v_fma_f32 v39 /*v807*/, v39 /*v807*/, s60, -v189 /*v957*/
	v_add_f32_e32 v66 /*v834*/, v53 /*v821*/, v66 /*v834*/
	v_exp_f32_e32 v35 /*v803*/, v35 /*v803*/
	v_fma_f32 v40 /*v808*/, v40 /*v808*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v36 /*v804*/, v36 /*v804*/
	v_fma_f32 v41 /*v809*/, v41 /*v809*/, s60, -v189 /*v957*/
	v_add_f32_e32 v66 /*v834*/, v54 /*v822*/, v66 /*v834*/
	v_exp_f32_e32 v37 /*v805*/, v37 /*v805*/
	v_fma_f32 v26 /*v794*/, v26 /*v794*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v38 /*v806*/, v38 /*v806*/
	v_fma_f32 v27 /*v795*/, v27 /*v795*/, s60, -v189 /*v957*/
	v_add_f32_e32 v66 /*v834*/, v55 /*v823*/, v66 /*v834*/
	v_exp_f32_e32 v39 /*v807*/, v39 /*v807*/
	v_fma_f32 v28 /*v796*/, v28 /*v796*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v40 /*v808*/, v40 /*v808*/
	v_fma_f32 v29 /*v797*/, v29 /*v797*/, s60, -v189 /*v957*/
	v_add_f32_e32 v66 /*v834*/, v56 /*v824*/, v66 /*v834*/
	v_exp_f32_e32 v41 /*v809*/, v41 /*v809*/
	v_fma_f32 v30 /*v798*/, v30 /*v798*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v26 /*v794*/, v26 /*v794*/
	v_fma_f32 v31 /*v799*/, v31 /*v799*/, s60, -v189 /*v957*/
	v_add_f32_e32 v66 /*v834*/, v57 /*v825*/, v66 /*v834*/
	v_exp_f32_e32 v27 /*v795*/, v27 /*v795*/
	v_fma_f32 v32 /*v800*/, v32 /*v800*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v28 /*v796*/, v28 /*v796*/
	v_fma_f32 v33 /*v801*/, v33 /*v801*/, s60, -v189 /*v957*/
	v_add_f32_e32 v66 /*v834*/, v42 /*v810*/, v66 /*v834*/
	v_exp_f32_e32 v29 /*v797*/, v29 /*v797*/
	v_fma_f32 v18 /*v786*/, v18 /*v786*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v30 /*v798*/, v30 /*v798*/
	v_fma_f32 v19 /*v787*/, v19 /*v787*/, s60, -v189 /*v957*/
	v_add_f32_e32 v66 /*v834*/, v43 /*v811*/, v66 /*v834*/
	v_exp_f32_e32 v31 /*v799*/, v31 /*v799*/
	v_fma_f32 v20 /*v788*/, v20 /*v788*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v32 /*v800*/, v32 /*v800*/
	v_fma_f32 v21 /*v789*/, v21 /*v789*/, s60, -v189 /*v957*/
	v_add_f32_e32 v66 /*v834*/, v44 /*v812*/, v66 /*v834*/
	v_exp_f32_e32 v33 /*v801*/, v33 /*v801*/
	v_fma_f32 v22 /*v790*/, v22 /*v790*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v18 /*v786*/, v18 /*v786*/
	v_fma_f32 v23 /*v791*/, v23 /*v791*/, s60, -v189 /*v957*/
	v_add_f32_e32 v66 /*v834*/, v45 /*v813*/, v66 /*v834*/
	v_exp_f32_e32 v19 /*v787*/, v19 /*v787*/
	v_fma_f32 v24 /*v792*/, v24 /*v792*/, s60, -v189 /*v957*/
	v_exp_f32_e32 v20 /*v788*/, v20 /*v788*/
	v_fma_f32 v25 /*v793*/, v25 /*v793*/, s60, -v189 /*v957*/
	v_add_f32_e32 v66 /*v834*/, v46 /*v814*/, v66 /*v834*/
	v_exp_f32_e32 v21 /*v789*/, v21 /*v789*/
	s_set_vgpr_msb 0xffb2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v8 /*v520*/, v8 /*v520*/, s60, -v189 /*v957*/
	s_set_vgpr_msb 0xb2c3                   ;  msbs: dst=3 src0=3 src1=0 src2=0
	v_exp_f32_e32 v22 /*v790*/, v22 /*v790*/
	s_set_vgpr_msb 0xc3b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v0 /*v512*/, v0 /*v512*/, s60, -v189 /*v957*/
	s_set_vgpr_msb 0xb2cf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v47 /*v815*/, v66 /*v834*/
	v_exp_f32_e32 v23 /*v791*/, v23 /*v791*/
	s_set_vgpr_msb 0xcfb2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v10 /*v522*/, v10 /*v522*/, s60, -v189 /*v957*/
	s_set_vgpr_msb 0xb2c3                   ;  msbs: dst=3 src0=3 src1=0 src2=0
	v_exp_f32_e32 v24 /*v792*/, v24 /*v792*/
	s_set_vgpr_msb 0xc3b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v2 /*v514*/, v2 /*v514*/, s60, -v189 /*v957*/
	s_set_vgpr_msb 0xb2cf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v48 /*v816*/, v66 /*v834*/
	v_exp_f32_e32 v25 /*v793*/, v25 /*v793*/
	s_set_vgpr_msb 0xcf0c                   ;  msbs: dst=0 src0=0 src1=3 src2=0
	v_sub_f32_e32 v38, v38, v189 /*v957*/
	s_set_vgpr_msb 0xcc2                    ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v74 /*v842*/, v0 /*v512*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v0 /*v512*/, v1 /*v513*/, s60, -v189 /*v957*/
	s_set_vgpr_msb 0xb2cf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v49 /*v817*/, v66 /*v834*/
	s_set_vgpr_msb 0xcfc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v68 /*v836*/, v10 /*v522*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v10 /*v522*/, v12 /*v524*/, s60, -v189 /*v957*/
	s_set_vgpr_msb 0xb2c2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v76 /*v844*/, v2 /*v514*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v2 /*v514*/, v4 /*v516*/, s60, -v189 /*v957*/
	s_set_vgpr_msb 0xb2cf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v34 /*v802*/, v66 /*v834*/
	s_set_vgpr_msb 0xcfc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v75 /*v843*/, v0 /*v512*/
	s_set_vgpr_msb 0xc2cf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v35 /*v803*/, v66 /*v834*/
	s_set_vgpr_msb 0xcfc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v70 /*v838*/, v10 /*v522*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v10 /*v522*/, v14 /*v526*/, s60, -v189 /*v957*/
	s_set_vgpr_msb 0xb2c2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v78 /*v846*/, v2 /*v514*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v2 /*v514*/, v6 /*v518*/, s60, -v189 /*v957*/
	s_set_vgpr_msb 0xb2cf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v36 /*v804*/, v66 /*v834*/
	s_set_vgpr_msb 0xcf00                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_exp_f32_e32 v38, v38
	s_set_vgpr_msb 0xcf                     ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v37 /*v805*/, v66 /*v834*/
	s_set_vgpr_msb 0xcfc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v72 /*v840*/, v10 /*v522*/
	s_set_vgpr_msb 0xc2cf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v38 /*v806*/, v66 /*v834*/
	s_set_vgpr_msb 0xcfc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v80 /*v848*/, v2 /*v514*/
	s_set_vgpr_msb 0xc2cf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v39 /*v807*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v40 /*v808*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v41 /*v809*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v26 /*v794*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v27 /*v795*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v28 /*v796*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v29 /*v797*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v30 /*v798*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v31 /*v799*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v32 /*v800*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v33 /*v801*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v18 /*v786*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v19 /*v787*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v20 /*v788*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v21 /*v789*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v22 /*v790*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v23 /*v791*/, v66 /*v834*/
	v_add_f32_e32 v67 /*v835*/, v24 /*v792*/, v66 /*v834*/
	s_set_vgpr_msb 0xcfc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v66 /*v834*/, v8 /*v520*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v8 /*v520*/, v9 /*v521*/, s60, -v189 /*v957*/
	s_set_vgpr_msb 0xb28f                   ;  msbs: dst=2 src0=3 src1=3 src2=0
	v_add_f32_e32 v9 /*v521*/, v25 /*v793*/, v67 /*v835*/
	s_set_vgpr_msb 0x8fc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v67 /*v835*/, v8 /*v520*/
	v_nop
	s_set_vgpr_msb 0xc28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v8 /*v520*/, v66 /*v834*/, v9 /*v521*/
	s_set_vgpr_msb 0x8bb2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v9 /*v521*/, v11 /*v523*/, s60, -v189 /*v957*/
	s_set_vgpr_msb 0xb28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v8 /*v520*/, v67 /*v835*/, v8 /*v520*/
	s_set_vgpr_msb 0x8bc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v69 /*v837*/, v9 /*v521*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v9 /*v521*/, v13 /*v525*/, s60, -v189 /*v957*/
	s_set_vgpr_msb 0xb28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v8 /*v520*/, v68 /*v836*/, v8 /*v520*/
	s_set_vgpr_msb 0x8bc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v71 /*v839*/, v9 /*v521*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v9 /*v521*/, v15 /*v527*/, s60, -v189 /*v957*/
	s_set_vgpr_msb 0xb28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v8 /*v520*/, v69 /*v837*/, v8 /*v520*/
	s_set_vgpr_msb 0x8bc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v73 /*v841*/, v9 /*v521*/
	s_set_vgpr_msb 0xc28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v8 /*v520*/, v70 /*v838*/, v8 /*v520*/
	v_add_f32_e32 v8 /*v520*/, v71 /*v839*/, v8 /*v520*/
	v_add_f32_e32 v8 /*v520*/, v72 /*v840*/, v8 /*v520*/
	v_add_f32_e32 v1 /*v513*/, v73 /*v841*/, v8 /*v520*/
	v_add_f32_e32 v0 /*v512*/, v74 /*v842*/, v1 /*v513*/
	s_set_vgpr_msb 0x8bb2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v1 /*v513*/, v3 /*v515*/, s60, -v189 /*v957*/
	s_set_vgpr_msb 0xb28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v0 /*v512*/, v75 /*v843*/, v0 /*v512*/
	s_set_vgpr_msb 0x8bc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v77 /*v845*/, v1 /*v513*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v1 /*v513*/, v5 /*v517*/, s60, -v189 /*v957*/
	s_set_vgpr_msb 0xb28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v0 /*v512*/, v76 /*v844*/, v0 /*v512*/
	s_set_vgpr_msb 0x8bc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v79 /*v847*/, v1 /*v513*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v1 /*v513*/, v7 /*v519*/, s60, -v189 /*v957*/
	s_set_vgpr_msb 0xb28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v0 /*v512*/, v77 /*v845*/, v0 /*v512*/
	s_set_vgpr_msb 0x8bc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v81 /*v849*/, v1 /*v513*/
	s_set_vgpr_msb 0xc28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v0 /*v512*/, v78 /*v846*/, v0 /*v512*/
	v_add_f32_e32 v0 /*v512*/, v79 /*v847*/, v0 /*v512*/
	v_add_f32_e32 v0 /*v512*/, v80 /*v848*/, v0 /*v512*/
	s_set_vgpr_msb 0x8bcb                   ;  msbs: dst=3 src0=3 src1=2 src2=0
	v_add_f32_e32 v249 /*v1017*/, v81 /*v849*/, v0 /*v512*/
	v_mov_b32_e32 v250 /*v1018*/, v249 /*v1017*/
	v_permlanex16_b32 v250 /*v1018*/, v250 /*v1018*/, s65, 0xfedcba98
	s_set_vgpr_msb 0xcb00                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_vccz .LBB0_14
; %bb.13:                               ;   in Loop: Header=BB0_6 Depth=1
	s_set_vgpr_msb 0x82                     ;  msbs: dst=2 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[206:207] /*v[718:719]*/, v[206:207] /*v[718:719]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[204:205] /*v[716:717]*/, v[204:205] /*v[716:717]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[202:203] /*v[714:715]*/, v[202:203] /*v[714:715]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[200:201] /*v[712:713]*/, v[200:201] /*v[712:713]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[198:199] /*v[710:711]*/, v[198:199] /*v[710:711]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[196:197] /*v[708:709]*/, v[196:197] /*v[708:709]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[194:195] /*v[706:707]*/, v[194:195] /*v[706:707]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[192:193] /*v[704:705]*/, v[192:193] /*v[704:705]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[190:191] /*v[702:703]*/, v[190:191] /*v[702:703]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[188:189] /*v[700:701]*/, v[188:189] /*v[700:701]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[186:187] /*v[698:699]*/, v[186:187] /*v[698:699]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[184:185] /*v[696:697]*/, v[184:185] /*v[696:697]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[182:183] /*v[694:695]*/, v[182:183] /*v[694:695]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[180:181] /*v[692:693]*/, v[180:181] /*v[692:693]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[178:179] /*v[690:691]*/, v[178:179] /*v[690:691]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[176:177] /*v[688:689]*/, v[176:177] /*v[688:689]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[174:175] /*v[686:687]*/, v[174:175] /*v[686:687]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[172:173] /*v[684:685]*/, v[172:173] /*v[684:685]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[170:171] /*v[682:683]*/, v[170:171] /*v[682:683]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[168:169] /*v[680:681]*/, v[168:169] /*v[680:681]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[166:167] /*v[678:679]*/, v[166:167] /*v[678:679]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[164:165] /*v[676:677]*/, v[164:165] /*v[676:677]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[162:163] /*v[674:675]*/, v[162:163] /*v[674:675]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[160:161] /*v[672:673]*/, v[160:161] /*v[672:673]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[158:159] /*v[670:671]*/, v[158:159] /*v[670:671]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[156:157] /*v[668:669]*/, v[156:157] /*v[668:669]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[154:155] /*v[666:667]*/, v[154:155] /*v[666:667]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[152:153] /*v[664:665]*/, v[152:153] /*v[664:665]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[150:151] /*v[662:663]*/, v[150:151] /*v[662:663]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[148:149] /*v[660:661]*/, v[148:149] /*v[660:661]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[146:147] /*v[658:659]*/, v[146:147] /*v[658:659]*/, v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[144:145] /*v[656:657]*/, v[144:145] /*v[656:657]*/, v[38:39] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_14:                               ; %_ZZ12k_mla_fwd_mxENKUlT_E4_clISt17integral_constantIiLi1EEEEDaS_.exit.i.i
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_set_vgpr_msb 0x83                     ;  msbs: dst=2 src0=3 src1=0 src2=0
	v_cvt_scalef32_pk8_fp8_f32 v[0:1] /*v[512:513]*/, v[58:65] /*v[826:833]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[2:3] /*v[514:515]*/, v[50:57] /*v[818:825]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[4:5] /*v[516:517]*/, v[42:49] /*v[810:817]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[6:7] /*v[518:519]*/, v[34:41] /*v[802:809]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[8:9] /*v[520:521]*/, v[26:33] /*v[794:801]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[10:11] /*v[522:523]*/, v[18:25] /*v[786:793]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[12:13] /*v[524:525]*/, v[66:73] /*v[834:841]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[14:15] /*v[526:527]*/, v[74:81] /*v[842:849]*/, 0x3c000000
	s_set_vgpr_msb 0x83a8                   ;  msbs: dst=2 src0=0 src1=2 src2=2
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207] /*v[712:719]*/, v[128:143], v[0:15] /*v[512:527]*/, v[200:207] /*v[712:719]*/, v34, v16
	s_cmp_le_i32 s6, s62
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199] /*v[704:711]*/, v[144:159], v[0:15] /*v[512:527]*/, v[192:199] /*v[704:711]*/, v34, v16 matrix_a_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191] /*v[696:703]*/, v[160:175], v[0:15] /*v[512:527]*/, v[184:191] /*v[696:703]*/, v35, v16
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183] /*v[688:695]*/, v[176:191], v[0:15] /*v[512:527]*/, v[176:183] /*v[688:695]*/, v35, v16 matrix_a_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175] /*v[680:687]*/, v[192:207], v[0:15] /*v[512:527]*/, v[168:175] /*v[680:687]*/, v32, v16
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167] /*v[672:679]*/, v[208:223], v[0:15] /*v[512:527]*/, v[160:167] /*v[672:679]*/, v32, v16 matrix_a_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159] /*v[664:671]*/, v[224:239], v[0:15] /*v[512:527]*/, v[152:159] /*v[664:671]*/, v33, v16
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151] /*v[656:663]*/, v[240:255], v[0:15] /*v[512:527]*/, v[144:151] /*v[656:663]*/, v33, v16 matrix_a_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa8c1                   ;  msbs: dst=3 src0=1 src1=0 src2=0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[58:65] /*v[826:833]*/, v[0:15] /*v[256:271]*/, v[64:79], 0, v17, v30
	v_wmma_scale_f32_16x16x128_f8f6f4 v[50:57] /*v[818:825]*/, v[32:47] /*v[288:303]*/, v[64:79], 0, v18, v30
	v_wmma_scale_f32_16x16x128_f8f6f4 v[42:49] /*v[810:817]*/, v[64:79] /*v[320:335]*/, v[64:79], 0, v19, v30
	v_wmma_scale_f32_16x16x128_f8f6f4 v[34:41] /*v[802:809]*/, v[96:111] /*v[352:367]*/, v[64:79], 0, v20, v30
	v_wmma_scale_f32_16x16x128_f8f6f4 v[26:33] /*v[794:801]*/, v[128:143] /*v[384:399]*/, v[64:79], 0, v21, v30
	v_wmma_scale_f32_16x16x128_f8f6f4 v[18:25] /*v[786:793]*/, v[160:175] /*v[416:431]*/, v[64:79], 0, v22, v30
	s_set_vgpr_msb 0xc181                   ;  msbs: dst=2 src0=1 src1=0 src2=0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15] /*v[520:527]*/, v[192:207] /*v[448:463]*/, v[64:79], 0, v23, v30
	v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7] /*v[512:519]*/, v[224:239] /*v[480:495]*/, v[64:79], 0, v24, v30
	s_set_vgpr_msb 0x81f1                   ;  msbs: dst=3 src0=1 src1=0 src2=3
	v_wmma_scale_f32_16x16x128_f8f6f4 v[58:65] /*v[826:833]*/, v[16:31] /*v[272:287]*/, v[80:95], v[58:65] /*v[826:833]*/, v17, v30 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[50:57] /*v[818:825]*/, v[48:63] /*v[304:319]*/, v[80:95], v[50:57] /*v[818:825]*/, v18, v30 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[42:49] /*v[810:817]*/, v[80:95] /*v[336:351]*/, v[80:95], v[42:49] /*v[810:817]*/, v19, v30 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[34:41] /*v[802:809]*/, v[112:127] /*v[368:383]*/, v[80:95], v[34:41] /*v[802:809]*/, v20, v30 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[26:33] /*v[794:801]*/, v[144:159] /*v[400:415]*/, v[80:95], v[26:33] /*v[794:801]*/, v21, v30 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[18:25] /*v[786:793]*/, v[176:191] /*v[432:447]*/, v[80:95], v[18:25] /*v[786:793]*/, v22, v30 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xf1a1                   ;  msbs: dst=2 src0=1 src1=0 src2=2
	v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15] /*v[520:527]*/, v[208:223] /*v[464:479]*/, v[80:95], v[8:15] /*v[520:527]*/, v23, v30 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7] /*v[512:519]*/, v[240:255] /*v[496:511]*/, v[80:95], v[0:7] /*v[512:519]*/, v24, v30 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_scc1 .LBB0_16
; %bb.15:                               ; %_ZZZZ12k_mla_fwd_mxENKUlT_E4_clISt17integral_constantIiLi2EEEEDaS_ENKUlS_E0_clIS2_IiLi0EEEEDaS_ENKUlS_E_clIS6_EEDaS_.exit.i.i.i.i.i.i.i.i
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_set_vgpr_msb 0xcf                     ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v247 /*v1015*/, v183 /*v951*/
	v_cndmask_b32_e32 v58 /*v826*/, 0xff800000, v58 /*v826*/, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v247 /*v1015*/, v183 /*v951*/
	v_cndmask_b32_e32 v59 /*v827*/, 0xff800000, v59 /*v827*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v246 /*v1014*/, v183 /*v951*/
	v_cndmask_b32_e32 v60 /*v828*/, 0xff800000, v60 /*v828*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v245 /*v1013*/, v183 /*v951*/
	v_cndmask_b32_e32 v61 /*v829*/, 0xff800000, v61 /*v829*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v244 /*v1012*/, v183 /*v951*/
	v_cndmask_b32_e32 v62 /*v830*/, 0xff800000, v62 /*v830*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v243 /*v1011*/, v183 /*v951*/
	v_cndmask_b32_e32 v63 /*v831*/, 0xff800000, v63 /*v831*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v241 /*v1009*/, v183 /*v951*/
	v_cndmask_b32_e32 v64 /*v832*/, 0xff800000, v64 /*v832*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v242 /*v1010*/, v183 /*v951*/
	v_cndmask_b32_e32 v65 /*v833*/, 0xff800000, v65 /*v833*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v240 /*v1008*/, v183 /*v951*/
	v_cndmask_b32_e32 v50 /*v818*/, 0xff800000, v50 /*v818*/, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v240 /*v1008*/, v183 /*v951*/
	v_cndmask_b32_e32 v51 /*v819*/, 0xff800000, v51 /*v819*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v239 /*v1007*/, v183 /*v951*/
	v_cndmask_b32_e32 v52 /*v820*/, 0xff800000, v52 /*v820*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v238 /*v1006*/, v183 /*v951*/
	v_cndmask_b32_e32 v53 /*v821*/, 0xff800000, v53 /*v821*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v237 /*v1005*/, v183 /*v951*/
	v_cndmask_b32_e32 v54 /*v822*/, 0xff800000, v54 /*v822*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v236 /*v1004*/, v183 /*v951*/
	v_cndmask_b32_e32 v55 /*v823*/, 0xff800000, v55 /*v823*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v234 /*v1002*/, v183 /*v951*/
	v_cndmask_b32_e32 v56 /*v824*/, 0xff800000, v56 /*v824*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v235 /*v1003*/, v183 /*v951*/
	v_cndmask_b32_e32 v57 /*v825*/, 0xff800000, v57 /*v825*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v233 /*v1001*/, v183 /*v951*/
	v_cndmask_b32_e32 v42 /*v810*/, 0xff800000, v42 /*v810*/, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v233 /*v1001*/, v183 /*v951*/
	v_cndmask_b32_e32 v43 /*v811*/, 0xff800000, v43 /*v811*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v232 /*v1000*/, v183 /*v951*/
	v_cndmask_b32_e32 v44 /*v812*/, 0xff800000, v44 /*v812*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v231 /*v999*/, v183 /*v951*/
	v_cndmask_b32_e32 v45 /*v813*/, 0xff800000, v45 /*v813*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v230 /*v998*/, v183 /*v951*/
	v_cndmask_b32_e32 v46 /*v814*/, 0xff800000, v46 /*v814*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v229 /*v997*/, v183 /*v951*/
	v_cndmask_b32_e32 v47 /*v815*/, 0xff800000, v47 /*v815*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v227 /*v995*/, v183 /*v951*/
	v_cndmask_b32_e32 v48 /*v816*/, 0xff800000, v48 /*v816*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v228 /*v996*/, v183 /*v951*/
	v_cndmask_b32_e32 v49 /*v817*/, 0xff800000, v49 /*v817*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v226 /*v994*/, v183 /*v951*/
	v_cndmask_b32_e32 v34 /*v802*/, 0xff800000, v34 /*v802*/, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v226 /*v994*/, v183 /*v951*/
	v_cndmask_b32_e32 v35 /*v803*/, 0xff800000, v35 /*v803*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v225 /*v993*/, v183 /*v951*/
	v_cndmask_b32_e32 v36 /*v804*/, 0xff800000, v36 /*v804*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v224 /*v992*/, v183 /*v951*/
	v_cndmask_b32_e32 v37 /*v805*/, 0xff800000, v37 /*v805*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v223 /*v991*/, v183 /*v951*/
	v_cndmask_b32_e32 v38 /*v806*/, 0xff800000, v38 /*v806*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v222 /*v990*/, v183 /*v951*/
	v_cndmask_b32_e32 v39 /*v807*/, 0xff800000, v39 /*v807*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v220 /*v988*/, v183 /*v951*/
	v_cndmask_b32_e32 v40 /*v808*/, 0xff800000, v40 /*v808*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v221 /*v989*/, v183 /*v951*/
	v_cndmask_b32_e32 v41 /*v809*/, 0xff800000, v41 /*v809*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v219 /*v987*/, v183 /*v951*/
	v_cndmask_b32_e32 v26 /*v794*/, 0xff800000, v26 /*v794*/, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v219 /*v987*/, v183 /*v951*/
	v_cndmask_b32_e32 v27 /*v795*/, 0xff800000, v27 /*v795*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v218 /*v986*/, v183 /*v951*/
	v_cndmask_b32_e32 v28 /*v796*/, 0xff800000, v28 /*v796*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v217 /*v985*/, v183 /*v951*/
	v_cndmask_b32_e32 v29 /*v797*/, 0xff800000, v29 /*v797*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v216 /*v984*/, v183 /*v951*/
	v_cndmask_b32_e32 v30 /*v798*/, 0xff800000, v30 /*v798*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v215 /*v983*/, v183 /*v951*/
	v_cndmask_b32_e32 v31 /*v799*/, 0xff800000, v31 /*v799*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v213 /*v981*/, v183 /*v951*/
	v_cndmask_b32_e32 v32 /*v800*/, 0xff800000, v32 /*v800*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v214 /*v982*/, v183 /*v951*/
	v_cndmask_b32_e32 v33 /*v801*/, 0xff800000, v33 /*v801*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v212 /*v980*/, v183 /*v951*/
	v_cndmask_b32_e32 v18 /*v786*/, 0xff800000, v18 /*v786*/, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v212 /*v980*/, v183 /*v951*/
	v_cndmask_b32_e32 v19 /*v787*/, 0xff800000, v19 /*v787*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v211 /*v979*/, v183 /*v951*/
	v_cndmask_b32_e32 v20 /*v788*/, 0xff800000, v20 /*v788*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v210 /*v978*/, v183 /*v951*/
	v_cndmask_b32_e32 v21 /*v789*/, 0xff800000, v21 /*v789*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v209 /*v977*/, v183 /*v951*/
	v_cndmask_b32_e32 v22 /*v790*/, 0xff800000, v22 /*v790*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v208 /*v976*/, v183 /*v951*/
	v_cndmask_b32_e32 v23 /*v791*/, 0xff800000, v23 /*v791*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v206 /*v974*/, v183 /*v951*/
	v_cndmask_b32_e32 v24 /*v792*/, 0xff800000, v24 /*v792*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v207 /*v975*/, v183 /*v951*/
	v_cndmask_b32_e32 v25 /*v793*/, 0xff800000, v25 /*v793*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v205 /*v973*/, v183 /*v951*/
	s_set_vgpr_msb 0xcf88                   ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v8 /*v520*/, 0xff800000, v8 /*v520*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_lt_i32_e32 vcc_lo, v205 /*v973*/, v183 /*v951*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v9 /*v521*/, 0xff800000, v9 /*v521*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v204 /*v972*/, v183 /*v951*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v10 /*v522*/, 0xff800000, v10 /*v522*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v203 /*v971*/, v183 /*v951*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v11 /*v523*/, 0xff800000, v11 /*v523*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v202 /*v970*/, v183 /*v951*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v12 /*v524*/, 0xff800000, v12 /*v524*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v201 /*v969*/, v183 /*v951*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v13 /*v525*/, 0xff800000, v13 /*v525*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v199 /*v967*/, v183 /*v951*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v14 /*v526*/, 0xff800000, v14 /*v526*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v200 /*v968*/, v183 /*v951*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v15 /*v527*/, 0xff800000, v15 /*v527*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v198 /*v966*/, v183 /*v951*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v0 /*v512*/, 0xff800000, v0 /*v512*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_lt_i32_e32 vcc_lo, v198 /*v966*/, v183 /*v951*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v1 /*v513*/, 0xff800000, v1 /*v513*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v197 /*v965*/, v183 /*v951*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v2 /*v514*/, 0xff800000, v2 /*v514*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v196 /*v964*/, v183 /*v951*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v3 /*v515*/, 0xff800000, v3 /*v515*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v195 /*v963*/, v183 /*v951*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v4 /*v516*/, 0xff800000, v4 /*v516*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v194 /*v962*/, v183 /*v951*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v5 /*v517*/, 0xff800000, v5 /*v517*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v193 /*v961*/, v183 /*v951*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v6 /*v518*/, 0xff800000, v6 /*v518*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v192 /*v960*/, v183 /*v951*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v7 /*v519*/, 0xff800000, v7 /*v519*/, vcc_lo
	s_set_vgpr_msb 0x8800                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_16:                               ;   in Loop: Header=BB0_6 Depth=1
	s_set_vgpr_msb 0xff                     ;  msbs: dst=3 src0=3 src1=3 src2=3
	v_max3_num_f32 v66 /*v834*/, v58 /*v826*/, 0xff800000, v59 /*v827*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v60 /*v828*/, v61 /*v829*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v62 /*v830*/, v63 /*v831*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v64 /*v832*/, v65 /*v833*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v50 /*v818*/, v51 /*v819*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v52 /*v820*/, v53 /*v821*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v54 /*v822*/, v55 /*v823*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v56 /*v824*/, v57 /*v825*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v42 /*v810*/, v43 /*v811*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v44 /*v812*/, v45 /*v813*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v46 /*v814*/, v47 /*v815*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v48 /*v816*/, v49 /*v817*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v34 /*v802*/, v35 /*v803*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v36 /*v804*/, v37 /*v805*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v38 /*v806*/, v39 /*v807*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v40 /*v808*/, v41 /*v809*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v26 /*v794*/, v27 /*v795*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v28 /*v796*/, v29 /*v797*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v30 /*v798*/, v31 /*v799*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v32 /*v800*/, v33 /*v801*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v18 /*v786*/, v19 /*v787*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v20 /*v788*/, v21 /*v789*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v22 /*v790*/, v23 /*v791*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v24 /*v792*/, v25 /*v793*/
	s_set_vgpr_msb 0xffeb                   ;  msbs: dst=3 src0=3 src1=2 src2=2
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v8 /*v520*/, v9 /*v521*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v10 /*v522*/, v11 /*v523*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v12 /*v524*/, v13 /*v525*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v14 /*v526*/, v15 /*v527*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v0 /*v512*/, v1 /*v513*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v2 /*v514*/, v3 /*v515*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v4 /*v516*/, v5 /*v517*/
	v_max3_num_f32 v66 /*v834*/, v66 /*v834*/, v6 /*v518*/, v7 /*v519*/
	v_mov_b32_e32 v67 /*v835*/, v66 /*v834*/
	v_permlanex16_b32 v67 /*v835*/, v67 /*v835*/, s65, 0xfedcba98
	s_set_vgpr_msb 0xebcf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_max_num_f32_e32 v66 /*v834*/, v66 /*v834*/, v67 /*v835*/
	v_fma_f32 v67 /*v835*/, v66 /*v834*/, s60, -v40
	v_mul_f32_e32 v66 /*v834*/, s60, v66 /*v834*/
	v_cmp_lt_f32_e32 vcc_lo, 0x3fb8aa3b, v67 /*v835*/
	s_set_vgpr_msb 0xcfcc                   ;  msbs: dst=3 src0=0 src1=3 src2=0
	v_max_num_f32_e32 v66 /*v834*/, v40, v66 /*v834*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0xccf3                   ;  msbs: dst=3 src0=3 src1=0 src2=3
	v_cndmask_b32_e64 v248 /*v1016*/, v66 /*v834*/, v40, s2
	v_fma_f32 v58 /*v826*/, v58 /*v826*/, s60, -v248 /*v1016*/
	v_fma_f32 v59 /*v827*/, v59 /*v827*/, s60, -v248 /*v1016*/
	v_fma_f32 v60 /*v828*/, v60 /*v828*/, s60, -v248 /*v1016*/
	v_fma_f32 v61 /*v829*/, v61 /*v829*/, s60, -v248 /*v1016*/
	v_fma_f32 v62 /*v830*/, v62 /*v830*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v58 /*v826*/, v58 /*v826*/
	v_fma_f32 v63 /*v831*/, v63 /*v831*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v59 /*v827*/, v59 /*v827*/
	v_fma_f32 v64 /*v832*/, v64 /*v832*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v60 /*v828*/, v60 /*v828*/
	v_fma_f32 v65 /*v833*/, v65 /*v833*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v61 /*v829*/, v61 /*v829*/
	v_fma_f32 v50 /*v818*/, v50 /*v818*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v62 /*v830*/, v62 /*v830*/
	v_fma_f32 v51 /*v819*/, v51 /*v819*/, s60, -v248 /*v1016*/
	s_set_vgpr_msb 0xf3ff                   ;  msbs: dst=3 src0=3 src1=3 src2=3
	v_add_f32_e32 v66 /*v834*/, v58 /*v826*/, v59 /*v827*/
	v_exp_f32_e32 v63 /*v831*/, v63 /*v831*/
	v_fma_f32 v52 /*v820*/, v52 /*v820*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v64 /*v832*/, v64 /*v832*/
	v_fma_f32 v53 /*v821*/, v53 /*v821*/, s60, -v248 /*v1016*/
	v_add_f32_e32 v66 /*v834*/, v60 /*v828*/, v66 /*v834*/
	v_exp_f32_e32 v65 /*v833*/, v65 /*v833*/
	v_fma_f32 v54 /*v822*/, v54 /*v822*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v50 /*v818*/, v50 /*v818*/
	v_fma_f32 v55 /*v823*/, v55 /*v823*/, s60, -v248 /*v1016*/
	v_add_f32_e32 v66 /*v834*/, v61 /*v829*/, v66 /*v834*/
	v_exp_f32_e32 v51 /*v819*/, v51 /*v819*/
	v_fma_f32 v56 /*v824*/, v56 /*v824*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v52 /*v820*/, v52 /*v820*/
	v_fma_f32 v57 /*v825*/, v57 /*v825*/, s60, -v248 /*v1016*/
	v_add_f32_e32 v66 /*v834*/, v62 /*v830*/, v66 /*v834*/
	v_exp_f32_e32 v53 /*v821*/, v53 /*v821*/
	v_fma_f32 v42 /*v810*/, v42 /*v810*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v54 /*v822*/, v54 /*v822*/
	v_fma_f32 v43 /*v811*/, v43 /*v811*/, s60, -v248 /*v1016*/
	v_add_f32_e32 v66 /*v834*/, v63 /*v831*/, v66 /*v834*/
	v_exp_f32_e32 v55 /*v823*/, v55 /*v823*/
	v_fma_f32 v44 /*v812*/, v44 /*v812*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v56 /*v824*/, v56 /*v824*/
	v_fma_f32 v45 /*v813*/, v45 /*v813*/, s60, -v248 /*v1016*/
	v_add_f32_e32 v66 /*v834*/, v64 /*v832*/, v66 /*v834*/
	v_exp_f32_e32 v57 /*v825*/, v57 /*v825*/
	v_fma_f32 v46 /*v814*/, v46 /*v814*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v42 /*v810*/, v42 /*v810*/
	v_fma_f32 v47 /*v815*/, v47 /*v815*/, s60, -v248 /*v1016*/
	v_add_f32_e32 v66 /*v834*/, v65 /*v833*/, v66 /*v834*/
	v_exp_f32_e32 v43 /*v811*/, v43 /*v811*/
	v_fma_f32 v48 /*v816*/, v48 /*v816*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v44 /*v812*/, v44 /*v812*/
	v_fma_f32 v49 /*v817*/, v49 /*v817*/, s60, -v248 /*v1016*/
	v_add_f32_e32 v66 /*v834*/, v50 /*v818*/, v66 /*v834*/
	v_exp_f32_e32 v45 /*v813*/, v45 /*v813*/
	v_fma_f32 v34 /*v802*/, v34 /*v802*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v46 /*v814*/, v46 /*v814*/
	v_fma_f32 v35 /*v803*/, v35 /*v803*/, s60, -v248 /*v1016*/
	v_add_f32_e32 v66 /*v834*/, v51 /*v819*/, v66 /*v834*/
	v_exp_f32_e32 v47 /*v815*/, v47 /*v815*/
	v_fma_f32 v36 /*v804*/, v36 /*v804*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v48 /*v816*/, v48 /*v816*/
	v_fma_f32 v37 /*v805*/, v37 /*v805*/, s60, -v248 /*v1016*/
	v_add_f32_e32 v66 /*v834*/, v52 /*v820*/, v66 /*v834*/
	v_exp_f32_e32 v49 /*v817*/, v49 /*v817*/
	v_fma_f32 v38 /*v806*/, v38 /*v806*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v34 /*v802*/, v34 /*v802*/
	v_fma_f32 v39 /*v807*/, v39 /*v807*/, s60, -v248 /*v1016*/
	v_add_f32_e32 v66 /*v834*/, v53 /*v821*/, v66 /*v834*/
	v_exp_f32_e32 v35 /*v803*/, v35 /*v803*/
	v_fma_f32 v40 /*v808*/, v40 /*v808*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v36 /*v804*/, v36 /*v804*/
	v_fma_f32 v41 /*v809*/, v41 /*v809*/, s60, -v248 /*v1016*/
	v_add_f32_e32 v66 /*v834*/, v54 /*v822*/, v66 /*v834*/
	v_exp_f32_e32 v37 /*v805*/, v37 /*v805*/
	v_fma_f32 v26 /*v794*/, v26 /*v794*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v38 /*v806*/, v38 /*v806*/
	v_fma_f32 v27 /*v795*/, v27 /*v795*/, s60, -v248 /*v1016*/
	v_add_f32_e32 v66 /*v834*/, v55 /*v823*/, v66 /*v834*/
	v_exp_f32_e32 v39 /*v807*/, v39 /*v807*/
	v_fma_f32 v28 /*v796*/, v28 /*v796*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v40 /*v808*/, v40 /*v808*/
	v_fma_f32 v29 /*v797*/, v29 /*v797*/, s60, -v248 /*v1016*/
	v_add_f32_e32 v66 /*v834*/, v56 /*v824*/, v66 /*v834*/
	v_exp_f32_e32 v41 /*v809*/, v41 /*v809*/
	v_fma_f32 v30 /*v798*/, v30 /*v798*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v26 /*v794*/, v26 /*v794*/
	v_fma_f32 v31 /*v799*/, v31 /*v799*/, s60, -v248 /*v1016*/
	v_add_f32_e32 v66 /*v834*/, v57 /*v825*/, v66 /*v834*/
	v_exp_f32_e32 v27 /*v795*/, v27 /*v795*/
	v_fma_f32 v32 /*v800*/, v32 /*v800*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v28 /*v796*/, v28 /*v796*/
	v_fma_f32 v33 /*v801*/, v33 /*v801*/, s60, -v248 /*v1016*/
	v_add_f32_e32 v66 /*v834*/, v42 /*v810*/, v66 /*v834*/
	v_exp_f32_e32 v29 /*v797*/, v29 /*v797*/
	v_fma_f32 v18 /*v786*/, v18 /*v786*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v30 /*v798*/, v30 /*v798*/
	v_fma_f32 v19 /*v787*/, v19 /*v787*/, s60, -v248 /*v1016*/
	v_add_f32_e32 v66 /*v834*/, v43 /*v811*/, v66 /*v834*/
	v_exp_f32_e32 v31 /*v799*/, v31 /*v799*/
	v_fma_f32 v20 /*v788*/, v20 /*v788*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v32 /*v800*/, v32 /*v800*/
	v_fma_f32 v21 /*v789*/, v21 /*v789*/, s60, -v248 /*v1016*/
	v_add_f32_e32 v66 /*v834*/, v44 /*v812*/, v66 /*v834*/
	v_exp_f32_e32 v33 /*v801*/, v33 /*v801*/
	v_fma_f32 v22 /*v790*/, v22 /*v790*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v18 /*v786*/, v18 /*v786*/
	v_fma_f32 v23 /*v791*/, v23 /*v791*/, s60, -v248 /*v1016*/
	v_add_f32_e32 v66 /*v834*/, v45 /*v813*/, v66 /*v834*/
	v_exp_f32_e32 v19 /*v787*/, v19 /*v787*/
	v_fma_f32 v24 /*v792*/, v24 /*v792*/, s60, -v248 /*v1016*/
	v_exp_f32_e32 v20 /*v788*/, v20 /*v788*/
	v_fma_f32 v25 /*v793*/, v25 /*v793*/, s60, -v248 /*v1016*/
	v_add_f32_e32 v66 /*v834*/, v46 /*v814*/, v66 /*v834*/
	v_exp_f32_e32 v21 /*v789*/, v21 /*v789*/
	s_set_vgpr_msb 0xffb2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v8 /*v520*/, v8 /*v520*/, s60, -v248 /*v1016*/
	s_set_vgpr_msb 0xb2c3                   ;  msbs: dst=3 src0=3 src1=0 src2=0
	v_exp_f32_e32 v22 /*v790*/, v22 /*v790*/
	s_set_vgpr_msb 0xc3b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v0 /*v512*/, v0 /*v512*/, s60, -v248 /*v1016*/
	s_set_vgpr_msb 0xb2cf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v47 /*v815*/, v66 /*v834*/
	v_exp_f32_e32 v23 /*v791*/, v23 /*v791*/
	s_set_vgpr_msb 0xcfb2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v10 /*v522*/, v10 /*v522*/, s60, -v248 /*v1016*/
	s_set_vgpr_msb 0xb2c3                   ;  msbs: dst=3 src0=3 src1=0 src2=0
	v_exp_f32_e32 v24 /*v792*/, v24 /*v792*/
	s_set_vgpr_msb 0xc3b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v2 /*v514*/, v2 /*v514*/, s60, -v248 /*v1016*/
	s_set_vgpr_msb 0xb2cf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v48 /*v816*/, v66 /*v834*/
	v_exp_f32_e32 v25 /*v793*/, v25 /*v793*/
	s_set_vgpr_msb 0xcf0c                   ;  msbs: dst=0 src0=0 src1=3 src2=0
	v_sub_f32_e32 v40, v40, v248 /*v1016*/
	s_set_vgpr_msb 0xcc2                    ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v74 /*v842*/, v0 /*v512*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v0 /*v512*/, v1 /*v513*/, s60, -v248 /*v1016*/
	s_set_vgpr_msb 0xb2cf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v49 /*v817*/, v66 /*v834*/
	s_set_vgpr_msb 0xcfc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v68 /*v836*/, v10 /*v522*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v10 /*v522*/, v12 /*v524*/, s60, -v248 /*v1016*/
	s_set_vgpr_msb 0xb2c2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v76 /*v844*/, v2 /*v514*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v2 /*v514*/, v4 /*v516*/, s60, -v248 /*v1016*/
	s_set_vgpr_msb 0xb2cf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v34 /*v802*/, v66 /*v834*/
	s_set_vgpr_msb 0xcfc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v75 /*v843*/, v0 /*v512*/
	s_set_vgpr_msb 0xc2cf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v35 /*v803*/, v66 /*v834*/
	s_set_vgpr_msb 0xcfc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v70 /*v838*/, v10 /*v522*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v10 /*v522*/, v14 /*v526*/, s60, -v248 /*v1016*/
	s_set_vgpr_msb 0xb2c2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v78 /*v846*/, v2 /*v514*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v2 /*v514*/, v6 /*v518*/, s60, -v248 /*v1016*/
	s_set_vgpr_msb 0xb2cf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v36 /*v804*/, v66 /*v834*/
	s_set_vgpr_msb 0xcf00                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_exp_f32_e32 v40, v40
	s_set_vgpr_msb 0xcf                     ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v37 /*v805*/, v66 /*v834*/
	s_set_vgpr_msb 0xcfc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v72 /*v840*/, v10 /*v522*/
	s_set_vgpr_msb 0xc2cf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v38 /*v806*/, v66 /*v834*/
	s_set_vgpr_msb 0xcfc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v80 /*v848*/, v2 /*v514*/
	s_set_vgpr_msb 0xc2cf                   ;  msbs: dst=3 src0=3 src1=3 src2=0
	v_add_f32_e32 v66 /*v834*/, v39 /*v807*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v40 /*v808*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v41 /*v809*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v26 /*v794*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v27 /*v795*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v28 /*v796*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v29 /*v797*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v30 /*v798*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v31 /*v799*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v32 /*v800*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v33 /*v801*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v18 /*v786*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v19 /*v787*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v20 /*v788*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v21 /*v789*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v22 /*v790*/, v66 /*v834*/
	v_add_f32_e32 v66 /*v834*/, v23 /*v791*/, v66 /*v834*/
	v_add_f32_e32 v67 /*v835*/, v24 /*v792*/, v66 /*v834*/
	s_set_vgpr_msb 0xcfc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v66 /*v834*/, v8 /*v520*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v8 /*v520*/, v9 /*v521*/, s60, -v248 /*v1016*/
	s_set_vgpr_msb 0xb28f                   ;  msbs: dst=2 src0=3 src1=3 src2=0
	v_add_f32_e32 v9 /*v521*/, v25 /*v793*/, v67 /*v835*/
	s_set_vgpr_msb 0x8fc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v67 /*v835*/, v8 /*v520*/
	v_nop
	s_set_vgpr_msb 0xc28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v8 /*v520*/, v66 /*v834*/, v9 /*v521*/
	s_set_vgpr_msb 0x8bb2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v9 /*v521*/, v11 /*v523*/, s60, -v248 /*v1016*/
	s_set_vgpr_msb 0xb28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v8 /*v520*/, v67 /*v835*/, v8 /*v520*/
	s_set_vgpr_msb 0x8bc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v69 /*v837*/, v9 /*v521*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v9 /*v521*/, v13 /*v525*/, s60, -v248 /*v1016*/
	s_set_vgpr_msb 0xb28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v8 /*v520*/, v68 /*v836*/, v8 /*v520*/
	s_set_vgpr_msb 0x8bc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v71 /*v839*/, v9 /*v521*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v9 /*v521*/, v15 /*v527*/, s60, -v248 /*v1016*/
	s_set_vgpr_msb 0xb28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v8 /*v520*/, v69 /*v837*/, v8 /*v520*/
	s_set_vgpr_msb 0x8bc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v73 /*v841*/, v9 /*v521*/
	s_set_vgpr_msb 0xc28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v8 /*v520*/, v70 /*v838*/, v8 /*v520*/
	v_add_f32_e32 v8 /*v520*/, v71 /*v839*/, v8 /*v520*/
	v_add_f32_e32 v8 /*v520*/, v72 /*v840*/, v8 /*v520*/
	v_add_f32_e32 v1 /*v513*/, v73 /*v841*/, v8 /*v520*/
	v_add_f32_e32 v0 /*v512*/, v74 /*v842*/, v1 /*v513*/
	s_set_vgpr_msb 0x8bb2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v1 /*v513*/, v3 /*v515*/, s60, -v248 /*v1016*/
	s_set_vgpr_msb 0xb28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v0 /*v512*/, v75 /*v843*/, v0 /*v512*/
	s_set_vgpr_msb 0x8bc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v77 /*v845*/, v1 /*v513*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v1 /*v513*/, v5 /*v517*/, s60, -v248 /*v1016*/
	s_set_vgpr_msb 0xb28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v0 /*v512*/, v76 /*v844*/, v0 /*v512*/
	s_set_vgpr_msb 0x8bc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v79 /*v847*/, v1 /*v513*/
	v_nop
	s_set_vgpr_msb 0xc2b2                   ;  msbs: dst=2 src0=2 src1=0 src2=3
	v_fma_f32 v1 /*v513*/, v7 /*v519*/, s60, -v248 /*v1016*/
	s_set_vgpr_msb 0xb28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v0 /*v512*/, v77 /*v845*/, v0 /*v512*/
	s_set_vgpr_msb 0x8bc2                   ;  msbs: dst=3 src0=2 src1=0 src2=0
	v_exp_f32_e32 v81 /*v849*/, v1 /*v513*/
	s_set_vgpr_msb 0xc28b                   ;  msbs: dst=2 src0=3 src1=2 src2=0
	v_add_f32_e32 v0 /*v512*/, v78 /*v846*/, v0 /*v512*/
	v_add_f32_e32 v0 /*v512*/, v79 /*v847*/, v0 /*v512*/
	v_add_f32_e32 v0 /*v512*/, v80 /*v848*/, v0 /*v512*/
	s_set_vgpr_msb 0x8bcb                   ;  msbs: dst=3 src0=3 src1=2 src2=0
	v_add_f32_e32 v251 /*v1019*/, v81 /*v849*/, v0 /*v512*/
	v_mov_b32_e32 v252 /*v1020*/, v251 /*v1019*/
	v_permlanex16_b32 v252 /*v1020*/, v252 /*v1020*/, s65, 0xfedcba98
	s_set_vgpr_msb 0xcb00                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_vccz .LBB0_18
; %bb.17:                               ;   in Loop: Header=BB0_6 Depth=1
	s_set_vgpr_msb 0x82                     ;  msbs: dst=2 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[142:143] /*v[654:655]*/, v[142:143] /*v[654:655]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[140:141] /*v[652:653]*/, v[140:141] /*v[652:653]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[138:139] /*v[650:651]*/, v[138:139] /*v[650:651]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[136:137] /*v[648:649]*/, v[136:137] /*v[648:649]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[134:135] /*v[646:647]*/, v[134:135] /*v[646:647]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[132:133] /*v[644:645]*/, v[132:133] /*v[644:645]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[130:131] /*v[642:643]*/, v[130:131] /*v[642:643]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[128:129] /*v[640:641]*/, v[128:129] /*v[640:641]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127] /*v[638:639]*/, v[126:127] /*v[638:639]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125] /*v[636:637]*/, v[124:125] /*v[636:637]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123] /*v[634:635]*/, v[122:123] /*v[634:635]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121] /*v[632:633]*/, v[120:121] /*v[632:633]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119] /*v[630:631]*/, v[118:119] /*v[630:631]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117] /*v[628:629]*/, v[116:117] /*v[628:629]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115] /*v[626:627]*/, v[114:115] /*v[626:627]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113] /*v[624:625]*/, v[112:113] /*v[624:625]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111] /*v[622:623]*/, v[110:111] /*v[622:623]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109] /*v[620:621]*/, v[108:109] /*v[620:621]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107] /*v[618:619]*/, v[106:107] /*v[618:619]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105] /*v[616:617]*/, v[104:105] /*v[616:617]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103] /*v[614:615]*/, v[102:103] /*v[614:615]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101] /*v[612:613]*/, v[100:101] /*v[612:613]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99] /*v[610:611]*/, v[98:99] /*v[610:611]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97] /*v[608:609]*/, v[96:97] /*v[608:609]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95] /*v[606:607]*/, v[94:95] /*v[606:607]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93] /*v[604:605]*/, v[92:93] /*v[604:605]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91] /*v[602:603]*/, v[90:91] /*v[602:603]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89] /*v[600:601]*/, v[88:89] /*v[600:601]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87] /*v[598:599]*/, v[86:87] /*v[598:599]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85] /*v[596:597]*/, v[84:85] /*v[596:597]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83] /*v[594:595]*/, v[82:83] /*v[594:595]*/, v[40:41] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81] /*v[592:593]*/, v[80:81] /*v[592:593]*/, v[40:41] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_18:                               ; %_ZZ12k_mla_fwd_mxENKUlT_E4_clISt17integral_constantIiLi2EEEEDaS_.exit.i.i
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_set_vgpr_msb 0x83                     ;  msbs: dst=2 src0=3 src1=0 src2=0
	v_cvt_scalef32_pk8_fp8_f32 v[0:1] /*v[512:513]*/, v[58:65] /*v[826:833]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[2:3] /*v[514:515]*/, v[50:57] /*v[818:825]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[4:5] /*v[516:517]*/, v[42:49] /*v[810:817]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[6:7] /*v[518:519]*/, v[34:41] /*v[802:809]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[8:9] /*v[520:521]*/, v[26:33] /*v[794:801]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[10:11] /*v[522:523]*/, v[18:25] /*v[786:793]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[12:13] /*v[524:525]*/, v[66:73] /*v[834:841]*/, 0x3c000000
	v_cvt_scalef32_pk8_fp8_f32 v[14:15] /*v[526:527]*/, v[74:81] /*v[842:849]*/, 0x3c000000
	s_set_vgpr_msb 0x83a8                   ;  msbs: dst=2 src0=0 src1=2 src2=2
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143] /*v[648:655]*/, v[128:143], v[0:15] /*v[512:527]*/, v[136:143] /*v[648:655]*/, v34, v16
	s_cmp_le_i32 s6, s63
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135] /*v[640:647]*/, v[144:159], v[0:15] /*v[512:527]*/, v[128:135] /*v[640:647]*/, v34, v16 matrix_a_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127] /*v[632:639]*/, v[160:175], v[0:15] /*v[512:527]*/, v[120:127] /*v[632:639]*/, v35, v16
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119] /*v[624:631]*/, v[176:191], v[0:15] /*v[512:527]*/, v[112:119] /*v[624:631]*/, v35, v16 matrix_a_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111] /*v[616:623]*/, v[192:207], v[0:15] /*v[512:527]*/, v[104:111] /*v[616:623]*/, v32, v16
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103] /*v[608:615]*/, v[208:223], v[0:15] /*v[512:527]*/, v[96:103] /*v[608:615]*/, v32, v16 matrix_a_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95] /*v[600:607]*/, v[224:239], v[0:15] /*v[512:527]*/, v[88:95] /*v[600:607]*/, v33, v16
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87] /*v[592:599]*/, v[240:255], v[0:15] /*v[512:527]*/, v[80:87] /*v[592:599]*/, v33, v16 matrix_a_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa8a1                   ;  msbs: dst=2 src0=1 src1=0 src2=2
	v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15] /*v[520:527]*/, v[0:15] /*v[256:271]*/, v[96:111], 0, v17, v31
	v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15] /*v[520:527]*/, v[16:31] /*v[272:287]*/, v[112:127], v[8:15] /*v[520:527]*/, v17, v31 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7] /*v[512:519]*/, v[32:47] /*v[288:303]*/, v[96:111], 0, v18, v31
	s_set_vgpr_msb 0xa141                   ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47] /*v[296:303]*/, v[64:79] /*v[320:335]*/, v[96:111], 0, v19, v31
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39] /*v[288:295]*/, v[96:111] /*v[352:367]*/, v[96:111], 0, v20, v31
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31] /*v[280:287]*/, v[128:143] /*v[384:399]*/, v[96:111], 0, v21, v31
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23] /*v[272:279]*/, v[160:175] /*v[416:431]*/, v[96:111], 0, v22, v31
	v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15] /*v[264:271]*/, v[192:207] /*v[448:463]*/, v[96:111], 0, v23, v31
	v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7] /*v[256:263]*/, v[224:239] /*v[480:495]*/, v[96:111], 0, v24, v31
	s_set_vgpr_msb 0x41a1                   ;  msbs: dst=2 src0=1 src1=0 src2=2
	v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7] /*v[512:519]*/, v[48:63] /*v[304:319]*/, v[112:127], v[0:7] /*v[512:519]*/, v18, v31 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa151                   ;  msbs: dst=1 src0=1 src1=0 src2=1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47] /*v[296:303]*/, v[80:95] /*v[336:351]*/, v[112:127], v[40:47] /*v[296:303]*/, v19, v31 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39] /*v[288:295]*/, v[112:127] /*v[368:383]*/, v[112:127], v[32:39] /*v[288:295]*/, v20, v31 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31] /*v[280:287]*/, v[144:159] /*v[400:415]*/, v[112:127], v[24:31] /*v[280:287]*/, v21, v31 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23] /*v[272:279]*/, v[176:191] /*v[432:447]*/, v[112:127], v[16:23] /*v[272:279]*/, v22, v31 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15] /*v[264:271]*/, v[208:223] /*v[464:479]*/, v[112:127], v[8:15] /*v[264:271]*/, v23, v31 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7] /*v[256:263]*/, v[240:255] /*v[496:511]*/, v[112:127], v[0:7] /*v[256:263]*/, v24, v31 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_scc1 .LBB0_20
; %bb.19:                               ; %_ZZZZ12k_mla_fwd_mxENKUlT_E4_clISt17integral_constantIiLi3EEEEDaS_ENKUlS_E0_clIS2_IiLi0EEEEDaS_ENKUlS_E_clIS6_EEDaS_.exit.i.i.i.i.i.i.i.i
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_set_vgpr_msb 15                       ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v247 /*v1015*/, v184 /*v952*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v8 /*v520*/, 0xff800000, v8 /*v520*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_lt_i32_e32 vcc_lo, v247 /*v1015*/, v184 /*v952*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v9 /*v521*/, 0xff800000, v9 /*v521*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v246 /*v1014*/, v184 /*v952*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v10 /*v522*/, 0xff800000, v10 /*v522*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v245 /*v1013*/, v184 /*v952*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v11 /*v523*/, 0xff800000, v11 /*v523*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v244 /*v1012*/, v184 /*v952*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v12 /*v524*/, 0xff800000, v12 /*v524*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v243 /*v1011*/, v184 /*v952*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v13 /*v525*/, 0xff800000, v13 /*v525*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v241 /*v1009*/, v184 /*v952*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v14 /*v526*/, 0xff800000, v14 /*v526*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v242 /*v1010*/, v184 /*v952*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v15 /*v527*/, 0xff800000, v15 /*v527*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v240 /*v1008*/, v184 /*v952*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v0 /*v512*/, 0xff800000, v0 /*v512*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_lt_i32_e32 vcc_lo, v240 /*v1008*/, v184 /*v952*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v1 /*v513*/, 0xff800000, v1 /*v513*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v239 /*v1007*/, v184 /*v952*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v2 /*v514*/, 0xff800000, v2 /*v514*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v238 /*v1006*/, v184 /*v952*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v3 /*v515*/, 0xff800000, v3 /*v515*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v237 /*v1005*/, v184 /*v952*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v4 /*v516*/, 0xff800000, v4 /*v516*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v236 /*v1004*/, v184 /*v952*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v5 /*v517*/, 0xff800000, v5 /*v517*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v234 /*v1002*/, v184 /*v952*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v6 /*v518*/, 0xff800000, v6 /*v518*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v235 /*v1003*/, v184 /*v952*/
	s_set_vgpr_msb 0xf88                    ;  msbs: dst=2 src0=0 src1=2 src2=0
	v_cndmask_b32_e32 v7 /*v519*/, 0xff800000, v7 /*v519*/, vcc_lo
	s_set_vgpr_msb 0x880f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v233 /*v1001*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v40 /*v296*/, 0xff800000, v40 /*v296*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_lt_i32_e32 vcc_lo, v233 /*v1001*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v41 /*v297*/, 0xff800000, v41 /*v297*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v232 /*v1000*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v42 /*v298*/, 0xff800000, v42 /*v298*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v231 /*v999*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v43 /*v299*/, 0xff800000, v43 /*v299*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v230 /*v998*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v44 /*v300*/, 0xff800000, v44 /*v300*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v229 /*v997*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v45 /*v301*/, 0xff800000, v45 /*v301*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v227 /*v995*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v46 /*v302*/, 0xff800000, v46 /*v302*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v228 /*v996*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v47 /*v303*/, 0xff800000, v47 /*v303*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v226 /*v994*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v32 /*v288*/, 0xff800000, v32 /*v288*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_lt_i32_e32 vcc_lo, v226 /*v994*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v33 /*v289*/, 0xff800000, v33 /*v289*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v225 /*v993*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v34 /*v290*/, 0xff800000, v34 /*v290*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v224 /*v992*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v35 /*v291*/, 0xff800000, v35 /*v291*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v223 /*v991*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v36 /*v292*/, 0xff800000, v36 /*v292*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v222 /*v990*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v37 /*v293*/, 0xff800000, v37 /*v293*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v220 /*v988*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v38 /*v294*/, 0xff800000, v38 /*v294*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v221 /*v989*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v39 /*v295*/, 0xff800000, v39 /*v295*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v219 /*v987*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v24 /*v280*/, 0xff800000, v24 /*v280*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_lt_i32_e32 vcc_lo, v219 /*v987*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v25 /*v281*/, 0xff800000, v25 /*v281*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v218 /*v986*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v26 /*v282*/, 0xff800000, v26 /*v282*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v217 /*v985*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v27 /*v283*/, 0xff800000, v27 /*v283*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v216 /*v984*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v28 /*v284*/, 0xff800000, v28 /*v284*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v215 /*v983*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v29 /*v285*/, 0xff800000, v29 /*v285*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v213 /*v981*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v30 /*v286*/, 0xff800000, v30 /*v286*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v214 /*v982*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v31 /*v287*/, 0xff800000, v31 /*v287*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v212 /*v980*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v16 /*v272*/, 0xff800000, v16 /*v272*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_lt_i32_e32 vcc_lo, v212 /*v980*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v17 /*v273*/, 0xff800000, v17 /*v273*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v211 /*v979*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v18 /*v274*/, 0xff800000, v18 /*v274*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v210 /*v978*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v19 /*v275*/, 0xff800000, v19 /*v275*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v209 /*v977*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v20 /*v276*/, 0xff800000, v20 /*v276*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v208 /*v976*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v21 /*v277*/, 0xff800000, v21 /*v277*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v206 /*v974*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v22 /*v278*/, 0xff800000, v22 /*v278*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v207 /*v975*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v23 /*v279*/, 0xff800000, v23 /*v279*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v205 /*v973*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v8 /*v264*/, 0xff800000, v8 /*v264*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_lt_i32_e32 vcc_lo, v205 /*v973*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v9 /*v265*/, 0xff800000, v9 /*v265*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v204 /*v972*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v10 /*v266*/, 0xff800000, v10 /*v266*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v203 /*v971*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v11 /*v267*/, 0xff800000, v11 /*v267*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v202 /*v970*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v12 /*v268*/, 0xff800000, v12 /*v268*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v201 /*v969*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v13 /*v269*/, 0xff800000, v13 /*v269*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v199 /*v967*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v14 /*v270*/, 0xff800000, v14 /*v270*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v200 /*v968*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v15 /*v271*/, 0xff800000, v15 /*v271*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v198 /*v966*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v0 /*v256*/, 0xff800000, v0 /*v256*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_lt_i32_e32 vcc_lo, v198 /*v966*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v1 /*v257*/, 0xff800000, v1 /*v257*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v197 /*v965*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v2 /*v258*/, 0xff800000, v2 /*v258*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v196 /*v964*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v3 /*v259*/, 0xff800000, v3 /*v259*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v195 /*v963*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v4 /*v260*/, 0xff800000, v4 /*v260*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v194 /*v962*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v5 /*v261*/, 0xff800000, v5 /*v261*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v193 /*v961*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v6 /*v262*/, 0xff800000, v6 /*v262*/, vcc_lo
	s_set_vgpr_msb 0x440f                   ;  msbs: dst=0 src0=3 src1=3 src2=0
	v_cmp_le_i32_e32 vcc_lo, v192 /*v960*/, v184 /*v952*/
	s_set_vgpr_msb 0xf44                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_cndmask_b32_e32 v7 /*v263*/, 0xff800000, v7 /*v263*/, vcc_lo
	s_set_vgpr_msb 0x4400                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_20:                               ;   in Loop: Header=BB0_6 Depth=1
	s_set_vgpr_msb 34                       ;  msbs: dst=0 src0=2 src1=0 src2=2
	v_max3_num_f32 v17, v8 /*v520*/, 0xff800000, v9 /*v521*/
	s_set_vgpr_msb 0x2228                   ;  msbs: dst=0 src0=0 src1=2 src2=2
	v_max3_num_f32 v17, v17, v10 /*v522*/, v11 /*v523*/
	v_max3_num_f32 v17, v17, v12 /*v524*/, v13 /*v525*/
	v_max3_num_f32 v17, v17, v14 /*v526*/, v15 /*v527*/
	v_max3_num_f32 v17, v17, v0 /*v512*/, v1 /*v513*/
	v_max3_num_f32 v17, v17, v2 /*v514*/, v3 /*v515*/
	v_max3_num_f32 v17, v17, v4 /*v516*/, v5 /*v517*/
	v_max3_num_f32 v17, v17, v6 /*v518*/, v7 /*v519*/
	s_set_vgpr_msb 0x2814                   ;  msbs: dst=0 src0=0 src1=1 src2=1
	v_max3_num_f32 v17, v17, v40 /*v296*/, v41 /*v297*/
	v_max3_num_f32 v17, v17, v42 /*v298*/, v43 /*v299*/
	v_max3_num_f32 v17, v17, v44 /*v300*/, v45 /*v301*/
	v_max3_num_f32 v17, v17, v46 /*v302*/, v47 /*v303*/
	v_max3_num_f32 v17, v17, v32 /*v288*/, v33 /*v289*/
	v_max3_num_f32 v17, v17, v34 /*v290*/, v35 /*v291*/
	v_max3_num_f32 v17, v17, v36 /*v292*/, v37 /*v293*/
	v_max3_num_f32 v17, v17, v38 /*v294*/, v39 /*v295*/
	v_max3_num_f32 v17, v17, v24 /*v280*/, v25 /*v281*/
	v_max3_num_f32 v17, v17, v26 /*v282*/, v27 /*v283*/
	v_max3_num_f32 v17, v17, v28 /*v284*/, v29 /*v285*/
	v_max3_num_f32 v17, v17, v30 /*v286*/, v31 /*v287*/
	v_max3_num_f32 v17, v17, v16 /*v272*/, v17 /*v273*/
	v_max3_num_f32 v17, v17, v18 /*v274*/, v19 /*v275*/
	v_max3_num_f32 v17, v17, v20 /*v276*/, v21 /*v277*/
	v_max3_num_f32 v17, v17, v22 /*v278*/, v23 /*v279*/
	v_max3_num_f32 v17, v17, v8 /*v264*/, v9 /*v265*/
	v_max3_num_f32 v17, v17, v10 /*v266*/, v11 /*v267*/
	v_max3_num_f32 v17, v17, v12 /*v268*/, v13 /*v269*/
	v_max3_num_f32 v17, v17, v14 /*v270*/, v15 /*v271*/
	v_max3_num_f32 v17, v17, v0 /*v256*/, v1 /*v257*/
	v_max3_num_f32 v17, v17, v2 /*v258*/, v3 /*v259*/
	v_max3_num_f32 v17, v17, v4 /*v260*/, v5 /*v261*/
	v_max3_num_f32 v17, v17, v6 /*v262*/, v7 /*v263*/
	v_mov_b32_e32 v18, v17
	v_permlanex16_b32 v18, v18, s65, 0xfedcba98
	s_set_vgpr_msb 0x1430                   ;  msbs: dst=0 src0=0 src1=0 src2=3
	v_max_num_f32_e32 v17, v17, v18
	v_fma_f32 v18, v17, s60, -v188 /*v956*/
	v_mul_f32_e32 v17, s60, v17
	v_cmp_lt_f32_e32 vcc_lo, 0x3fb8aa3b, v18
	s_set_vgpr_msb 0x3003                   ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_max_num_f32_e32 v17, v188 /*v956*/, v17
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x30c                    ;  msbs: dst=0 src0=0 src1=3 src2=0
	v_cndmask_b32_e64 v17, v17, v188 /*v956*/, s2
	s_set_vgpr_msb 0xc02                    ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_fma_f32 v18, v8 /*v520*/, s60, -v17
	v_fma_f32 v19, v9 /*v521*/, s60, -v17
	v_fma_f32 v20, v10 /*v522*/, s60, -v17
	s_set_vgpr_msb 0x240                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v48 /*v304*/, v18
	v_nop
	s_set_vgpr_msb 0x4002                   ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_fma_f32 v18, v11 /*v523*/, s60, -v17
	s_set_vgpr_msb 0x240                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v49 /*v305*/, v19
	v_nop
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_add_f32_e32 v19, v48 /*v304*/, v49 /*v305*/
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v50 /*v306*/, v20
	v_nop
	s_set_vgpr_msb 0x4002                   ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_fma_f32 v20, v12 /*v524*/, s60, -v17
	s_set_vgpr_msb 0x240                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v51 /*v307*/, v18
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v50 /*v306*/, v19
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v52 /*v308*/, v20
	s_set_vgpr_msb 0x4002                   ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_fma_f32 v19, v13 /*v525*/, s60, -v17
	v_fma_f32 v20, v14 /*v526*/, s60, -v17
	s_set_vgpr_msb 0x201                    ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v51 /*v307*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v53 /*v309*/, v19
	v_nop
	s_set_vgpr_msb 0x4002                   ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_fma_f32 v19, v15 /*v527*/, s60, -v17
	s_set_vgpr_msb 0x240                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v54 /*v310*/, v20
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v52 /*v308*/, v18
	s_set_vgpr_msb 0x102                    ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_fma_f32 v20, v0 /*v512*/, s60, -v17
	s_set_vgpr_msb 0x240                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v55 /*v311*/, v19
	v_nop
	s_set_vgpr_msb 0x4002                   ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_fma_f32 v19, v1 /*v513*/, s60, -v17
	s_set_vgpr_msb 0x201                    ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v53 /*v309*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v56 /*v312*/, v20
	v_nop
	s_set_vgpr_msb 0x4002                   ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_fma_f32 v20, v2 /*v514*/, s60, -v17
	s_set_vgpr_msb 0x240                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v57 /*v313*/, v19
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v54 /*v310*/, v18
	s_set_vgpr_msb 0x102                    ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_fma_f32 v19, v3 /*v515*/, s60, -v17
	s_set_vgpr_msb 0x240                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v58 /*v314*/, v20
	v_nop
	s_set_vgpr_msb 0x4002                   ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_fma_f32 v20, v4 /*v516*/, s60, -v17
	s_set_vgpr_msb 0x201                    ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v55 /*v311*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v59 /*v315*/, v19
	v_nop
	s_set_vgpr_msb 0x4002                   ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_fma_f32 v19, v5 /*v517*/, s60, -v17
	s_set_vgpr_msb 0x240                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v60 /*v316*/, v20
	v_nop
	s_set_vgpr_msb 0x4002                   ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_fma_f32 v20, v6 /*v518*/, s60, -v17
	s_set_vgpr_msb 0x201                    ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v56 /*v312*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v61 /*v317*/, v19
	v_nop
	s_set_vgpr_msb 0x4002                   ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_fma_f32 v19, v7 /*v519*/, s60, -v17
	s_set_vgpr_msb 0x240                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v62 /*v318*/, v20
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v57 /*v313*/, v18
	v_fma_f32 v20, v40 /*v296*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v63 /*v319*/, v19
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v19, v41 /*v297*/, s60, -v17
	v_add_f32_e32 v18, v58 /*v314*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v40 /*v296*/, v20
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v20, v42 /*v298*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v41 /*v297*/, v19
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v59 /*v315*/, v18
	v_fma_f32 v19, v43 /*v299*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v42 /*v298*/, v20
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v20, v44 /*v300*/, s60, -v17
	v_add_f32_e32 v18, v60 /*v316*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v43 /*v299*/, v19
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v19, v45 /*v301*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v44 /*v300*/, v20
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v20, v46 /*v302*/, s60, -v17
	v_add_f32_e32 v18, v61 /*v317*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v45 /*v301*/, v19
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v19, v47 /*v303*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v46 /*v302*/, v20
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v62 /*v318*/, v18
	v_fma_f32 v20, v32 /*v288*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v47 /*v303*/, v19
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v19, v33 /*v289*/, s60, -v17
	v_add_f32_e32 v18, v63 /*v319*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v32 /*v288*/, v20
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v20, v34 /*v290*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v33 /*v289*/, v19
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v40 /*v296*/, v18
	v_fma_f32 v19, v35 /*v291*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v34 /*v290*/, v20
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v20, v36 /*v292*/, s60, -v17
	v_add_f32_e32 v18, v41 /*v297*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v35 /*v291*/, v19
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v19, v37 /*v293*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v36 /*v292*/, v20
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v20, v38 /*v294*/, s60, -v17
	v_add_f32_e32 v18, v42 /*v298*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v37 /*v293*/, v19
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v19, v39 /*v295*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v38 /*v294*/, v20
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v43 /*v299*/, v18
	v_fma_f32 v20, v24 /*v280*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v39 /*v295*/, v19
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v19, v25 /*v281*/, s60, -v17
	v_add_f32_e32 v18, v44 /*v300*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v24 /*v280*/, v20
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v20, v26 /*v282*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v25 /*v281*/, v19
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v45 /*v301*/, v18
	v_fma_f32 v19, v27 /*v283*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v26 /*v282*/, v20
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v20, v28 /*v284*/, s60, -v17
	v_add_f32_e32 v18, v46 /*v302*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v27 /*v283*/, v19
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v19, v29 /*v285*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v28 /*v284*/, v20
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v20, v30 /*v286*/, s60, -v17
	v_add_f32_e32 v18, v47 /*v303*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v29 /*v285*/, v19
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v19, v31 /*v287*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v30 /*v286*/, v20
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v32 /*v288*/, v18
	v_fma_f32 v20, v16 /*v272*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v31 /*v287*/, v19
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v19, v17 /*v273*/, s60, -v17
	v_add_f32_e32 v18, v33 /*v289*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v16 /*v272*/, v20
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v20, v18 /*v274*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v17 /*v273*/, v19
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v34 /*v290*/, v18
	v_fma_f32 v19, v19 /*v275*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v18 /*v274*/, v20
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v20, v20 /*v276*/, s60, -v17
	v_add_f32_e32 v18, v35 /*v291*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v19 /*v275*/, v19
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v19, v21 /*v277*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v20 /*v276*/, v20
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v20, v22 /*v278*/, s60, -v17
	v_add_f32_e32 v18, v36 /*v292*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v21 /*v277*/, v19
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v19, v23 /*v279*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v22 /*v278*/, v20
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v37 /*v293*/, v18
	v_fma_f32 v20, v8 /*v264*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v23 /*v279*/, v19
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v19, v9 /*v265*/, s60, -v17
	v_add_f32_e32 v18, v38 /*v294*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v64 /*v320*/, v20
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v20, v10 /*v266*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v65 /*v321*/, v19
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v39 /*v295*/, v18
	v_fma_f32 v19, v11 /*v267*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v66 /*v322*/, v20
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v20, v12 /*v268*/, s60, -v17
	v_add_f32_e32 v18, v24 /*v280*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v67 /*v323*/, v19
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v19, v13 /*v269*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v68 /*v324*/, v20
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v20, v14 /*v270*/, s60, -v17
	v_add_f32_e32 v18, v25 /*v281*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v69 /*v325*/, v19
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v19, v15 /*v271*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v70 /*v326*/, v20
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v26 /*v282*/, v18
	v_fma_f32 v20, v0 /*v256*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v71 /*v327*/, v19
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v19, v1 /*v257*/, s60, -v17
	v_add_f32_e32 v18, v27 /*v283*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v72 /*v328*/, v20
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v20, v2 /*v258*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v73 /*v329*/, v19
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v28 /*v284*/, v18
	v_fma_f32 v19, v3 /*v259*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v74 /*v330*/, v20
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v20, v4 /*v260*/, s60, -v17
	v_add_f32_e32 v18, v29 /*v285*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v75 /*v331*/, v19
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v19, v5 /*v261*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v76 /*v332*/, v20
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v20, v6 /*v262*/, s60, -v17
	v_add_f32_e32 v18, v30 /*v286*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v77 /*v333*/, v19
	v_nop
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_fma_f32 v19, v7 /*v263*/, s60, -v17
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v78 /*v334*/, v20
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v31 /*v287*/, v18
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_exp_f32_e32 v79 /*v335*/, v19
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_add_f32_e32 v18, v16 /*v272*/, v18
	v_add_f32_e32 v18, v17 /*v273*/, v18
	v_add_f32_e32 v18, v18 /*v274*/, v18
	v_add_f32_e32 v18, v19 /*v275*/, v18
	v_add_f32_e32 v18, v20 /*v276*/, v18
	v_add_f32_e32 v18, v21 /*v277*/, v18
	v_add_f32_e32 v18, v22 /*v278*/, v18
	v_add_f32_e32 v18, v23 /*v279*/, v18
	v_add_f32_e32 v18, v64 /*v320*/, v18
	v_add_f32_e32 v18, v65 /*v321*/, v18
	v_add_f32_e32 v18, v66 /*v322*/, v18
	v_add_f32_e32 v18, v67 /*v323*/, v18
	v_add_f32_e32 v18, v68 /*v324*/, v18
	v_add_f32_e32 v18, v69 /*v325*/, v18
	v_add_f32_e32 v18, v70 /*v326*/, v18
	v_add_f32_e32 v18, v71 /*v327*/, v18
	v_add_f32_e32 v18, v72 /*v328*/, v18
	v_add_f32_e32 v18, v73 /*v329*/, v18
	v_add_f32_e32 v18, v74 /*v330*/, v18
	v_add_f32_e32 v18, v75 /*v331*/, v18
	v_add_f32_e32 v18, v76 /*v332*/, v18
	v_add_f32_e32 v18, v77 /*v333*/, v18
	v_add_f32_e32 v18, v78 /*v334*/, v18
	v_add_f32_e32 v19, v79 /*v335*/, v18
	s_set_vgpr_msb 0x103                    ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_sub_f32_e32 v18, v188 /*v956*/, v17
	s_set_vgpr_msb 0x300                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_mov_b32_e32 v20, v19
	v_exp_f32_e32 v18, v18
	v_permlanex16_b32 v20, v20, s65, 0xfedcba98
	s_cbranch_vccz .LBB0_5
; %bb.21:                               ;   in Loop: Header=BB0_6 Depth=1
	s_set_vgpr_msb 0x82                     ;  msbs: dst=2 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[78:79] /*v[590:591]*/, v[78:79] /*v[590:591]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77] /*v[588:589]*/, v[76:77] /*v[588:589]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75] /*v[586:587]*/, v[74:75] /*v[586:587]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73] /*v[584:585]*/, v[72:73] /*v[584:585]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71] /*v[582:583]*/, v[70:71] /*v[582:583]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67] /*v[578:579]*/, v[66:67] /*v[578:579]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65] /*v[576:577]*/, v[64:65] /*v[576:577]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63] /*v[574:575]*/, v[62:63] /*v[574:575]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61] /*v[572:573]*/, v[60:61] /*v[572:573]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[58:59] /*v[570:571]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[56:57] /*v[568:569]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[54:55] /*v[566:567]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[52:53] /*v[564:565]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[50:51] /*v[562:563]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49] /*v[560:561]*/, v[48:49] /*v[560:561]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[46:47] /*v[558:559]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[44:45] /*v[556:557]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43] /*v[554:555]*/, v[42:43] /*v[554:555]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[40:41] /*v[552:553]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[38:39] /*v[550:551]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[36:37] /*v[548:549]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[34:35] /*v[546:547]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33] /*v[544:545]*/, v[32:33] /*v[544:545]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31] /*v[542:543]*/, v[30:31] /*v[542:543]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29] /*v[540:541]*/, v[28:29] /*v[540:541]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27] /*v[538:539]*/, v[26:27] /*v[538:539]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[24:25] /*v[536:537]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[22:23] /*v[534:535]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[20:21] /*v[532:533]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[18:19] /*v[530:531]*/, v[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17] /*v[528:529]*/, v[16:17] /*v[528:529]*/, v[18:19] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_branch .LBB0_5
.LBB0_22:                               ; %._crit_edge.loopexit
	v_mov_b32_e32 v2, v27
.LBB0_23:                               ; %._crit_edge
	v_div_scale_f32 v3, null, v22, v22, 1.0
	s_load_b128 s[0:3], s[0:1], 0x50 nv
	v_div_scale_f32 v5, vcc_lo, 1.0, v22, 1.0
	s_mov_b32 s7, 0
	s_lshl_b32 s6, s68, 8
	v_rcp_f32_e32 v4, v3
	v_ashrrev_i32_e32 v27, 31, v26
	v_fma_f32 v0, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v0, v4
	s_wait_kmcnt 0x0
	v_mad_nc_u64_u32 v[0:1], s2, v26, 0
	s_mul_u64 s[0:1], s[0:1], s[70:71]
	v_mul_f32_e32 v6, v5, v4
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_nc_u64 s[4:5], s[16:17], s[0:1]
	v_cmp_eq_u32_e64 s0, 0, v37
	v_fma_f32 v7, -v3, v6, v5
	v_mad_u32 v1, s2, v27, v1
	v_fmac_f32_e32 v6, v7, v4
	v_mad_u32 v1, s3, v26, v1
	v_fma_f32 v3, -v3, v6, v5
	v_div_fmas_f32 v3, v3, v4, v6
	v_cmp_lt_f32_e32 vcc_lo, 0, v22
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[4:5]
	v_div_fixup_f32 v4, v3, v22, 1.0
	v_mov_b32_e32 v3, 0
	v_add_nc_u64_e32 v[0:1], s[6:7], v[0:1]
	v_cndmask_b32_e32 v16, 0, v4, vcc_lo
	v_lshl_add_u64 v[0:1], v[2:3], 1, v[0:1]
	s_set_vgpr_msb 3                        ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_dual_mul_f32 v4, v8 /*v776*/, v16 :: v_dual_mul_f32 v5, v9 /*v777*/, v16
	v_pk_mul_f32 v[6:7], v[10:11] /*v[778:779]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[12:13] /*v[780:781]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[14:15] /*v[782:783]*/, v[16:17] op_sel_hi:[1,0]
	v_dual_mul_f32 v18, v0 /*v768*/, v16 :: v_dual_mul_f32 v23, v1 /*v769*/, v16
	v_pk_mul_f32 v[12:13], v[2:3] /*v[770:771]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[4:5] /*v[772:773]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[6:7] /*v[774:775]*/, v[16:17] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x302                    ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_dual_mul_f32 v24, v248 /*v760*/, v16 :: v_dual_mul_f32 v36, v249 /*v761*/, v16
	v_pk_mul_f32 v[30:31], v[250:251] /*v[762:763]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[252:253] /*v[764:765]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[254:255] /*v[766:767]*/, v[16:17] op_sel_hi:[1,0]
	v_dual_mul_f32 v44, v240 /*v752*/, v16 :: v_dual_mul_f32 v45, v241 /*v753*/, v16
	v_pk_mul_f32 v[38:39], v[242:243] /*v[754:755]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[244:245] /*v[756:757]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[246:247] /*v[758:759]*/, v[16:17] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v4, v4, v5
	v_cvt_pk_bf16_f32 v5, v6, v7
	v_cvt_pk_bf16_f32 v6, v8, v9
	v_cvt_pk_bf16_f32 v7, v10, v11
	v_cvt_pk_bf16_f32 v8, v18, v23
	v_cvt_pk_bf16_f32 v9, v12, v13
	v_cvt_pk_bf16_f32 v10, v14, v15
	v_cvt_pk_bf16_f32 v11, v28, v29
	v_cvt_pk_bf16_f32 v12, v24, v36
	v_cvt_pk_bf16_f32 v13, v30, v31
	v_cvt_pk_bf16_f32 v14, v32, v33
	v_cvt_pk_bf16_f32 v15, v34, v35
	v_cvt_pk_bf16_f32 v28, v44, v45
	v_cvt_pk_bf16_f32 v29, v38, v39
	v_cvt_pk_bf16_f32 v30, v40, v41
	v_cvt_pk_bf16_f32 v31, v42, v43
	s_clause 0x3
	global_store_b128 v[0:1], v[4:7], off
	global_store_b128 v[0:1], v[8:11], off offset:32
	global_store_b128 v[0:1], v[12:15], off offset:64
	global_store_b128 v[0:1], v[28:31], off offset:96
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[6:7], v[234:235] /*v[746:747]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[236:237] /*v[748:749]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[238:239] /*v[750:751]*/, v[16:17] op_sel_hi:[1,0]
	v_dual_mul_f32 v14, v224 /*v736*/, v16 :: v_dual_mul_f32 v15, v225 /*v737*/, v16
	v_pk_mul_f32 v[12:13], v[226:227] /*v[738:739]*/, v[16:17] op_sel_hi:[1,0]
	v_dual_mul_f32 v46, v232 /*v744*/, v16 :: v_dual_mul_f32 v18, v233 /*v745*/, v16
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v5, v6, v7
	v_cvt_pk_bf16_f32 v6, v8, v9
	v_cvt_pk_bf16_f32 v7, v10, v11
	v_cvt_pk_bf16_f32 v8, v14, v15
	v_cvt_pk_bf16_f32 v9, v12, v13
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[10:11], v[228:229] /*v[740:741]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[230:231] /*v[742:743]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[218:219] /*v[730:731]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[220:221] /*v[732:733]*/, v[16:17] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v4, v46, v18
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_dual_mul_f32 v18, v216 /*v728*/, v16 :: v_dual_mul_f32 v23, v217 /*v729*/, v16
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v10, v10, v11
	v_cvt_pk_bf16_f32 v11, v12, v13
	v_cvt_pk_bf16_f32 v13, v14, v15
	v_cvt_pk_bf16_f32 v14, v28, v29
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[28:29], v[222:223] /*v[734:735]*/, v[16:17] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v12, v18, v23
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_dual_mul_f32 v18, v208 /*v720*/, v16 :: v_dual_mul_f32 v23, v209 /*v721*/, v16
	v_pk_mul_f32 v[30:31], v[210:211] /*v[722:723]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[212:213] /*v[724:725]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[214:215] /*v[726:727]*/, v[16:17] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v15, v28, v29
	v_cvt_pk_bf16_f32 v28, v18, v23
	v_cvt_pk_bf16_f32 v29, v30, v31
	v_cvt_pk_bf16_f32 v30, v32, v33
	v_cvt_pk_bf16_f32 v31, v34, v35
	s_clause 0x3
	global_store_b128 v[0:1], v[4:7], off offset:128
	global_store_b128 v[0:1], v[8:11], off offset:160
	global_store_b128 v[0:1], v[12:15], off offset:192
	global_store_b128 v[0:1], v[28:31], off offset:224
	s_wait_xcnt 0x0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_25
; %bb.24:
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v22
	s_lshl_b64 s[8:9], s[34:35], 2
	s_add_nc_u64 s[8:9], s[18:19], s[8:9]
	v_cndmask_b32_e64 v1, 0, 32, vcc_lo
	v_cndmask_b32_e64 v0, 0, 0x42000000, vcc_lo
	v_ldexp_f32 v1, v22, v1
	v_log_f32_e32 v1, v1
	v_nop
	v_sub_f32_e32 v0, v1, v0
	s_set_vgpr_msb 3                        ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_add_f32_e32 v0, v17 /*v785*/, v0
	v_mul_f32_e32 v4, 0x3f317218, v0
	s_set_vgpr_msb 0x300                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshl_add_u64 v[0:1], v[26:27], 2, s[8:9]
	global_store_b32 v[0:1], v4, off
.LBB0_25:                               ; %_ZZ12k_mla_fwd_mxENKUlT_E5_clISt17integral_constantIiLi0EEEEDaS_.exit.i.i
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s1
	v_div_scale_f32 v6, null, v21, v21, 1.0
	s_add_co_i32 s1, s33, 64
	v_div_scale_f32 v8, vcc_lo, 1.0, v21, 1.0
	v_rcp_f32_e32 v7, v6
	v_nop
	v_fma_f32 v1, -v6, v7, 1.0
	v_dual_fmac_f32 v7, v1, v7 :: v_dual_bitop2_b32 v0, s1, v25 bitop3:0x54
	v_mad_nc_u64_u32 v[4:5], s2, v0, 0
	v_dual_ashrrev_i32 v1, 31, v0 :: v_dual_mul_f32 v9, v8, v7
	v_mad_u32 v5, s2, v1, v5
	v_fma_f32 v10, -v6, v9, v8
	v_fmac_f32_e32 v9, v10, v7
	v_mad_u32 v5, s3, v0, v5
	v_fma_f32 v6, -v6, v9, v8
	v_div_fmas_f32 v6, v6, v7, v9
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[4:5]
	v_cmp_lt_f32_e32 vcc_lo, 0, v21
	v_div_fixup_f32 v6, v6, v21, 1.0
	v_add_nc_u64_e32 v[4:5], s[6:7], v[4:5]
	v_cndmask_b32_e32 v16, 0, v6, vcc_lo
	v_lshl_add_u64 v[22:23], v[2:3], 1, v[4:5]
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_dual_mul_f32 v4, v200 /*v712*/, v16 :: v_dual_mul_f32 v5, v201 /*v713*/, v16
	v_pk_mul_f32 v[6:7], v[202:203] /*v[714:715]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[204:205] /*v[716:717]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[206:207] /*v[718:719]*/, v[16:17] op_sel_hi:[1,0]
	v_dual_mul_f32 v18, v192 /*v704*/, v16 :: v_dual_mul_f32 v24, v193 /*v705*/, v16
	v_pk_mul_f32 v[12:13], v[194:195] /*v[706:707]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[196:197] /*v[708:709]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[198:199] /*v[710:711]*/, v[16:17] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v4, v4, v5
	v_cvt_pk_bf16_f32 v5, v6, v7
	v_cvt_pk_bf16_f32 v6, v8, v9
	v_cvt_pk_bf16_f32 v7, v10, v11
	v_cvt_pk_bf16_f32 v8, v18, v24
	v_cvt_pk_bf16_f32 v9, v12, v13
	v_cvt_pk_bf16_f32 v10, v14, v15
	v_cvt_pk_bf16_f32 v11, v26, v27
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[34:35], v[178:179] /*v[690:691]*/, v[16:17] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_clause 0x1
	global_store_b128 v[22:23], v[4:7], off
	global_store_b128 v[22:23], v[8:11], off offset:32
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[6:7], v[162:163] /*v[674:675]*/, v[16:17] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v27, v34, v35
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[8:9], v[166:167] /*v[678:679]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[154:155] /*v[666:667]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[156:157] /*v[668:669]*/, v[16:17] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v5, v6, v7
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[6:7], v[164:165] /*v[676:677]*/, v[16:17] op_sel_hi:[1,0]
	v_dual_mul_f32 v46, v184 /*v696*/, v16 :: v_dual_mul_f32 v47, v185 /*v697*/, v16
	v_pk_mul_f32 v[28:29], v[186:187] /*v[698:699]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[188:189] /*v[700:701]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[190:191] /*v[702:703]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[180:181] /*v[692:693]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[182:183] /*v[694:695]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[170:171] /*v[682:683]*/, v[16:17] op_sel_hi:[1,0]
	v_dual_mul_f32 v18, v152 /*v664*/, v16 :: v_dual_mul_f32 v24, v153 /*v665*/, v16
	v_dual_mul_f32 v48, v176 /*v688*/, v16 :: v_dual_mul_f32 v49, v177 /*v689*/, v16
	v_dual_mul_f32 v50, v168 /*v680*/, v16 :: v_dual_mul_f32 v51, v169 /*v681*/, v16
	v_pk_mul_f32 v[42:43], v[172:173] /*v[684:685]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[174:175] /*v[686:687]*/, v[16:17] op_sel_hi:[1,0]
	v_dual_mul_f32 v52, v160 /*v672*/, v16 :: v_dual_mul_f32 v53, v161 /*v673*/, v16
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v6, v6, v7
	v_cvt_pk_bf16_f32 v7, v8, v9
	v_cvt_pk_bf16_f32 v9, v10, v11
	v_cvt_pk_bf16_f32 v10, v34, v35
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[34:35], v[158:159] /*v[670:671]*/, v[16:17] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v12, v46, v47
	v_cvt_pk_bf16_f32 v13, v28, v29
	v_cvt_pk_bf16_f32 v14, v30, v31
	v_cvt_pk_bf16_f32 v15, v32, v33
	v_cvt_pk_bf16_f32 v28, v36, v37
	v_cvt_pk_bf16_f32 v29, v38, v39
	v_cvt_pk_bf16_f32 v31, v40, v41
	v_cvt_pk_bf16_f32 v8, v18, v24
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_dual_mul_f32 v18, v144 /*v656*/, v16 :: v_dual_mul_f32 v24, v145 /*v657*/, v16
	v_pk_mul_f32 v[36:37], v[146:147] /*v[658:659]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[148:149] /*v[660:661]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[150:151] /*v[662:663]*/, v[16:17] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v26, v48, v49
	v_cvt_pk_bf16_f32 v30, v50, v51
	v_cvt_pk_bf16_f32 v32, v42, v43
	v_cvt_pk_bf16_f32 v33, v44, v45
	v_cvt_pk_bf16_f32 v4, v52, v53
	v_cvt_pk_bf16_f32 v11, v34, v35
	v_cvt_pk_bf16_f32 v34, v18, v24
	v_cvt_pk_bf16_f32 v35, v36, v37
	v_cvt_pk_bf16_f32 v36, v38, v39
	v_cvt_pk_bf16_f32 v37, v40, v41
	s_clause 0x5
	global_store_b128 v[22:23], v[12:15], off offset:64
	global_store_b128 v[22:23], v[26:29], off offset:96
	global_store_b128 v[22:23], v[30:33], off offset:128
	global_store_b128 v[22:23], v[4:7], off offset:160
	global_store_b128 v[22:23], v[8:11], off offset:192
	global_store_b128 v[22:23], v[34:37], off offset:224
	s_wait_xcnt 0x0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_27
; %bb.26:
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v21
	s_lshl_b64 s[8:9], s[34:35], 2
	s_add_nc_u64 s[8:9], s[18:19], s[8:9]
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[8:9]
	v_cndmask_b32_e64 v5, 0, 32, vcc_lo
	v_cndmask_b32_e64 v4, 0, 0x42000000, vcc_lo
	v_ldexp_f32 v5, v21, v5
	v_log_f32_e32 v5, v5
	v_nop
	v_sub_f32_e32 v4, v5, v4
	s_set_vgpr_msb 3                        ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_add_f32_e32 v4, v189 /*v957*/, v4
	v_mul_f32_e32 v4, 0x3f317218, v4
	s_set_vgpr_msb 0x300                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	global_store_b32 v[0:1], v4, off
.LBB0_27:                               ; %_ZZ12k_mla_fwd_mxENKUlT_E5_clISt17integral_constantIiLi1EEEEDaS_.exit.i.i
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s1
	v_div_scale_f32 v6, null, v20, v20, 1.0
	s_add_co_i32 s1, s33, 0x80
	v_div_scale_f32 v8, vcc_lo, 1.0, v20, 1.0
	v_rcp_f32_e32 v7, v6
	v_nop
	v_fma_f32 v1, -v6, v7, 1.0
	v_dual_fmac_f32 v7, v1, v7 :: v_dual_bitop2_b32 v0, s1, v25 bitop3:0x54
	v_mad_nc_u64_u32 v[4:5], s2, v0, 0
	v_dual_ashrrev_i32 v1, 31, v0 :: v_dual_mul_f32 v9, v8, v7
	v_mad_u32 v5, s2, v1, v5
	v_fma_f32 v10, -v6, v9, v8
	v_fmac_f32_e32 v9, v10, v7
	v_mad_u32 v5, s3, v0, v5
	v_fma_f32 v6, -v6, v9, v8
	v_div_fmas_f32 v6, v6, v7, v9
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[4:5]
	v_cmp_lt_f32_e32 vcc_lo, 0, v20
	v_div_fixup_f32 v6, v6, v20, 1.0
	v_add_nc_u64_e32 v[4:5], s[6:7], v[4:5]
	v_cndmask_b32_e32 v16, 0, v6, vcc_lo
	v_lshl_add_u64 v[22:23], v[2:3], 1, v[4:5]
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_dual_mul_f32 v4, v136 /*v648*/, v16 :: v_dual_mul_f32 v5, v137 /*v649*/, v16
	v_pk_mul_f32 v[6:7], v[138:139] /*v[650:651]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[140:141] /*v[652:653]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[142:143] /*v[654:655]*/, v[16:17] op_sel_hi:[1,0]
	v_dual_mul_f32 v18, v128 /*v640*/, v16 :: v_dual_mul_f32 v21, v129 /*v641*/, v16
	v_pk_mul_f32 v[12:13], v[130:131] /*v[642:643]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[132:133] /*v[644:645]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[134:135] /*v[646:647]*/, v[16:17] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v4, v4, v5
	v_cvt_pk_bf16_f32 v5, v6, v7
	v_cvt_pk_bf16_f32 v6, v8, v9
	v_cvt_pk_bf16_f32 v7, v10, v11
	v_cvt_pk_bf16_f32 v8, v18, v21
	v_cvt_pk_bf16_f32 v9, v12, v13
	v_cvt_pk_bf16_f32 v10, v14, v15
	v_cvt_pk_bf16_f32 v11, v26, v27
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[34:35], v[114:115] /*v[626:627]*/, v[16:17] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_clause 0x1
	global_store_b128 v[22:23], v[4:7], off
	global_store_b128 v[22:23], v[8:11], off offset:32
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[6:7], v[98:99] /*v[610:611]*/, v[16:17] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v27, v34, v35
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[8:9], v[102:103] /*v[614:615]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[90:91] /*v[602:603]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[92:93] /*v[604:605]*/, v[16:17] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v5, v6, v7
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[6:7], v[100:101] /*v[612:613]*/, v[16:17] op_sel_hi:[1,0]
	v_dual_mul_f32 v24, v120 /*v632*/, v16 :: v_dual_mul_f32 v46, v121 /*v633*/, v16
	v_pk_mul_f32 v[28:29], v[122:123] /*v[634:635]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[124:125] /*v[636:637]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[126:127] /*v[638:639]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[116:117] /*v[628:629]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[118:119] /*v[630:631]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[106:107] /*v[618:619]*/, v[16:17] op_sel_hi:[1,0]
	v_dual_mul_f32 v18, v88 /*v600*/, v16 :: v_dual_mul_f32 v21, v89 /*v601*/, v16
	v_dual_mul_f32 v47, v112 /*v624*/, v16 :: v_dual_mul_f32 v48, v113 /*v625*/, v16
	v_dual_mul_f32 v49, v104 /*v616*/, v16 :: v_dual_mul_f32 v50, v105 /*v617*/, v16
	v_pk_mul_f32 v[42:43], v[108:109] /*v[620:621]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[110:111] /*v[622:623]*/, v[16:17] op_sel_hi:[1,0]
	v_dual_mul_f32 v51, v96 /*v608*/, v16 :: v_dual_mul_f32 v52, v97 /*v609*/, v16
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v6, v6, v7
	v_cvt_pk_bf16_f32 v7, v8, v9
	v_cvt_pk_bf16_f32 v9, v10, v11
	v_cvt_pk_bf16_f32 v10, v34, v35
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[34:35], v[94:95] /*v[606:607]*/, v[16:17] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v12, v24, v46
	v_cvt_pk_bf16_f32 v13, v28, v29
	v_cvt_pk_bf16_f32 v14, v30, v31
	v_cvt_pk_bf16_f32 v15, v32, v33
	v_cvt_pk_bf16_f32 v28, v36, v37
	v_cvt_pk_bf16_f32 v29, v38, v39
	v_cvt_pk_bf16_f32 v31, v40, v41
	v_cvt_pk_bf16_f32 v8, v18, v21
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_dual_mul_f32 v18, v80 /*v592*/, v16 :: v_dual_mul_f32 v21, v81 /*v593*/, v16
	v_pk_mul_f32 v[36:37], v[82:83] /*v[594:595]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[84:85] /*v[596:597]*/, v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[86:87] /*v[598:599]*/, v[16:17] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v26, v47, v48
	v_cvt_pk_bf16_f32 v30, v49, v50
	v_cvt_pk_bf16_f32 v32, v42, v43
	v_cvt_pk_bf16_f32 v33, v44, v45
	v_cvt_pk_bf16_f32 v4, v51, v52
	v_cvt_pk_bf16_f32 v11, v34, v35
	v_cvt_pk_bf16_f32 v34, v18, v21
	v_cvt_pk_bf16_f32 v35, v36, v37
	v_cvt_pk_bf16_f32 v36, v38, v39
	v_cvt_pk_bf16_f32 v37, v40, v41
	s_clause 0x5
	global_store_b128 v[22:23], v[12:15], off offset:64
	global_store_b128 v[22:23], v[26:29], off offset:96
	global_store_b128 v[22:23], v[30:33], off offset:128
	global_store_b128 v[22:23], v[4:7], off offset:160
	global_store_b128 v[22:23], v[8:11], off offset:192
	global_store_b128 v[22:23], v[34:37], off offset:224
	s_wait_xcnt 0x0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_29
; %bb.28:
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v20
	s_lshl_b64 s[8:9], s[34:35], 2
	s_add_nc_u64 s[8:9], s[18:19], s[8:9]
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[8:9]
	v_cndmask_b32_e64 v5, 0, 32, vcc_lo
	v_cndmask_b32_e64 v4, 0, 0x42000000, vcc_lo
	v_ldexp_f32 v5, v20, v5
	v_log_f32_e32 v5, v5
	v_nop
	v_sub_f32_e32 v4, v5, v4
	s_set_vgpr_msb 3                        ;  msbs: dst=0 src0=3 src1=0 src2=0
	v_add_f32_e32 v4, v248 /*v1016*/, v4
	v_mul_f32_e32 v4, 0x3f317218, v4
	s_set_vgpr_msb 0x300                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	global_store_b32 v[0:1], v4, off
.LBB0_29:                               ; %_ZZ12k_mla_fwd_mxENKUlT_E5_clISt17integral_constantIiLi2EEEEDaS_.exit.i.i
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s1
	v_div_scale_f32 v6, null, v19, v19, 1.0
	s_add_co_i32 s1, s33, 0xc0
	v_div_scale_f32 v8, vcc_lo, 1.0, v19, 1.0
	v_rcp_f32_e32 v7, v6
	v_nop
	v_fma_f32 v1, -v6, v7, 1.0
	v_dual_fmac_f32 v7, v1, v7 :: v_dual_bitop2_b32 v0, s1, v25 bitop3:0x54
	v_mad_nc_u64_u32 v[4:5], s2, v0, 0
	v_dual_ashrrev_i32 v1, 31, v0 :: v_dual_mul_f32 v9, v8, v7
	v_mad_u32 v5, s2, v1, v5
	v_fma_f32 v10, -v6, v9, v8
	v_fmac_f32_e32 v9, v10, v7
	v_mad_u32 v5, s3, v0, v5
	v_fma_f32 v6, -v6, v9, v8
	v_div_fmas_f32 v6, v6, v7, v9
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[4:5]
	v_cmp_lt_f32_e32 vcc_lo, 0, v19
	v_div_fixup_f32 v6, v6, v19, 1.0
	v_add_nc_u64_e32 v[4:5], s[6:7], v[4:5]
	v_cndmask_b32_e32 v14, 0, v6, vcc_lo
	v_lshl_add_u64 v[32:33], v[2:3], 1, v[4:5]
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_dual_mul_f32 v2, v72 /*v584*/, v14 :: v_dual_mul_f32 v3, v73 /*v585*/, v14
	v_pk_mul_f32 v[4:5], v[74:75] /*v[586:587]*/, v[14:15] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[76:77] /*v[588:589]*/, v[14:15] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[78:79] /*v[590:591]*/, v[14:15] op_sel_hi:[1,0]
	v_dual_mul_f32 v15, v64 /*v576*/, v14 :: v_dual_mul_f32 v16, v65 /*v577*/, v14
	v_pk_mul_f32 v[10:11], v[66:67] /*v[578:579]*/, v[14:15] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[68:69] /*v[580:581]*/, v[14:15] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[70:71] /*v[582:583]*/, v[14:15] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_cvt_pk_bf16_f32 v3, v4, v5
	v_cvt_pk_bf16_f32 v4, v6, v7
	v_cvt_pk_bf16_f32 v5, v8, v9
	v_cvt_pk_bf16_f32 v6, v15, v16
	v_cvt_pk_bf16_f32 v7, v10, v11
	v_cvt_pk_bf16_f32 v8, v12, v13
	v_cvt_pk_bf16_f32 v9, v20, v21
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[28:29], v[50:51] /*v[562:563]*/, v[14:15] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_clause 0x1
	global_store_b128 v[32:33], v[2:5], off
	global_store_b128 v[32:33], v[6:9], off offset:32
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[4:5], v[34:35] /*v[546:547]*/, v[14:15] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[58:59] /*v[570:571]*/, v[14:15] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[60:61] /*v[572:573]*/, v[14:15] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[62:63] /*v[574:575]*/, v[14:15] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[52:53] /*v[564:565]*/, v[14:15] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[54:55] /*v[566:567]*/, v[14:15] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[42:43] /*v[554:555]*/, v[14:15] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[44:45] /*v[556:557]*/, v[14:15] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[46:47] /*v[558:559]*/, v[14:15] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v21, v28, v29
	v_cvt_pk_bf16_f32 v3, v4, v5
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[4:5], v[36:37] /*v[548:549]*/, v[14:15] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[38:39] /*v[550:551]*/, v[14:15] op_sel_hi:[1,0]
	v_dual_mul_f32 v15, v24 /*v536*/, v14 :: v_dual_mul_f32 v16, v25 /*v537*/, v14
	v_pk_mul_f32 v[8:9], v[26:27] /*v[538:539]*/, v[14:15] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29] /*v[540:541]*/, v[14:15] op_sel_hi:[1,0]
	v_dual_mul_f32 v18, v56 /*v568*/, v14 :: v_dual_mul_f32 v42, v57 /*v569*/, v14
	v_dual_mul_f32 v43, v48 /*v560*/, v14 :: v_dual_mul_f32 v44, v49 /*v561*/, v14
	v_dual_mul_f32 v45, v40 /*v552*/, v14 :: v_dual_mul_f32 v46, v41 /*v553*/, v14
	v_dual_mul_f32 v47, v32 /*v544*/, v14 :: v_dual_mul_f32 v48, v33 /*v545*/, v14
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v4, v4, v5
	v_cvt_pk_bf16_f32 v5, v6, v7
	v_cvt_pk_bf16_f32 v7, v8, v9
	v_cvt_pk_bf16_f32 v8, v28, v29
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_pk_mul_f32 v[28:29], v[30:31] /*v[542:543]*/, v[14:15] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v10, v18, v42
	v_cvt_pk_bf16_f32 v11, v22, v23
	v_cvt_pk_bf16_f32 v12, v24, v25
	v_cvt_pk_bf16_f32 v13, v26, v27
	v_cvt_pk_bf16_f32 v22, v30, v31
	v_cvt_pk_bf16_f32 v23, v34, v35
	v_cvt_pk_bf16_f32 v6, v15, v16
	s_set_vgpr_msb 2                        ;  msbs: dst=0 src0=2 src1=0 src2=0
	v_dual_mul_f32 v16, v16 /*v528*/, v14 :: v_dual_mul_f32 v18, v17 /*v529*/, v14
	v_pk_mul_f32 v[30:31], v[18:19] /*v[530:531]*/, v[14:15] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[20:21] /*v[532:533]*/, v[14:15] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[22:23] /*v[534:535]*/, v[14:15] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x200                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cvt_pk_bf16_f32 v20, v43, v44
	v_cvt_pk_bf16_f32 v24, v45, v46
	v_cvt_pk_bf16_f32 v25, v36, v37
	v_cvt_pk_bf16_f32 v26, v38, v39
	v_cvt_pk_bf16_f32 v27, v40, v41
	v_cvt_pk_bf16_f32 v2, v47, v48
	v_cvt_pk_bf16_f32 v9, v28, v29
	v_cvt_pk_bf16_f32 v28, v16, v18
	v_cvt_pk_bf16_f32 v29, v30, v31
	v_cvt_pk_bf16_f32 v30, v34, v35
	v_cvt_pk_bf16_f32 v31, v14, v15
	s_clause 0x5
	global_store_b128 v[32:33], v[10:13], off offset:64
	global_store_b128 v[32:33], v[20:23], off offset:96
	global_store_b128 v[32:33], v[24:27], off offset:128
	global_store_b128 v[32:33], v[2:5], off offset:160
	global_store_b128 v[32:33], v[6:9], off offset:192
	global_store_b128 v[32:33], v[28:31], off offset:224
	s_wait_xcnt 0x0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_31
; %bb.30:
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v19
	s_lshl_b64 s[2:3], s[34:35], 2
	s_add_nc_u64 s[2:3], s[18:19], s[2:3]
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[2:3]
	v_cndmask_b32_e64 v3, 0, 32, vcc_lo
	v_cndmask_b32_e64 v2, 0, 0x42000000, vcc_lo
	v_ldexp_f32 v3, v19, v3
	v_log_f32_e32 v3, v3
	v_nop
	v_sub_f32_e32 v2, v3, v2
	v_add_f32_e32 v2, v17, v2
	v_mul_f32_e32 v2, 0x3f317218, v2
	global_store_b32 v[0:1], v2, off
.LBB0_31:                               ; %_ZN2mx10static_forILi4EZ12k_mla_fwd_mxEUlT_E5_EEvOT0_.exit
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_tensorcnt 0x0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	k_mla_fwd_mx, .Lfunc_end0-k_mla_fwd_mx
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel k_mla_fwd_mx
		.amdhsa_group_segment_fixed_size 241664
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 96
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 1021
		.amdhsa_next_free_sgpr 82
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-k_mla_fwd_mx)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .Lk_mla_fwd_mx.num_vgpr, 1021
	.set .Lk_mla_fwd_mx.num_agpr, 0
	.set .Lk_mla_fwd_mx.numbered_sgpr, 82
	.set .Lk_mla_fwd_mx.num_named_barrier, 0
	.set .Lk_mla_fwd_mx.private_seg_size, 0
	.set .Lk_mla_fwd_mx.uses_vcc, 1
	.set .Lk_mla_fwd_mx.uses_flat_scratch, 0
	.set .Lk_mla_fwd_mx.has_dyn_sized_stack, 0
	.set .Lk_mla_fwd_mx.has_recursion, 0
	.set .Lk_mla_fwd_mx.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 24540
; TotalNumSgprs: 84
; NumVgprs: 1021
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 241664 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 63
; NumSGPRsForWavesPerEU: 84
; NumVGPRsForWavesPerEU: 1021
; NamedBarCnt: 0
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.type	__hip_cuid_b91c1cbb6a653d2d,@object ; @__hip_cuid_b91c1cbb6a653d2d
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_b91c1cbb6a653d2d
__hip_cuid_b91c1cbb6a653d2d:
	.byte	0                               ; 0x0
	.size	__hip_cuid_b91c1cbb6a653d2d, 1

	.ident	"AMD clang version 24.0.0git (https://github.com/ROCm/llvm-project.git 862306352beb9946c78b3e355d0d5201bbccfa3a)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_b91c1cbb6a653d2d
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .offset:         68
        .size:           4
        .value_kind:     by_value
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .offset:         76
        .size:           4
        .value_kind:     by_value
      - .offset:         80
        .size:           8
        .value_kind:     by_value
      - .offset:         88
        .size:           8
        .value_kind:     by_value
    .group_segment_fixed_size: 241664
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 128
    .name:           k_mla_fwd_mx
    .private_segment_fixed_size: 0
    .sgpr_count:     84
    .sgpr_spill_count: 0
    .symbol:         k_mla_fwd_mx.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     1021
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgpu12.50-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
