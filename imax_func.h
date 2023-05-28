


tensor conv3d_split(tensor input, tensor kernels, tensor bias, int stride, int padding,
                    int depth_split, int channel_split, int out_channel_split) {
    // ... Assertions and output tensor initialization ...
    assert(input.channel == kernels.channel);
    assert(bias.channel == kernels.num);

    int output_depth = (input.depth - kernels.depth + 2 * padding) / stride + 1;
    int output_height = (input.height - kernels.height + 2 * padding) / stride + 1;
    int output_width = (input.width - kernels.width + 2 * padding) / stride + 1;

    tensor output;
    output.num = input.num;
    output.channel = kernels.num;
    output.depth = output_depth;
    output.height = output_height;
    output.width = output_width;
    output.data = (float *) malloc(output.num * output.channel * output.depth * output.height * output.width * sizeof(float));

    for (int n = 0; n < input.num; n++) {
        for (int k_start = 0; k_start < output.channel; k_start += out_channel_split) {
            int k_end = fmin(k_start + out_channel_split, output.channel);
            for (int c_start = 0; c_start < input.channel; c_start += channel_split) {
                int c_end = fmin(c_start + channel_split, input.channel);
                for (int d_start = 0; d_start < input.depth; d_start += depth_split) {
                    int d_end = fmin(d_start + depth_split, input.depth);

                    // Call conv3d() on the smaller task
                    tensor input_subtask = get_subtask(input, n, c_start, d_start, c_end, d_end);
                    tensor kernels_subtask = get_subtask(kernels, k_start, c_start, k_end, c_end);
                    tensor bias_subtask = get_subtask(bias, k_start, k_end);
                    tensor output_subtask = conv3d(input_subtask, kernels_subtask, bias_subtask, stride, padding);

                    // Merge the output_subtask into the output tensor
                    merge_output_subtask(output, output_subtask, n, k_start, d_start);

                    // Free the memory allocated for the subtasks
                    free_tensor(input_subtask);
                    free_tensor(kernels_subtask);
                    free_tensor(bias_subtask);
                    free_tensor(output_subtask);
                }
            }
        }
    }

    return output;
}

int get_tensor_index(tensor t, int n, int c, int d, int h, int w) {
    return (((n * t.channel + c) * t.depth + d) * t.height + h) * t.width + w;
}

tensor get_subtask(tensor input, int n, int c_start, int d_start, int c_end, int d_end) {
    tensor subtask;
    subtask.num = 1;
    subtask.channel = c_end - c_start;
    subtask.depth = d_end - d_start;
    subtask.height = input.height;
    subtask.width = input.width;

    int subtask_size = subtask.channel * subtask.depth * subtask.height * subtask.width;
    subtask.data = (float *) malloc(subtask_size * sizeof(float));

    for (int c = c_start, sub_c = 0; c < c_end; c++, sub_c++) {
        for (int d = d_start, sub_d = 0; d < d_end; d++, sub_d++) {
            for (int h = 0; h < input.height; h++) {
                for (int w = 0; w < input.width; w++) {
                    int input_idx = get_tensor_index(input, n, c, d, h, w);
                    int subtask_idx = get_tensor_index(subtask, 0, sub_c, sub_d, h, w);
                    subtask.data[subtask_idx] = input.data[input_idx];
                }
            }
        }
    }

    return subtask;
}

void merge_output_subtask(tensor output, tensor output_subtask, int n, int k_start, int d_start) {
    int subtask_channel = output_subtask.channel;
    int subtask_depth = output_subtask.depth;

    for (int k = 0; k < subtask_channel; k++) {
        for (int d = 0; d < subtask_depth; d++) {
            for (int h = 0; h < output_subtask.height; h++) {
                for (int w = 0; w < output_subtask.width; w++) {
                    int output_idx = get_tensor_index(output, n, k_start + k, d_start + d, h, w);
                    int subtask_idx = get_tensor_index(output_subtask, 0, k, d, h, w);
                    output.data[output_idx] += output_subtask.data[subtask_idx];
                }
            }
        }
    }
}


void xmax_conv_forward(float4D *in, float2D *kernel, float4D *out, int ksize)
{
  /* float4D.nstrides    .. batch_size        */
  /* float4D.nchannel    .. ichan/ochan       */
  /* float4D.kstrides    .. isize/osize       */
  /* float4D.stride_size .. isize/osize       */
  /* float4D.data                             */
  /* float2D.nstrides    .. ochan             */
  /* float2D.stride_size .. ichan*ksize*ksize */
  /* float2D.data                             */
  /* in[batch_size, ichan, isize*isize] * weight[ochan, ichan, ksize*ksize] */
  /*  -> out[batch_size, ochan, osize*osize ] */
  /* IM == M�ξ��, in->data�μ��դ�PAD�ɲ�   */
  /* float *i_inp; PAD+in->data ��copy     */
  /* float *i_ker; ker->data    ��copy     */
  /* float *i_out; out->data    ��copy     */

  /* PAD+IM*2                                 */
  /*      <-----Nich*28*28----->  <next img>  */
  /* IM A +--------++--------+ .. +--------+  */
  /*  A | | +-24-+ || +----+ | .. | +----+ |  */
  /*  | | | | ch0| || | ch1| | .. | | ch0| |  */
  /*  | | | 24   | || |    | | .. | |    | |  */
  /*  V | | +----+ || +----+ | .. | +----+ |  */
  /*    V +--------++--------+ .. +--------+  */
  /*      <PAD+IM*2>                          */
  /*        <-IM->                            */

  int   BATCH  = in->nstrides;  // 100
  int   RMGRP;
  int   IC     = in->nchannel;  // IMAP*Xin
  int   IM     = in->kstrides;  // 28
  int   OC     = out->nchannel; // W*Xout
  int   M      = out->kstrides; // 24
  int   K      = ksize;         // 5,4,3,2,1
  int   Klen   = OC*IC*K*K;
  int   OC4    = (OC+3)&~3;
  Uint  *in0   = in->data;      // IC*IM*IM
  Uint  *ker   = kernel->data;  // OC*IC*K*K
  Uint  *out0  = out->data;     // OC*M*M
  Uint  *ip0, *ip1, *ip2, *ip3, *ip4, *ip5, *kp,  kidx, *op;
  int   pad;
  int   count, top, iset, oc, w, ic, y, x;
  Ull   IM4, M4, IM4M4, IMlen, Mlen, Force;
  Ull   CHIP, img, rofs, cofs, iofs, oofs;
  Ull   LOOP1, LOOP0;
  Ull   INIT1, INIT0;
  Ull   AR[64][4];                     /* output of EX     in each unit */
  Ull   BR[64][4][4];                  /* output registers in each unit */
  Ull   r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull   r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull   cc0, cc1, cc2, cc3, ex0, ex1;

  if (IM == M)
    pad = 0;   /* PAD̵��.in����0.0���� */
  else if ((IM - K)/1 + 1 == M)
    pad = K/2; /* PADͭ��.in�������̰������� */
  else {
    printf("xmax_conv_forward error: IM=%d K=%d M=%d\n", IM, K, M);
    printf("IM == M || (IM-K)/1+1 == M\n");
    exit(-1);
  }

  /* i_inp, i_ker, i_out�ϳ��ݺѤ�����ǽɾ���ˤϻȤ�ʤ� */
  /*printf("<<<XMAX(C)>>>\n");*/
  /*printf("xmax IM=%d M=%d K=%d %d*%d*%d\n", IM, M, K, OC, BATCH*M*M, IC*K*K);*/
  /*printf("<<<XMAX(REAL)>>>\n");*/

  switch (K) { /* 5, 3, 2 */
  case 5:
    RMGRP = M; /* RMGRP = 24 {28,1,5,24,9,2} */
               /* RMGRP = 28 {32,3,5,28,11,2}*/
#undef  IMAP
#undef  W
#undef  NCHIP
/* IMAP > 1 ������ PBL1-1 ������ */
/* NCHIP  4 ������ PBL1-1 ������ */
#define IMAP  1
#define W     4
#define NCHIP 1
#undef XMAX_VALIDATE
//#define XMAX_VALIDATE
#ifdef XMAX_VALIDATE
    for (img=0; img<BATCH; img++) {
      for (top=0; top<M; top+=RMGRP) {
        for (iset=0; iset<IC; iset+=IMAP) { /* accumulate multiple sets of IC */
          for (oc=0; oc<OC4/NCHIP; oc+=W) { /* set output channel */                                            /* ICH��ʤ�٤���¸��,��¦LOOP����OCH��������դ��� */
      /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */ /* ICH��ʤ�٤���¸��,ʣ��CHIP����OCH��������դ��� */
        /*2*/ for (rofs=0; rofs<RMGRP&&(top+rofs)<M; rofs++) { /* image loop (row) */                          /* 1.��ICHʣ����   */
          /*1*/ for (cofs=0; cofs<M; cofs++) { /* image loop (cofs) */                                         /* 2.��ICH��ʿ���� */
                  iofs = rofs*IM+cofs;
                  oofs = rofs*M+cofs;
                  for (w=0; w<W&&(oc+w)<OC/NCHIP; w++) { /* set output channel */                              /* ICH��ʤ�٤���¸��,������Ѥ���OCH��������դ��� */
                    op = &out0[(img*OC+CHIP*OC/NCHIP+oc+w)*M*M+top*M+oofs]; /* top of output */
                    for (ic=0; ic<IMAP&&(iset+ic)<IC; ic++) { /* set offset of input channel */
                      ip0  = &in0[(img*IC+iset+ic)*IM*IM+pad*IM+pad]; /* top of input */
                      kp   = &ker[((CHIP*OC/NCHIP+oc+w)*IC+iset+ic)*K*K];
                      kidx = 0;
                      for (y=-(K/2); y<K-(K/2); y++) { /* kernel loop */
                        for (x=-(K/2); x<K-(K/2); x++) {
                          float in = (0 <= top+rofs+y+pad && top+rofs+y+pad < IM
                                   && 0 <=     cofs+x+pad &&     cofs+x+pad < IM)
                            ? *(float*)&ip0[top*IM+iofs+y*IM+x] : 0.0;
                          if (iset == 0 && ic == 0 && kidx == 0)
                            *(float*)op  = in * *(float*)&kp[kidx];
                          else
                            *(float*)op += in * *(float*)&kp[kidx];
                          kidx++;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
#endif
    /*{28,1,5,24,9,2}/{32,3,5,28,11,2},{14,11,3,14,16,2},{7,16,2,7,32,1},{7,32,2,6, 32,2}*/
    /* AAAAAAAAAAAAA / AAAAAAAAAAAAAA                                                    */
    xmax_cpyin(0, i_inp, &IM, in0, BATCH, IC, IM, M, K); /* copy the input data to the appropriate memory locations. */
    xmax_cpyin(0, i_ker, &K,  ker, IC,    OC,  K, K, 1); /* copy kernel (filter) data to the appropriate memory locations.*/
    xmax_bzero(i_out, BATCH*OC4*M*M); // initialize the output memory buffer with zeros.
    IM4   = IM*4;
    M4    = M*4;
    IM4M4 = IM4<<32|M4;
    IMlen = IM*(RMGRP+4);
    Mlen  = M*RMGRP;
    Force = 1;

    if (Klen > 65536/4/2 || IMlen > 65536/4/2 || Mlen > 65536/4/4)
      printf("   CNN5x5  Klen=%dB IMlen=%dB Mlen*4=%dB\n", (Uint)Klen*4, (Uint)IMlen*4, (Uint)Mlen*4*4);

   i_ker, Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][0][0],  (Ull)kp01[CHIP], o, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][1][1],  (Ull)kp02[CHIP], o, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][1][0],  (Ull)kp03[CHIP], o, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][2][1],  (Ull)ip00[n], iofs, MSK_W1, (Ull)it00, IMlen, 0, 0, (Ull)NULL, IMlen);\
  exe(OP_FMA, &AR[bp1][0], AR[b][0], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][1], AR[b][1], EXP_H3210, BR[b][2][1], EXP_H32 for (img=0; img<BATCH; img++) {
      for (top=0; top<M; top+=RMGRP) { /* output rows blk loop, top is index of row, step is RMGRP */
        for (iset=0; iset<IC; iset+=IMAP) {  /* accumulate multiple sets of IC */
          Uint *ip0  = &i_inp[(img*IC+iset+0)*IM*IM]; /*calculates the starting address (or top) of the input data, 
          i_inp holds the input data for the convolution. (img*IC+iset+0): prev img with IC chnnles, and curr img's iset*/
          /*IM*IM: herght*width, finally, *ip0 point to teh start addr */
          Uint *it00 = ip0+top*IM, *ip00[25]; // 25 pointers, store the pointers to each element in a 5x5 region of the input data.
	  ip00[ 0] = ip0+(top+0)*IM+0; ip00[ 1] = ip0+(top+0)*IM+1; ip00[ 2] = ip0+(top+0)*IM+2; ip00[ 3] = ip0+(top+0)*IM+3; ip00[ 4] = ip0+(top+0)*IM+4;
	  ip00[ 5] = ip0+(top+1)*IM+0; ip00[ 6] = ip0+(top+1)*IM+1; ip00[ 7] = ip0+(top+1)*IM+2; ip00[ 8] = ip0+(top+1)*IM+3; ip00[ 9] = ip0+(top+1)*IM+4;
	  ip00[10] = ip0+(top+2)*IM+0; ip00[11] = ip0+(top+2)*IM+1; ip00[12] = ip0+(top+2)*IM+2; ip00[13] = ip0+(top+2)*IM+3; ip00[14] = ip0+(top+2)*IM+4;
	  ip00[15] = ip0+(top+3)*IM+0; ip00[16] = ip0+(top+3)*IM+1; ip00[17] = ip0+(top+3)*IM+2; ip00[18] = ip0+(top+3)*IM+3; ip00[19] = ip0+(top+3)*IM+4;
	  ip00[20] = ip0+(top+4)*IM+0; ip00[21] = ip0+(top+4)*IM+1; ip00[22] = ip0+(top+4)*IM+2; ip00[23] = ip0+(top+4)*IM+3; ip00[24] = ip0+(top+4)*IM+4;

          for (oc=0; oc<OC4/NCHIP; oc+=W) { /* set output channel */
            Uint *kp00[NCHIP],*kp01[NCHIP],*kp02[NCHIP],*kp03[NCHIP];
            Uint *op0[NCHIP], *op1[NCHIP], *op2[NCHIP], *op3[NCHIP];
            Uint *ot0[NCHIP], *ot1[NCHIP], *ot2[NCHIP], *ot3[NCHIP];

            for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
              Uint choc  = CHIP*OC4/NCHIP+oc;
              kp00[CHIP]= (choc+0<OC) ? i_ker+((choc+0)*IC+iset+0)*K*K : i_ker;
	      kp01[CHIP]= (choc+1<OC) ? i_ker+((choc+1)*IC+iset+0)*K*K : i_ker;
	      kp02[CHIP]= (choc+2<OC) ? i_ker+((choc+2)*IC+iset+0)*K*K : i_ker;
	      kp03[CHIP]= (choc+3<OC) ? i_ker+((choc+3)*IC+iset+0)*K*K : i_ker;
              op0[CHIP] = i_out+(img*OC4+choc+0)*M*M+top*M; op1[CHIP] = i_out+(img*OC4+choc+1)*M*M+top*M; op2[CHIP] = i_out+(img*OC4+choc+2)*M*M+top*M; op3[CHIP] = i_out+(img*OC4+choc+3)*M*M+top*M;
              ot0[CHIP] = i_out+(img*OC4+choc+0)*M*M+top*M; ot1[CHIP] = i_out+(img*OC4+choc+1)*M*M+top*M; ot2[CHIP] = i_out+(img*OC4+choc+2)*M*M+top*M; ot3[CHIP] = i_out+(img*OC4+choc+3)*M*M+top*M;
            }

#define cnn5x5_core1(b, o, bp1, n) \
  mop(OP_LDWR,   1, &BR[b][0][1],  (Ull)kp00[CHIP], o, MSK_D0, (Ull)10, BR[b][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][2], AR[b][2], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][3], AR[b][3], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define cnn5x5_final(b, bp1) \
  mop(OP_LDWR,   1, &BR[bp1][0][1],  (Ull)op0[CHIP], oofs, MSK_W0, (Ull)ot0[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][1][1],  (Ull)op1[CHIP], oofs, MSK_W0, (Ull)ot1[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][2][1],  (Ull)op2[CHIP], oofs, MSK_W0, (Ull)ot2[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][3][1],  (Ull)op3[CHIP], oofs, MSK_W0, (Ull)ot3[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  exe(OP_FAD, &AR[bp1][0], AR[b][0], EXP_H3210, BR[bp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][1], AR[b][1], EXP_H3210, BR[bp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][2], AR[b][2], EXP_H3210, BR[bp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][3], AR[b][3], EXP_H3210, BR[bp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  mop(OP_STWR,   1, &AR[bp1][0], oofs, (Ull)op0[CHIP], MSK_D0, (Ull)ot0[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][1], oofs, (Ull)op1[CHIP], MSK_D0, (Ull)ot1[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][2], oofs, (Ull)op2[CHIP], MSK_D0, (Ull)ot2[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][3], oofs, (Ull)op3[CHIP], MSK_D0, (Ull)ot3[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen)

//EMAX5A begin cnn5x5 mapdist=0
      /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
        /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=(0-IM4)<<32|((0-M4)&0xffffffff); LOOP1--; INIT1=0) {                      /* mapped to FOR() on BR[63][1][0] */ /* stage#0 */
          /*1*/ for (INIT0=1,LOOP0=M,cofs=(0-4LL)<<32|((0-4LL)&0xffffffff); LOOP0--; INIT0=0) {                       /* mapped to FOR() on BR[63][0][0] */ /* stage#0 */
                  exe(OP_ADD,    &rofs, rofs,            EXP_H3210, INIT0?IM4M4:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,                  OP_NOP, 0LL); /* stage#0 */
                  exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, 4LL<<32|4LL,   EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffffffffffffLL, OP_NOP, 0LL); /* stage#0 */
                  exe(OP_ADD,    &iofs, rofs,            EXP_H3210, cofs,          EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffff00000000LL, OP_NOP, 0LL); /* stage#1 */
                  exe(OP_ADD,    &oofs, rofs,            EXP_H3210, cofs,          EXP_H3210, 0LL, EXP_H3210, OP_AND,   0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */

                  /****in0*****/
                  mop(OP_LDWR,   1, &BR[2][0][1],  (Ull)kp00[CHIP], 0LL, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][0][0],  (Ull)kp01[CHIP], 0LL, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][1][1],  (Ull)kp02[CHIP], 0LL, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][1][0],  (Ull)kp03[CHIP], 0LL, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen); /* stage#2 10KB */
                  mop(OP_LDWR,   1, &BR[2][2][1],  (Ull)ip00[0],   iofs, MSK_W1, (Ull)it00, IMlen, 0, 0, (Ull)NULL, IMlen);    /* stage#2 10KB */
                  exe(OP_FML, &AR[3][0], BR[2][2][1], EXP_H3210, BR[2][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][1], BR[2][2][1], EXP_H3210, BR[2][0][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][2], BR[2][2][1], EXP_H3210, BR[2][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][3], BR[2][2][1], EXP_H3210, BR[2][1][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
		  cnn5x5_core1( 3, 4LL, 4, 1);
		  cnn5x5_core1( 4, 8LL, 5, 2);
		  cnn5x5_core1( 5,12LL, 6, 3);
		  cnn5x5_core1( 6,16LL, 7, 4);
		  cnn5x5_core1( 7,20LL, 8, 5);
		  cnn5x5_core1( 8,24LL, 9, 6);
		  cnn5x5_core1( 9,28LL,10, 7);
		  cnn5x5_core1(10,32LL,11, 8);
		  cnn5x5_core1(11,36LL,12, 9);
		  cnn5x5_core1(12,40LL,13,10);
		  cnn5x5_core1(13,44LL,14,11);
		  cnn5x5_core1(14,48LL,15,12);
		  cnn5x5_core1(15,52LL,16,13);
		  cnn5x5_core1(16,56LL,17,14);
		  cnn5x5_core1(17,60LL,18,15);
		  cnn5x5_core1(18,64LL,19,16);
		  cnn5x5_core1(19,68LL,20,17);
		  cnn5x5_core1(20,72LL,21,18);
		  cnn5x5_core1(21,76LL,22,19);
		  cnn5x5_core1(22,80LL,23,20);
		  cnn5x5_core1(23,84LL,24,21);
		  cnn5x5_core1(24,88LL,25,22);
		  cnn5x5_core1(25,92LL,26,23);
		  cnn5x5_core1(26,96LL,27,24);
                  /****final*****/
		  cnn5x5_final(27,     28);
                }
              }
            }
//EMAX5A end
            if (Force) Force = 0;
          }
        }
      }
    }