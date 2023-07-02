
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-arm64/sample/mm_cnn_lf/RCS/cnn.c,v 1.4 2018/02/04 10:28:43 nakashim Exp nakashim $";

/*                          Copyright (C) 2013- by NAIST */
/*                           Primary writer: Y.Nakashima */
/*                                  nakashim@is.naist.jp */

#ifndef UTYPEDEF
#define UTYPEDEF
typedef unsigned char      Uchar;
typedef unsigned short     Ushort;
typedef unsigned int       Uint;
typedef unsigned long long Ull;
typedef long long int      Sll;
#if __AARCH64EL__ == 1
typedef long double Dll;
#else
typedef struct {Ull u[2];} Dll;
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fcntl.h>
#include <math.h>
#include <assert.h>//追加ren-im
#ifndef ARMSIML
#include <unistd.h>
#include <sys/times.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <pthread.h>
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>
#include <X11/extensions/Xdbe.h>
#endif


#define EMAX7//見やすくren-im
#define ARMZYNQ//見やすくren-im



int WD=320, HT=240, BITMAP=320*240, SCRWD=5, SCRHT=5, VECWD=240, VECHT=240, VECSTEP=4;

#if defined(EMAX6)
#include "../../src/conv-c2c/emax6.h"
#include "../../src/conv-c2c/emax6lib.c"
#endif
#if defined(EMAX7)
#include "../../src/conv-c2d/emax7.h"
#include "../../src/conv-c2d/emax7lib.c"
#endif
#if !defined(ARMSIML)
#include "./xdisp.c"
#endif

Uchar* membase;

sysinit(memsize, alignment) Uint memsize, alignment;
{
#if defined(ARMZYNQ) && defined(EMAX5)
  if (emax5_open() == NULL)
    exit(1);
  membase = emax_info.hpp_mmap;
  {volatile int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)membase+i)=0;}
#elif defined(ARMZYNQ) && defined(EMAX6)
  if (emax6_open() == NULL)
    exit(1);
  membase = emax_info.ddr_mmap;
  {volatile int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)membase+i)=0;}
#elif defined(ARMZYNQ) && defined(EMAX7)
  if (emax7_open() == NULL)
    exit(1);
  membase = emax_info[0].ddr_mmap;
  {volatile int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)membase+i)=0;}
#elif __linux__ == 1
  posix_memalign(&membase, alignment, memsize);
#else
  membase = (void*)malloc(memsize+alignment);
  if ((int)membase & (alignment-1))
    membase = (void*)(((int)membase & ~(alignment-1))+alignment);
#endif

#if !defined(ARMZYNQ) && defined(EMAX5)
  emax_info.hpp_phys = membase;
  emax_info.hpp_mmap = emax_info.hpp_phys;
  emax_info.acp_phys = ACP_BASE2_PHYS; /* defined in emax5lib.h >= ALOCLIMIT */
  emax_info.acp_mmap = emax_info.acp_phys;
#endif
#if defined(EMAX5)
  acp_conf = emax_info.acp_mmap; /* 8KB * 256sets */
  acp_lmmi = emax_info.acp_mmap + 0x200000;
  acp_regv = emax_info.acp_mmap + 0x304000;
#endif

#if !defined(ARMZYNQ) && defined(EMAX6)
  emax_info.dma_phys = DMA_BASE2_PHYS; /* defined in emax6lib.h */
  emax_info.dma_mmap = emax_info.dma_phys;
  emax_info.reg_phys = REG_BASE2_PHYS; /* defined in emax6lib.h */
  emax_info.reg_mmap = emax_info.reg_phys;
  emax_info.lmm_phys = LMM_BASE2_PHYS;
  emax_info.lmm_mmap = emax_info.lmm_phys;
  emax_info.ddr_phys = membase;
  emax_info.ddr_mmap = emax_info.ddr_phys;
#endif
#if (defined(ARMSIML) || defined(ARMZYNQ)) && defined(EMAX6)
  emax6.dma_ctrl  = emax_info.dma_mmap;
  emax6.reg_ctrl  = emax_info.reg_mmap;
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].cmd = CMD_RESET;  // ★★★ RESET
#if defined(ARMZYNQ)
  usleep(1);
#endif
  switch (((struct reg_ctrl*)emax6.reg_ctrl)->i[0].stat>>8 & 0xf) {
  case  3:EMAX_DEPTH = 64;break;
  case  2:EMAX_DEPTH = 32;break;
  case  1:EMAX_DEPTH = 16;break;
  default:EMAX_DEPTH =  8;break;
  }
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].adtr = emax_info.ddr_mmap - emax_info.lmm_phys;
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].dmrp = 0LL;
#endif
#if (defined(ARMSIML) || defined(ARMZYNQ)) && defined(EMAX7)
  emax7[0].dma_ctrl  = emax_info[0].dma_mmap;
  emax7[0].reg_ctrl  = emax_info[0].reg_mmap;
  ((struct reg_ctrl*)emax7[0].reg_ctrl)->i[0].cmd = CMD_RESET;  // ★★★ RESET
#if defined(ARMZYNQ)
  usleep(1);
#endif
  switch (((struct reg_ctrl*)emax7[0].reg_ctrl)->i[0].stat>>8 & 0xf) {
  case  3:EMAX_DEPTH = 64;break;
  case  2:EMAX_DEPTH = 32;break;
  case  1:EMAX_DEPTH = 16;break;
  default:EMAX_DEPTH =  8;break;
  }
  ((struct reg_ctrl*)emax7[0].reg_ctrl)->i[0].adtr = emax_info[0].ddr_mmap - emax_info[0].lmm_phys;
  ((struct reg_ctrl*)emax7[0].reg_ctrl)->i[0].dmrp = 0LL;
#endif
}

#define IC    18
//IMAP should be specified in Makefile* to share src for emax6/emax7.
//#define IMAP  6  for 64unit
//#define IMAP  3  for 32unit
#define IMAP 3

#define OC    16
#define M     242
#define RMGRP 16
//NCHIP should be specified in Makefile* to share src for emax6/emax7.
//#define NCHIP 4 for emax6
//#define NCHIP 1 for emax7
#define NCHIP 1

#define K     3
#define W     4
Uint *in;  /*[IC*M*M];*/
Uint *ker; /*[IC*OC*K*K];*/
Uint *out0;/*[OC*M*M];*/
Uint *out1;/*[OC*M*M];*/
Uint *ip0, *ip1, *ip2, *ip3, *ip4, *ip5, *kp, *op;
int ic, row, col, oc, y, x;
int top, iset;
int w, kidx;
int count0, count1, count2;

#define CSIMWD 320
#define CSIMHT 240
#define CSIMBM (CSIMWD*CSIMHT)
Uint Z[CSIMBM];

#define MAXINT (~(1<<(sizeof(int)*8-1)))
#define adif(a,b) (((a)>(b))?(a)-(b):(b)-(a))
#define dif(a,b)  (adif((((a)>>24)&255), (((b)>>24)&255))\
                  +adif((((a)>>16)&255), (((b)>>16)&255))\
                  +adif((((a)>> 8)&255), (((b)>> 8)&255)))
#define abs(a) (((a)<0)?-(a):(a))





















































//時間計測用---------
#define timeMeasureQuantity 128

char timeMeasureSectionName[timeMeasureQuantity][32];
int timeMeasureSectionCount = 0;
clock_t timeMeasureResult[timeMeasureQuantity*2];

void timeMeasureStart(char *section){
  strcpy(timeMeasureSectionName[timeMeasureSectionCount], section);
  timeMeasureResult[timeMeasureSectionCount*2] = clock();
}

void timeMeasureEnd(){
  timeMeasureResult[timeMeasureSectionCount*2 + 1] = clock();
  printf("%s: %lf (ms)\n", timeMeasureSectionName[timeMeasureSectionCount],
  (double)(timeMeasureResult[timeMeasureSectionCount*2+1] - timeMeasureResult[timeMeasureSectionCount*2]) / CLOCKS_PER_SEC * 1000.0);
  timeMeasureSectionCount++;
}

void timeMeasurePrint(){
  printf("---RESULT---\n");
  printf("Time measure Total %d\n", timeMeasureSectionCount);
  for (int i = 0; i < timeMeasureSectionCount; i++){
    printf("%s: %lf ms\n", timeMeasureSectionName[i], (double)(timeMeasureResult[i*2+1] - timeMeasureResult[i*2]) / CLOCKS_PER_SEC * 1000.0);
  }
}

void timeNow(){
    char buf[20];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&(time_t){time(NULL)}));
    printf("%s ", buf);
}
//時間計測用end-------















typedef struct {
    int num;
    int channel;
    int depth;
    int height;
    int width;
    float *data; // vector
} tensor;

tensor create_tensor(int num, int channel, int depth, int height, int width) {
    tensor result;
    result.num = num;
    result.channel = channel;
    result.depth = depth;
    result.height = height;
    result.width = width;
    result.data = (float *)calloc(num * channel * depth * height * width, sizeof(float));
    return result;
}

int get_tensor_index(tensor t, int n, int c, int d, int h, int w) {
    return n * t.channel * t.depth * t.height * t.width +
           c * t.depth * t.height * t.width +
           d * t.height * t.width +
           h * t.width +
           w;
}

tensor create_tensor_from_file(const char *filename, int num, int channel, int depth, int height, int width) {
    tensor result = create_tensor(num, channel, depth, height, width);

    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Unable to open file %s\n", filename);
        exit(1);
    }

    for (int n = 0; n < num; n++) {
        for (int c = 0; c < channel; c++) {
            for (int d = 0; d < depth; d++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        if (fscanf(file, "%f", &result.data[((n * channel + c) * depth + d) * height * width + h * width + w]) != 1) {
                            printf("Error: Not enough values in file %s\n", filename);
                            fclose(file);
                            exit(1);
                        }
                    }
                }
            }
        }
    }

    fclose(file);
    return result;
}

void print_tensor(tensor t, const char *tensor_name) {
    printf("%s:\n", tensor_name);
    for (int n = 0; n < t.num; n++) {
        for (int c = 0; c < t.channel; c++) {
            for (int d = 0; d < t.depth; d++) {
                for (int h = 0; h < t.height; h++) {
                    for (int w = 0; w < t.width; w++) {
                        printf("%f ", t.data[get_tensor_index(t, n, c, d, h, w)]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

void write_tensor_to_file(const char *file_name, tensor t) {
    FILE *file = fopen(file_name, "w");
    if (file == NULL) {
        printf("Error opening file %s\n", file_name);
        return;
    }

    for (int n = 0; n < t.num; n++) {
        for (int c = 0; c < t.channel; c++) {
            for (int d = 0; d < t.depth; d++) {
                for (int h = 0; h < t.height; h++) {
                    for (int w = 0; w < t.width; w++) {
                        fprintf(file, "%f ", t.data[get_tensor_index(t, n, c, d, h, w)]);
                    }
                    fprintf(file, "\n");
                }
                fprintf(file, "\n");
            }
            fprintf(file, "\n");
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

void save_tensor_to_npy(tensor t, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Unable to open file %s\n", filename);
        exit(1);
    }

    // Write the numpy magic number
    const uint8_t magic_number[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    fwrite(magic_number, sizeof(uint8_t), 6, file);

    // Write the numpy version (1.0)
    const uint8_t version[] = {0x01, 0x00};
    fwrite(version, sizeof(uint8_t), 2, file);

    // Write the header length (uint16_t)
    uint16_t header_length = 128;  // Allocate enough space for a typical header
    fwrite(&header_length, sizeof(uint16_t), 1, file);

    // Write the header
    char header[header_length];
    snprintf(header, header_length, "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d, %d, %d, %d), }",
             t.num, t.channel, t.depth, t.height, t.width);
    size_t padding = header_length - strlen(header) - 1;
    memset(header + strlen(header), ' ', padding);
    header[header_length - 1] = '\n';
    fwrite(header, sizeof(char), header_length, file);

    // Write the data
    fwrite(t.data, sizeof(float), t.num * t.channel * t.depth * t.height * t.width, file);

    fclose(file);
}

tensor load_tensor_from_npy(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Unable to open file %s\n", filename);
        exit(1);
    }

    // Read the numpy magic number
    uint8_t magic_number[6];
    fread(magic_number, sizeof(uint8_t), 6, file);
    if (memcmp(magic_number, "\x93NUMPY", 6) != 0) {
        printf("Error: File %s is not a numpy file\n", filename);
        exit(1);
    }

    // Read the numpy version (1.0)
    uint8_t version[2];
    fread(version, sizeof(uint8_t), 2, file);
    if (version[0] != 1 || version[1] != 0) {
        printf("Error: Unsupported numpy version %d.%d\n", version[0], version[1]);
        exit(1);
    }

    // Read the header length (uint16_t)
    uint16_t header_length;
    fread(&header_length, sizeof(uint16_t), 1, file);

    // Read the header
    char *header = (char *)malloc(header_length);
    fread(header, sizeof(char), header_length, file);

    // Parse the header
    int num, channel, depth, height, width;
    if (sscanf(header, "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d, %d, %d, %d), }",
               &num, &channel, &depth, &height, &width) != 5) {
        printf("Error: Unable to parse header of file %s\n", filename);
        exit(1);
    }

    // Read the data
    tensor result = create_tensor(num, channel, depth, height, width);
    fread(result.data, sizeof(float), num * channel * depth * height * width, file);

    fclose(file);
    free(header);
    return result;
}

tensor conv3d(tensor input, tensor kernels, tensor bias, int stride, int padding) {
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
        for (int k = 0; k < output.channel; k++) {
            for (int d = 0; d < output_depth; d++) {
                for (int h = 0; h < output_height; h++) {
                    for (int w = 0; w < output_width; w++) {
                        float value = 0;
                        for (int c = 0; c < input.channel; c++) {
                            for (int fd = 0; fd < kernels.depth; fd++) {
                                for (int fh = 0; fh < kernels.height; fh++) {
                                    for (int fw = 0; fw < kernels.width; fw++) {
                                        int d_in = d * stride - padding + fd;
                                        int h_in = h * stride - padding + fh;
                                        int w_in = w * stride - padding + fw;

                                        if (d_in >= 0 && d_in < input.depth &&
                                            h_in >= 0 && h_in < input.height &&
                                            w_in >= 0 && w_in < input.width) {
                                            int t_idx = get_tensor_index(input, n, c, d_in, h_in, w_in);
                                            int f_idx = get_tensor_index(kernels, k, c, fd, fh, fw);
                                            value += input.data[t_idx] * kernels.data[f_idx];
                                        }
                                    }
                                }
                            }
                        }
                        value += bias.data[k]; // Add the bias term
                        output.data[get_tensor_index(output, n, k, d, h, w)] = value;
                    }
                }
            }
        }
    }

    return output;
}




void test_conv3d_17(){
  // conv17 (in=256, out=128, kernel=3, stride=1, padding=1), _____________________________________________
  // create kernel18 tensor, shape (128,256,3,3,3)// read from file
  tensor concat14_out = load_tensor_from_npy("./INPUT/conv3d_1-35.npy");
  puts("input loaded");

  int num17 = 128;int channel17 = 256;int depth17 = 3;int height17 = 3;int width17 = 3;
  const char *filename34 = "./INPUT/35.txt";
  tensor kernel18 = create_tensor_from_file(filename34, num17, channel17, depth17, height17, width17);
  puts("kernel loaded");

  int num_b17 = 1;int channel_b17 = 128;int depth_b17 = 1;int height_b17 = 1;int width_b17 = 1;
  const char *filename35 = "./INPUT/34.txt";
  tensor bias18 = create_tensor_from_file(filename35, num_b17, channel_b17, depth_b17, height_b17, width_b17);
  puts("bias loaded");

  int stride14 = 1; int padding18 = 1;
  puts("conv3d start");
  timeMeasureStart("conv3d");
  tensor conv3d_17_out = conv3d(concat14_out, kernel18, bias18, stride14, padding18);
  timeMeasureEnd();
  puts("conv3d end");
  save_tensor_to_npy(conv3d_17_out, "./OUTPUT/conv3d_17_out.npy");
  // tensor conv3d_17_out_relu = leakyrelu(conv3d_17_out, 0.01);
  // save_tensor_to_npy(conv3d_17_out_relu, "conv3d_17_out_relu.npy"); // okkk, mse=7.402755e-14
}






















main()
{
  #ifndef ARMSIML
  printf("ARMSIML is not defined\n");
  #endif

  // test_conv3d_17();
  // timeMeasurePrint();
  // return 0;







  sysinit(IC*M*M*sizeof(int)
         +IC*OC*K*K*sizeof(int)
         +OC*M*M*sizeof(int)
         +OC*M*M*sizeof(int),32);
  //print IC*M*M*sizeof(int)
  printf("IC*M*M*sizeof(int): %d\n", IC*M*M*sizeof(int));
  printf("IC*OC*K*K*sizeof(int): %d\n", IC*OC*K*K*sizeof(int));
  printf("OC*M*M*sizeof(int): %d\n", OC*M*M*sizeof(int));
  printf("IC*M*M*sizeof(int) +IC*OC*K*K*sizeof(int) +OC*M*M*sizeof(int) +OC*M*M*sizeof(int): %d Bytes\n", IC*M*M*sizeof(int) +IC*OC*K*K*sizeof(int) +OC*M*M*sizeof(int) +OC*M*M*sizeof(int));
  printf("IC*M*M*sizeof(int) +IC*OC*K*K*sizeof(int) +OC*M*M*sizeof(int) +OC*M*M*sizeof(int): %f MB\n", (IC*M*M*sizeof(int) +IC*OC*K*K*sizeof(int) +OC*M*M*sizeof(int) +OC*M*M*sizeof(int))/1024.0/1024.0);


  printf("membase: %08.8x\n", (Uint)membase);
  in   = (Uint*)membase;
  ker  = (Uint*)((Uchar*)in   + IC*M*M*sizeof(int));
  out0 = (Uint*)((Uchar*)ker  + IC*OC*K*K*sizeof(int));
  out1 = (Uint*)((Uchar*)out0 + OC*M*M*sizeof(int));
  printf("in  : %08.8x\n", in);
  printf("ker : %08.8x\n", ker);
  printf("out0: %08.8x\n", out0);
  printf("out1: %08.8x\n", out1);

  for (ic=0; ic<IC; ic++) {
    // add for depth loop
    for (row=0; row<M; row++) {
      for (col=0; col<M; col++) {
        // *(float*)&in[ic*M*M+row*M+col] = ic<<12|(((M/2-abs(row-M/2))*(M/2-abs(col-M/2)))&0xfff);
        *(float*)&in[ic*M*M+row*M+col] = 1;//test
      }
    }
  }
  for (oc=0; oc<OC; oc++) {
    for (ic=0; ic<IC; ic++) {
      // for depth
      for (y=0; y<K; y++) {
        for (x=0; x<K; x++) {
          *(float*)&ker[ic*OC*K*K+oc*K*K+y*K+x] = (oc-ic)*((2-abs(y-K/2))*(2-abs(x-K/2)))/OC;
        }
      }
    }
  }

#if !defined(ARMSIML)
  x11_open(0);
#endif

  // reset_nanosec();
  // orig();
  // get_nanosec(0);
  // show_nanosec();

  reset_nanosec();
  imax();
  get_nanosec(0);
  show_nanosec();

#ifdef ARMSIML
  copy_Z(0, out1); _copyX(0, Z);
  copy_Z(1, out1); _copyX(1, Z);
  copy_Z(2, out1); _copyX(2, Z);
  copy_Z(3, out1); _copyX(3, Z);
  copy_Z(5, out1); _copyX(4, Z);
  copy_Z(6, out1); _copyX(5, Z);
  copy_Z(7, out1); _copyX(6, Z);
  copy_Z(8, out1); _copyX(7, Z);
  copy_Z(10,out1); _copyX(8 ,Z);
  copy_Z(11,out1); _copyX(9 ,Z);
  copy_Z(12,out1); _copyX(10,Z);
  copy_Z(13,out1); _copyX(11,Z);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_Z(0, out1); BGR_to_X(0, Z);
  copy_Z(1, out1); BGR_to_X(1, Z);
  copy_Z(2, out1); BGR_to_X(2, Z);
  copy_Z(3, out1); BGR_to_X(3, Z);
  copy_Z(4, out1); BGR_to_X(4, Z);
  copy_Z(5, out1); BGR_to_X(5, Z);
  copy_Z(6, out1); BGR_to_X(6, Z);
  copy_Z(7, out1); BGR_to_X(7, Z);
  copy_Z(8, out1); BGR_to_X(8, Z);
  copy_Z(9 ,out1); BGR_to_X(9, Z);
  copy_Z(10,out1); BGR_to_X(10,Z);
  copy_Z(11,out1); BGR_to_X(11,Z);
  copy_Z(12,out1); BGR_to_X(12,Z);
  copy_Z(13,out1); BGR_to_X(13,Z);
  copy_Z(14,out1); BGR_to_X(14,Z);
  copy_Z(15,out1); BGR_to_X(15,Z);
  x11_update();
#endif

  printf("Num of MULT: orig=%d imax=%d\n", count0, count1);

  for (oc=0; oc<OC; oc++) {
    for (row=1; row<M-1; row++) {
      for (col=0; col<M-2; col++) {
        if (out0[oc*M*M+row*M+col] != out1[oc*M*M+row*M+col]) {
          count2++;
          // printf("o0[%d]=%f o1[%d]=%f\n",
          //        oc*M*M+row*M+col, (double)*(float*)&out0[oc*M*M+row*M+col],
          //        oc*M*M+row*M+col, (double)*(float*)&out1[oc*M*M+row*M+col]);
        }
      }
    }
  }
  if (count2)
    printf("Num of diffs: %d\n", count2);
  else
    printf("Results are equal\n");

  show_nanosec();

#if !defined(ARMSIML)
  printf("==== Normal end. Type any in ImageWin ====\n");
  while (!x11_checkevent());
#endif
}

copy_Z(id, from)
     int id; /* 0 .. 11 */
     unsigned int *from;
{
  int i, j;
  volatile unsigned int *to = Z;

  switch (id) {
  case 0:                   break;
  case 1:  from += M*M;     break;
  case 2:  from += M*M*2;   break;
  case 3:  from += M*M*3;   break;
  case 4:  from += M*M*4;   break;
  case 5:  from += M*M*5;   break;
  case 6:  from += M*M*6;   break;
  case 7:  from += M*M*7;   break;
  case 8:  from += M*M*8;   break;
  case 9:  from += M*M*9;   break;
  case 10: from += M*M*10;  break;
  case 11: from += M*M*11;  break;
  case 12: from += M*M*12;  break;
  case 13: from += M*M*13;  break;
  case 14: from += M*M*14;  break;
  case 15: from += M*M*15;  break;
  }
  for (i=0; i<HT; i++, from+=M) {
    if (i<M) {
      for (j=0; j<WD; j++) {
        if (j<M) *to++ = (*(from+j))<<2;
        else     *to++ = 0;
      }
    }
    else {
      for (j=0; j<WD; j++)
        *to++ = 0;
    }
  }
}

orig() {
  printf("<<<ORIG>>>\n");
  for (ic=0; ic<IC; ic++) { /* set input channel */
    ip0 = &in[ic*M*M]; /* top of input */
    for (row=1; row<M-1; row++) { /* image loop */
      for (col=0; col<M-2; col++) {
        for (oc=0; oc<OC; oc++) { /* set output channel */
          op = &out0[oc*M*M+row*M+col]; /* top of output */
          kp = &ker[(oc*IC+ic)*K*K];
          kidx = 0;
          for (y=-((K-1)/2); y<=(K-1)/2; y++) { /* kernel loop */
            for (x=-((K-1)/2); x<=(K-1)/2; x++) {
              if (ic == 0 && kidx == 0) {
                *(float*)&*op  = *(float*)&ip0[(row+y)*M+col+x+1] * *(float*)&kp[kidx];
                /*printf("head [%d %d %d][%d %d %d] out0[%d]=%d\n", ic, row, col, oc, y, x, op-&out0[0], *op);*/
              }
              else {
                *(float*)&*op += *(float*)&ip0[(row+y)*M+col+x+1] * *(float*)&kp[kidx];
                /*printf("     [%d %d %d][%d %d %d] out0[%d]=%d\n", ic, row, col, oc, y, x, op-&out0[0], *op);*/
              }
              kidx++;
              count0++;
            }
          }
        }
      }
    }
  }
}

#if 0
imax() {
  Ull CHIP;
  Ull rofs;
  printf("<<<IMAX>>>\n");
  for (top=1; top<M-1; top+=RMGRP) {
    for (iset=0; iset<IC; iset+=IMAP) { /* accumulate multiple sets of IC */
      for (oc=0; oc<OC/NCHIP; oc+=W) { /* set output channel */
  /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
    /*2*/ for (rofs=0; rofs<RMGRP; rofs++) { /* image loop (row) */
      /*1*/ for (col=0; col<M-2; col++) { /* image loop (col) */
              for (w=0; w<W; w++) { /* set output channel */
                op = &out1[(CHIP*OC/NCHIP+oc+w)*M*M+(top+rofs)*M+col]; /* top of output */
                for (ic=0; ic<IMAP; ic++) { /* set offset of input channel */
                  ip0 = &in[(iset+ic)*M*M]; /* top of input */
                  kp = &ker[((CHIP*OC/NCHIP+oc+w)*IC+iset+ic)*K*K];
                  kidx = 0;
                  for (y=-((K-1)/2); y<=(K-1)/2; y++) { /* kernel loop */
                    for (x=-((K-1)/2); x<=(K-1)/2; x++) {
                      if (iset == 0 && ic == 0 && kidx == 0) {
                        *(float*)&*op  = *(float*)&ip0[(top+rofs+y)*M+col+x] * *(float*)&kp[kidx];
                        /*printf("head [%d %d %d][%d %d %d] out1[%d]=%d\n", ic, row, col, (oc+w), y, x, op-&out1[0], *op);*/
                      }
                      else {
                        *(float*)&*op += *(float*)&ip0[(top+rofs+y)*M+col+x] * *(float*)&kp[kidx];
                        /*printf("     [%d %d %d][%d %d %d] out1[%d]=%d\n", ic, row, col, (oc+w), y, x, op-&out1[0], *op);*/
                      }
                      kidx++;
                      count1++;
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

#else

imax() {
  Ull  CHIP;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  Ull  cofs, rofs, oofs, k;
  /*  ┌─────┐convolutionの場合   ┌─────┐                             */
  /*  │┌────┴┐Bが複数と考える     │┌────┴┐                           */
  /*  ││┌────┴┐┌─────┐┐          ││┌────┴┐┐                       */
  /*  │││iii         ││k k k k k ││RMGRP │││          ││RMGRP                  */
  /*  │││iii         ┤│          │┤      │││o o o o o │┤                       */
  /*  │││iii in      ││ k(weight)││      │││   out    ││ mmの場合は行で分割    */
  /*  └││            ┤│          │┤      └││          │┤ cnnの場合はoutで分割  */
  /*   └│            ││          ││       └│          ││                       */
  /*    └─────┘└─────┘┘          └─────┘┘                       */
  printf("<<<IMAX>>>\n");
  for (top=1; top<M-1; top+=RMGRP) {
    for (iset=0; iset<IC; iset+=IMAP) { /* accumulate multiple sets of IC */
      Uint *ip[IMAP], *it[IMAP], *ip0[IMAP][K*K], *ip1[IMAP][K*K];
      /* IMAP likely represents the number of input channels being processed simultaneously */
      for (k=0; k<IMAP; k++) {
        // printf("iset=%d k=%d ip=%p\n", iset, k, ip[k]);//ren-im
        ip[k] = &in[(iset+k)*M*M]; /* top of input#0-5 */
        it[k] = ip[k]+(top-1)*M+1-1;
        /* the code calculates pointers to the input tensor and its corresponding positions in two 3x3 grids (ip0[k] and ip1[k]).*/
        ip0[k][0] = ip[k]+(top-1)*M+1-1; ip0[k][1] = ip[k]+(top-1)*M+1+0; ip0[k][2] = ip[k]+(top-1)*M+1+1;
        ip0[k][3] = ip[k]+(top+0)*M+1-1; ip0[k][4] = ip[k]+(top+0)*M+1+0; ip0[k][5] = ip[k]+(top+0)*M+1+1;
        ip0[k][6] = ip[k]+(top+1)*M+1-1; ip0[k][7] = ip[k]+(top+1)*M+1+0; ip0[k][8] = ip[k]+(top+1)*M+1+1;
        
        ip1[k][0] = ip[k]+(top-1)*M+1+1; ip1[k][1] = ip[k]+(top-1)*M+1+2; ip1[k][2] = ip[k]+(top-1)*M+1+3;
        ip1[k][3] = ip[k]+(top+0)*M+1+1; ip1[k][4] = ip[k]+(top+0)*M+1+2; ip1[k][5] = ip[k]+(top+0)*M+1+3;
        ip1[k][6] = ip[k]+(top+1)*M+1+1; ip1[k][7] = ip[k]+(top+1)*M+1+2; ip1[k][8] = ip[k]+(top+1)*M+1+3;
      }
      for (oc=0; oc<OC/NCHIP; oc+=W) { /* set output channel */
        Uint *kp0[IMAP][NCHIP], *kp1[IMAP][NCHIP], *kp2[IMAP][NCHIP], *kp3[IMAP][NCHIP];
        /* 4 output channels in parallel (op0, op1, op2, op3) */
        Uint *op0[NCHIP], *op1[NCHIP], *op2[NCHIP], *op3[NCHIP];
        Uint *ot0[NCHIP], *ot1[NCHIP], *ot2[NCHIP], *ot3[NCHIP];
        for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
          Uint choc  = CHIP*OC/NCHIP+oc; // if CHIP=0,NCHIP=1, then choc = oc
          for (k=0; k<IMAP; k++) {
            kp0[k][CHIP] = ker+((choc+0)*IC+iset+k)*K*K;
            kp1[k][CHIP] = ker+((choc+1)*IC+iset+k)*K*K;
            kp2[k][CHIP] = ker+((choc+2)*IC+iset+k)*K*K;
            kp3[k][CHIP] = ker+((choc+3)*IC+iset+k)*K*K;
          }
          op0[CHIP] = out1+(choc+0)*M*M+top*M+0; op1[CHIP] = out1+(choc+1)*M*M+top*M+0; op2[CHIP] = out1+(choc+2)*M*M+top*M+0; op3[CHIP] = out1+(choc+3)*M*M+top*M+0;
          ot0[CHIP] = out1+(choc+0)*M*M+top*M+0; ot1[CHIP] = out1+(choc+1)*M*M+top*M+0; ot2[CHIP] = out1+(choc+2)*M*M+top*M+0; ot3[CHIP] = out1+(choc+3)*M*M+top*M+0;
        }

#define cnn_core1(r, i, ofs, k, rp1) \
        mop(OP_LDWR,   1, &BR[r][0][1],  (Ull)kp0[i][CHIP], ofs, MSK_D0, (Ull)ker, IC*OC*K*K, 0, 0, (Ull)NULL, IC*OC*K*K);\
        mop(OP_LDWR,   1, &BR[r][0][0],  (Ull)kp1[i][CHIP], ofs, MSK_D0, (Ull)ker, IC*OC*K*K, 0, 0, (Ull)NULL, IC*OC*K*K);\
        mop(OP_LDWR,   1, &BR[r][1][1],  (Ull)kp2[i][CHIP], ofs, MSK_D0, (Ull)ker, IC*OC*K*K, 0, 0, (Ull)NULL, IC*OC*K*K);\
        mop(OP_LDWR,   1, &BR[r][1][0],  (Ull)kp3[i][CHIP], ofs, MSK_D0, (Ull)ker, IC*OC*K*K, 0, 0, (Ull)NULL, IC*OC*K*K);\
        mop(OP_LDR,    1, &BR[r][2][1],  (Ull)ip1[i][k], oofs, MSK_W0, (Ull)it[i], M*(RMGRP+2), 0, 0, (Ull)NULL, M*(RMGRP+2));\
        mop(OP_LDR,    1, &BR[r][2][0],  (Ull)ip0[i][k], oofs, MSK_W0, (Ull)it[i], M*(RMGRP+2), 0, 0, (Ull)NULL, M*(RMGRP+2));\
        exe(OP_FMA, &AR[rp1][0], AR[r][0], EXP_H3210, BR[r][2][0], EXP_H3210, BR[r][0][1], EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
        exe(OP_FMA, &AR[rp1][1], AR[r][1], EXP_H3210, BR[r][2][0], EXP_H3210, BR[r][0][0], EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
        exe(OP_FMA, &AR[rp1][2], AR[r][2], EXP_H3210, BR[r][2][0], EXP_H3210, BR[r][1][1], EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL);\
        exe(OP_FMA, &AR[rp1][3], AR[r][3], EXP_H3210, BR[r][2][0], EXP_H3210, BR[r][1][0], EXP_H1010, OP_NOP, 0LL, OP_NOP, 0LL)
        // exe(FMA) usage:(Ull*)d|s1, ((Ull*)s1, (Ull*)s2, (Ull*)s3); mode:32bit*2*4 3in; description: floating-point s1+s2*s3;
        // mop(LDR) usage:(-, (Ull*)d, base(++), offs, msk, top, len, blk, force-read, ptop, plen); mode:64bit lmm; description: LMM rand-access;
        /*Overall, the cnn_core1 macro function loads weight kernels and input data into buffers and then computes the FMA operations for the convolution. This function is designed to be used in a larger context, 
        where it would be called multiple times with different parameters to perform the full CNN computation.
        */

#define cnn_final(r, rp1) \
        mop(OP_LDR,  1, &BR[rp1][0][1],  (Ull)op0[CHIP], oofs, MSK_W0, (Ull)ot0[CHIP], M*RMGRP, 0, 1, (Ull)NULL, M*RMGRP);\
        mop(OP_LDR,  1, &BR[rp1][1][1],  (Ull)op1[CHIP], oofs, MSK_W0, (Ull)ot1[CHIP], M*RMGRP, 0, 1, (Ull)NULL, M*RMGRP);\
        mop(OP_LDR,  1, &BR[rp1][2][1],  (Ull)op2[CHIP], oofs, MSK_W0, (Ull)ot2[CHIP], M*RMGRP, 0, 1, (Ull)NULL, M*RMGRP);\
        mop(OP_LDR,  1, &BR[rp1][3][1],  (Ull)op3[CHIP], oofs, MSK_W0, (Ull)ot3[CHIP], M*RMGRP, 0, 1, (Ull)NULL, M*RMGRP);\
        exe(OP_FAD, &AR[rp1][0], AR[r][0], EXP_H3210, BR[rp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
        exe(OP_FAD, &AR[rp1][1], AR[r][1], EXP_H3210, BR[rp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
        exe(OP_FAD, &AR[rp1][2], AR[r][2], EXP_H3210, BR[rp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
        exe(OP_FAD, &AR[rp1][3], AR[r][3], EXP_H3210, BR[rp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
        mop(OP_STR,  3, &AR[rp1][0], oofs, (Ull)op0[CHIP], MSK_D0, (Ull)ot0[CHIP], M*RMGRP, 0, 1, (Ull)NULL, M*RMGRP);\
        mop(OP_STR,  3, &AR[rp1][1], oofs, (Ull)op1[CHIP], MSK_D0, (Ull)ot1[CHIP], M*RMGRP, 0, 1, (Ull)NULL, M*RMGRP);\
        mop(OP_STR,  3, &AR[rp1][2], oofs, (Ull)op2[CHIP], MSK_D0, (Ull)ot2[CHIP], M*RMGRP, 0, 1, (Ull)NULL, M*RMGRP);\
        mop(OP_STR,  3, &AR[rp1][3], oofs, (Ull)op3[CHIP], MSK_D0, (Ull)ot3[CHIP], M*RMGRP, 0, 1, (Ull)NULL, M*RMGRP)
        // exe(FAD) usage: (Ull*)d|s1, ((Ull*)s1, (Ull*)s2, -); mode: 32bit*2*4 2in; description: floating-point s1+s2;
        // AR[rp1][0] = AR[r][0] + BR[rp1][0][1]
        /*
        Overall, the cnn_final macro function loads the output data, adds the result of the CNN computation stored in the accumulator registers, 
        and then stores the final output back to memory. This function is designed to be used in a larger context, 
        typically after the cnn_core1 function has been called multiple times to perform the full CNN computation.
        */

//EMAX5A begin cnn mapdist=0
  /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
    /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=0-M*4; LOOP1--; INIT1=0) {            /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
      /*1*/ for (INIT0=1,LOOP0=(M-2)/2,cofs=0-8; LOOP0--; INIT0=0) {          /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
              exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, 8, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);/* stage#0 */
              exe(OP_ADD,    &rofs, rofs, EXP_H3210, INIT0?M*4:0,  EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#0 */
              exe(OP_ADD,    &oofs, rofs, EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);  /* stage#1 */

              mop(OP_LDWR,   1, &BR[2][0][1],  (Ull)kp0[0][CHIP], 0LL, MSK_D0, (Ull)ker, IC*OC*K*K, 0, 0, (Ull)NULL, IC*OC*K*K); /* stage#2 */
              mop(OP_LDWR,   1, &BR[2][0][0],  (Ull)kp1[0][CHIP], 0LL, MSK_D0, (Ull)ker, IC*OC*K*K, 0, 0, (Ull)NULL, IC*OC*K*K); /* stage#2 */
              mop(OP_LDWR,   1, &BR[2][1][1],  (Ull)kp2[0][CHIP], 0LL, MSK_D0, (Ull)ker, IC*OC*K*K, 0, 0, (Ull)NULL, IC*OC*K*K); /* stage#2 */
              mop(OP_LDWR,   1, &BR[2][1][0],  (Ull)kp3[0][CHIP], 0LL, MSK_D0, (Ull)ker, IC*OC*K*K, 0, 0, (Ull)NULL, IC*OC*K*K); /* stage#2 10KB */
              mop(OP_LDR,    1, &BR[2][2][1],  (Ull)ip1[0][0], oofs, MSK_W0, (Ull)it[0], M*(RMGRP+2), 0, 0, (Ull)NULL, M*(RMGRP+2)); /* stage#2 8KB *//* unaligned load */
              mop(OP_LDR,    1, &BR[2][2][0],  (Ull)ip0[0][0], oofs, MSK_W0, (Ull)it[0], M*(RMGRP+2), 0, 0, (Ull)NULL, M*(RMGRP+2)); /* stage#2 8KB *//* unaligned load */
              exe(OP_FML, &AR[3][0], BR[2][2][0], EXP_H3210, BR[2][0][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
              exe(OP_FML, &AR[3][1], BR[2][2][0], EXP_H3210, BR[2][0][0], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
              exe(OP_FML, &AR[3][2], BR[2][2][0], EXP_H3210, BR[2][1][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
              exe(OP_FML, &AR[3][3], BR[2][2][0], EXP_H3210, BR[2][1][0], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */

              cnn_core1( 3, 0,  4LL, 1,  4);
              cnn_core1( 4, 0,  8LL, 2,  5);
              cnn_core1( 5, 0, 12LL, 3,  6);
              cnn_core1( 6, 0, 16LL, 4,  7);
              cnn_core1( 7, 0, 20LL, 5,  8);
              cnn_core1( 8, 0, 24LL, 6,  9);
              cnn_core1( 9, 0, 28LL, 7, 10);
              cnn_core1(10, 0, 32LL, 8, 11);

              cnn_core1(11, 1,  0LL, 0, 12);
              cnn_core1(12, 1,  4LL, 1, 13);
              cnn_core1(13, 1,  8LL, 2, 14);
              cnn_core1(14, 1, 12LL, 3, 15);
              cnn_core1(15, 1, 16LL, 4, 16);
              cnn_core1(16, 1, 20LL, 5, 17);
              cnn_core1(17, 1, 24LL, 6, 18);
              cnn_core1(18, 1, 28LL, 7, 19);
              cnn_core1(19, 1, 32LL, 8, 20);

              cnn_core1(20, 2,  0LL, 0, 21);
              cnn_core1(21, 2,  4LL, 1, 22);
              cnn_core1(22, 2,  8LL, 2, 23);
              cnn_core1(23, 2, 12LL, 3, 24);
              cnn_core1(24, 2, 16LL, 4, 25);
              cnn_core1(25, 2, 20LL, 5, 26);
              cnn_core1(26, 2, 24LL, 6, 27);
              cnn_core1(27, 2, 28LL, 7, 28);
              cnn_core1(28, 2, 32LL, 8, 29);
#if (IMAP==3)
	      cnn_final(29,     30);
#endif
#if (IMAP>3)
	      cnn_core1(29, 3,  0LL, 0, 30);
	      cnn_core1(30, 3,  4LL, 1, 31);
	      cnn_core1(31, 3,  8LL, 2, 32);
	      cnn_core1(32, 3, 12LL, 3, 33);
	      cnn_core1(33, 3, 16LL, 4, 34);
	      cnn_core1(34, 3, 20LL, 5, 35);
	      cnn_core1(35, 3, 24LL, 6, 36);
	      cnn_core1(36, 3, 28LL, 7, 37);
	      cnn_core1(37, 3, 32LL, 8, 38);

	      cnn_core1(38, 4,  0LL, 0, 39);
	      cnn_core1(39, 4,  4LL, 1, 40);
	      cnn_core1(40, 4,  8LL, 2, 41);
	      cnn_core1(41, 4, 12LL, 3, 42);
	      cnn_core1(42, 4, 16LL, 4, 43);
	      cnn_core1(43, 4, 20LL, 5, 44);
	      cnn_core1(44, 4, 24LL, 6, 45);
	      cnn_core1(45, 4, 28LL, 7, 46);
	      cnn_core1(46, 4, 32LL, 8, 47);

	      cnn_core1(47, 5,  0LL, 0, 48);
	      cnn_core1(48, 5,  4LL, 1, 49);
	      cnn_core1(49, 5,  8LL, 2, 50);
	      cnn_core1(50, 5, 12LL, 3, 51);
	      cnn_core1(51, 5, 16LL, 4, 52);
	      cnn_core1(52, 5, 20LL, 5, 53);
	      cnn_core1(53, 5, 24LL, 6, 54);
	      cnn_core1(54, 5, 28LL, 7, 55);
	      cnn_core1(55, 5, 32LL, 8, 56);
#endif
#if (IMAP==6)
	      cnn_final(56,     57);
/*
from GPT4:
The provided code snippet seems to be an implementation of a Convolutional Neural Network (CNN) using the EMAX5A architecture. 
The code is structured into several nested loops and uses a combination of memory operations (mop()) and arithmetic operations (exe()). 
The CNN architecture uses kernels and input feature maps to perform a convolution operation and generate output feature maps.

Here is an overview of the code structure and functionality:

Outer loop: Iterates over the output channels parallelized by multiple chips (CHIP variable).
Middle loop: Iterates over output feature map rows (LOOP1 variable) and handles the row offset (rofs).
Inner loop: Iterates over output feature map columns (LOOP0 variable) and handles the column offset (cofs).
Inside the inner loop, the following operations are performed:
a. Address calculation for input feature map (oofs).
b. Loading kernel values into BR registers.
c. Loading input feature map values into BR registers.
d. Performing floating-point multiplications (FML operations) between input feature map values and kernel values.
e. Calling cnn_core1() function 30 times to perform the convolution operation and accumulate the results.
f. Calling cnn_final() function to finalize the accumulation, add partial sums, and store the results back to memory.
In summary, the provided code snippet implements a Convolutional Neural Network using the EMAX5A architecture by performing a series of load operations, 
arithmetic operations, and custom CNN core functions. The convolution operation is applied to input feature maps using the specified kernels, 
and the results are stored in the output feature maps.
*/
#endif
            }
          }
        }
//EMAX5A end
      }
    }
  }
//EMAX5A drain_dirty_lmm
}
#endif
