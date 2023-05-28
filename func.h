#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// define the data structure tensor, dimension is 5, which is num, channel, depth, height, width
typedef struct {
    int num;
    int channel;
    int depth;
    int height;
    int width;
    float *data; // vector
} tensor;

// tensor multiply function, return the result tensor
tensor tensor_mul(tensor a, tensor b) {
    // check the dimension of two tensors, all dimension should be the same
    if (a.num != b.num || a.channel != b.channel || a.depth != b.depth || a.height != b.height || a.width != b.width) {
        printf("Error: the dimension of two tensors is not the same!");
        exit(1);
    }
    tensor c;
    c.num = a.num;
    c.channel = b.channel;
    c.depth = a.depth;
    c.height = a.height;
    c.width = b.width;
    c.data = (float *)malloc(sizeof(float) * c.num * c.channel * c.depth * c.height * c.width);
    for (int i = 0; i < c.num; i++) {
        for (int j = 0; j < c.channel; j++) {
            for (int k = 0; k < c.depth; k++) {
                for (int l = 0; l < c.height; l++) {
                    for (int m = 0; m < c.width; m++) {
                        c.data[i * c.channel * c.depth * c.height * c.width + j * c.depth * c.height * c.width + k * c.height * c.width + l * c.width + m] = 0;
                        for (int n = 0; n < a.width; n++) {
                            c.data[i * c.channel * c.depth * c.height * c.width + j * c.depth * c.height * c.width + k * c.height * c.width + l * c.width + m] += a.data[i * a.channel * a.depth * a.height * a.width + j * a.depth * a.height * a.width + k * a.height * a.width + l * a.width + n] * b.data[i * b.channel * b.depth * b.height * b.width + j * b.depth * b.height * b.width + k * b.height * b.width + l * b.width + m];
                        }
                    }
                }
            }
        }
    }
    return c;
}

// tensor 3d convolution function
// Helper function to get the index of the tensor data
int get_tensor_index(tensor *t, int n, int c, int d, int h, int w) {
    return n * t->channel * t->depth * t->height * t->width +
           c * t->depth * t->height * t->width +
           d * t->height * t->width +
           h * t->width +
           w;
}

// Dot product function
float dot_product(tensor *t, tensor *f, int n, int c, int k, int d, int h, int w) {
    float result = 0;
    for (int fd = 0; fd < f->depth; fd++) {
        for (int fh = 0; fh < f->height; fh++) {
            for (int fw = 0; fw < f->width; fw++) {
                int t_idx = get_tensor_index(t, n, c, d + fd, h + fh, w + fw);
                int f_idx = get_tensor_index(f, k, c, fd, fh, fw);
                result += t->data[t_idx] * f->data[f_idx];
            }
        }
    }
    return result;
}

// Main 3D convolution function
tensor conv3d(tensor *input, tensor *kernels, tensor *bias, int stride, int padding) {
    assert(input->channel == kernels->channel);
    assert(bias->channel == kernels->num);

    int output_depth = (input->depth - kernels->depth + 2 * padding) / stride + 1;
    int output_height = (input->height - kernels->height + 2 * padding) / stride + 1;
    int output_width = (input->width - kernels->width + 2 * padding) / stride + 1;

    tensor output;
    output.num = input->num;
    output.channel = kernels->num;
    output.depth = output_depth;
    output.height = output_height;
    output.width = output_width;
    output.data = (float *) malloc(output.num * output.channel * output.depth * output.height * output.width * sizeof(float));

    for (int n = 0; n < input->num; n++) {
        for (int k = 0; k < output.channel; k++) {
            for (int d = 0; d < output_depth; d++) {
                for (int h = 0; h < output_height; h++) {
                    for (int w = 0; w < output_width; w++) {
                        float value = 0;
                        for (int c = 0; c < input->channel; c++) {
                            for (int fd = 0; fd < kernels->depth; fd++) {
                                for (int fh = 0; fh < kernels->height; fh++) {
                                    for (int fw = 0; fw < kernels->width; fw++) {
                                        int d_in = d * stride - padding + fd;
                                        int h_in = h * stride - padding + fh;
                                        int w_in = w * stride - padding + fw;

                                        if (d_in >= 0 && d_in < input->depth &&
                                            h_in >= 0 && h_in < input->height &&
                                            w_in >= 0 && w_in < input->width) {
                                            int t_idx = get_tensor_index(input, n, c, d_in, h_in, w_in);
                                            int f_idx = get_tensor_index(kernels, k, c, fd, fh, fw);
                                            value += input->data[t_idx] * kernels->data[f_idx];
                                        }
                                    }
                                }
                            }
                        }
                        value += bias->data[k]; // Add the bias term
                        output.data[get_tensor_index(&output, n, k, d, h, w)] = value;
                    }
                }
            }
        }
    }

    return output;
}