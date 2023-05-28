#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>

// define the data structure tensor, dimension is 5, which is num, channel, depth, height, width
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

// tensor 3d convolution function -------------------------------------------------------
// Helper function to get the index of the tensor data

// Dot product function
float dot_product(tensor t, tensor f, int n, int c, int k, int d, int h, int w) {
    float result = 0;
    for (int fd = 0; fd < f.depth; fd++) {
        for (int fh = 0; fh < f.height; fh++) {
            for (int fw = 0; fw < f.width; fw++) {
                int t_idx = get_tensor_index(t, n, c, d + fd, h + fh, w + fw);
                int f_idx = get_tensor_index(f, k, c, fd, fh, fw);
                result += t.data[t_idx] * f.data[f_idx];
            }
        }
    }
    return result;
}

// Main 3D convolution function
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

// tensor 3d deconvolution function -------------------------------------------------------
// Helper functions


void free_tensor(tensor *t) {
    free(t->data);
}


tensor deconv3d(tensor input, tensor kernel, tensor bias, int stride_d, int stride_h, int stride_w, int padding_d, int padding_h, int padding_w) {
    assert(kernel.channel == bias.num);

    int output_depth = (input.depth - 1) * stride_d - 2 * padding_d + kernel.depth;
    int output_height = (input.height - 1) * stride_h - 2 * padding_h + kernel.height;
    int output_width = (input.width - 1) * stride_w - 2 * padding_w + kernel.width;

    tensor output = create_tensor(input.num, kernel.channel, output_depth, output_height, output_width);

    for (int n = 0; n < input.num; n++) {
        for (int oc = 0; oc < kernel.channel; oc++) {
            for (int od = 0; od < output.depth; od++) {
                for (int oh = 0; oh < output.height; oh++) {
                    for (int ow = 0; ow < output.width; ow++) {
                        float result = 0.0f;

                        for (int ic = 0; ic < input.channel; ic++) {
                            for (int kd = 0; kd < kernel.depth; kd++) {
                                for (int kh = 0; kh < kernel.height; kh++) {
                                    for (int kw = 0; kw < kernel.width; kw++) {
                                        int id = od - kd + padding_d;
                                        int ih = oh - kh + padding_h;
                                        int iw = ow - kw + padding_w;

                                        float input_value = 0.0f;
                                        if (id % stride_d == 0 && ih % stride_h == 0 && iw % stride_w == 0) {
                                            id /= stride_d;
                                            ih /= stride_h;
                                            iw /= stride_w;
                                            if (id >= 0 && id < input.depth && ih >= 0 && ih < input.height && iw >= 0 && iw < input.width) {
                                                input_value = input.data[((n * input.channel + ic) * input.depth + id) * input.height * input.width + ih * input.width + iw];
                                            }
                                        }

                                        float kernel_value = kernel.data[((ic * kernel.channel + oc) * kernel.depth + kd) * kernel.height * kernel.width + kh * kernel.width + kw];
                                        result += input_value * kernel_value;
                                    }
                                }
                            }
                        }

                        float bias_value = bias.data[oc];
                        result += bias_value;
                        output.data[((n * output.channel + oc) * output.depth + od) * output.height * output.width + oh * output.width + ow] = result;
                    }
                }
            }
        }
    }

    return output;
}


// tensor maxpool -------------------------------------------------------
tensor maxpool3d(tensor input, int kernel_size, int stride_d, int stride_h, int stride_w, int padding) {
    int output_depth = (input.depth + 2 * padding - kernel_size) / stride_d + 1;
    int output_height = (input.height + 2 * padding - kernel_size) / stride_h + 1;
    int output_width = (input.width + 2 * padding - kernel_size) / stride_w + 1;

    tensor output = create_tensor(input.num, input.channel, output_depth, output_height, output_width);

    for (int n = 0; n < input.num; n++) {
        for (int c = 0; c < input.channel; c++) {
            for (int od = 0; od < output_depth; od++) {
                for (int oh = 0; oh < output_height; oh++) {
                    for (int ow = 0; ow < output_width; ow++) {
                        float max_value = -INFINITY;

                        for (int kd = 0; kd < kernel_size; kd++) {
                            for (int kh = 0; kh < kernel_size; kh++) {
                                for (int kw = 0; kw < kernel_size; kw++) {
                                    int id = od * stride_d + kd - padding;
                                    int ih = oh * stride_h + kh - padding;
                                    int iw = ow * stride_w + kw - padding;

                                    if (id >= 0 && id < input.depth && ih >= 0 && ih < input.height && iw >= 0 && iw < input.width) {
                                        float input_value = input.data[((n * input.channel + c) * input.depth + id) * input.height * input.width + ih * input.width + iw];
                                        if (input_value > max_value) {
                                            max_value = input_value;
                                        }
                                    }
                                }
                            }
                        }

                        output.data[((n * output.channel + c) * output.depth + od) * output.height * output.width + oh * output.width + ow] = max_value;
                    }
                }
            }
        }
    }

    return output;
}

// tensor concat -------------------------------------------------------
tensor concat(tensor input1, tensor input2) {
    assert(input1.num == input2.num);
    assert(input1.depth == input2.depth);
    assert(input1.height == input2.height);
    assert(input1.width == input2.width);

    int output_channel = input1.channel + input2.channel;

    tensor output = create_tensor(input1.num, output_channel, input1.depth, input1.height, input1.width);

    for (int n = 0; n < input1.num; n++) {
        for (int d = 0; d < input1.depth; d++) {
            for (int h = 0; h < input1.height; h++) {
                for (int w = 0; w < input1.width; w++) {
                    for (int c = 0; c < input1.channel; c++) {
                        float value = input1.data[((n * input1.channel + c) * input1.depth + d) * input1.height * input1.width + h * input1.width + w];
                        output.data[((n * output_channel + c) * output.depth + d) * output.height * output.width + h * output.width + w] = value;
                    }
                    for (int c = 0; c < input2.channel; c++) {
                        float value = input2.data[((n * input2.channel + c) * input2.depth + d) * input2.height * input2.width + h * input2.width + w];
                        output.data[((n * output_channel + input1.channel + c) * output.depth + d) * output.height * output.width + h * output.width + w] = value;
                    }
                }
            }
        }
    }

    return output;
}

// tensor LeakyReLU -------------------------------------------------------
tensor leakyrelu(tensor input, float alpha) {
    tensor output = create_tensor(input.num, input.channel, input.depth, input.height, input.width);

    for (int i = 0; i < input.num * input.channel * input.depth * input.height * input.width; i++) {
        float value = input.data[i];
        if (value < 0) {
            value *= alpha;
        }
        output.data[i] = value;
    }
    return output;
}

// tensor norminstance -------------------------------------------------------
tensor instance_norm3d(tensor input, float eps, float momentum) {
    int num = input.num;
    int channel = input.channel;
    int depth = input.depth;
    int height = input.height;
    int width = input.width;
    int num_elements = depth * height * width;

    tensor output;
    output.num = num;
    output.channel = channel;
    output.depth = depth;
    output.height = height;
    output.width = width;
    output.data = (float *) malloc(num * channel * depth * height * width * sizeof(float));

    for (int n = 0; n < num; n++) {
        for (int c = 0; c < channel; c++) {
            float mean = 0;
            float squared_mean = 0;

            // Calculate the mean and squared mean of the input
            for (int d = 0; d < depth; d++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int idx = get_tensor_index(input, n, c, d, h, w);
                        float val = input.data[idx];
                        mean += val;
                        squared_mean += val * val;
                    }
                }
            }

            mean /= num_elements;
            squared_mean /= num_elements;

            // Calculate the variance
            float variance = squared_mean - mean * mean;

            // Normalize the input using mean, variance, and epsilon
            for (int d = 0; d < depth; d++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int idx = get_tensor_index(input, n, c, d, h, w);
                        float normalized_value = (input.data[idx] - mean) / sqrtf(variance + eps);
                        output.data[idx] = normalized_value;
                    }
                }
            }
        }
    }

    return output;
}

// tensor center crop -------------------------------------------------------
tensor center_crop(tensor input, int target_depth, int target_height, int target_width) {
    int num = input.num;
    int channel = input.channel;
    int depth = input.depth;
    int height = input.height;
    int width = input.width;

    int d0 = (depth - target_depth) / 2;
    int h0 = (height - target_height) / 2;
    int w0 = (width - target_width) / 2;

    tensor output;
    output.num = num;
    output.channel = channel;
    output.depth = target_depth;
    output.height = target_height;
    output.width = target_width;
    output.data = (float *) malloc(num * channel * target_depth * target_height * target_width * sizeof(float));

    for (int n = 0; n < num; n++) {
        for (int c = 0; c < channel; c++) {
            for (int d = 0; d < target_depth; d++) {
                for (int h = 0; h < target_height; h++) {
                    for (int w = 0; w < target_width; w++) {
                        int in_idx = get_tensor_index(input, n, c, d + d0, h + h0, w + w0);
                        int out_idx = get_tensor_index(output, n, c, d, h, w);
                        output.data[out_idx] = input.data[in_idx];
                    }
                }
            }
        }
    }

    return output;
}



// def test conv3d()
void testconv3d(){
    tensor input;
    input.num = 1; input.channel = 1; input.depth = 8; input.height = 8; input.width = 8;
    input.data = (float *) malloc(input.num * input.channel * input.depth * input.height * input.width * sizeof(float));
    for (int i = 0; i < input.num * input.channel * input.depth * input.height * input.width; i++) {
        input.data[i] = 1;
    }

    tensor kernel;
    kernel.num = 3; kernel.channel = 1; kernel.depth = 3; kernel.height = 3; kernel.width = 3;
    // in conv, kernel.num control output channel, kernel chnl == input chnl
    kernel.data = (float *) malloc(kernel.num * kernel.channel * kernel.depth * kernel.height * kernel.width * sizeof(float));
    for (int i = 0; i < kernel.num * kernel.channel * kernel.depth * kernel.height * kernel.width; i++) {
        kernel.data[i] = 1;
    }

    tensor bias;
    bias.num = 1; bias.channel = 3; bias.depth = 1; bias.height = 1; bias.width = 1;
    bias.data = (float *) malloc(bias.channel * sizeof(float));
    for (int i = 0; i < bias.channel; i++) {
        bias.data[i] = 0;
    }

    int stride = 2;
    int padding = 1;

    tensor output = conv3d(input, kernel, bias, stride, padding);
    // print output as vec
    write_tensor_to_file("output.txt", output);

}

// def test deconv3d()
void testdeconv3d() {
    tensor input;//(1,1024,8,2,2)
    input.num = 1; input.channel = 1024; input.depth = 8; input.height = 2; input.width = 2;
    input.data = (float *)calloc(input.num * input.channel * input.depth * input.height * input.width, sizeof(float));
    for (int i = 0; i < input.num * input.channel * input.depth * input.height * input.width; i++) {
        input.data[i] = 0.1;
    }

    // kernel (1024,512,2,2,2)
    int num = 1024; int channel = 512; int depth = 2; int height = 2; int width = 2;
    const char *filename = "savedWB2/21.txt";
    tensor kernel1 = create_tensor_from_file(filename, num, channel, depth, height, width);

    // bias (512,)
    int bias_num = 512; int bias_channel = 1; int bias_depth = 1; int bias_height = 1; int bias_width = 1;
    const char *filename2 = "savedWB2/20.txt";
    tensor bias1 = create_tensor_from_file(filename2, bias_num, bias_channel, bias_depth, bias_height, bias_width);

    int stride_d = 1; int stride_h = 2; int stride_w = 2; int padding_d = 0; int padding_h = 0; int padding_w = 0;
    tensor output = deconv3d(input, kernel1, bias1, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w);  
    save_tensor_to_npy(output, "testdeconv3d.npy");

}

// def test maxpool3d()
void testmaxpool3d() {
    tensor input;
    input.num = 1; input.channel = 1; input.depth = 8; input.height = 8; input.width = 8;
    input.data = (float *) malloc(input.num * input.channel * input.depth * input.height * input.width * sizeof(float));
    // for (int i = 0; i < input.num * input.channel * input.depth * input.height * input.width; i++) {
    //     input.data[i] = 1;
    // }
    // rand init
    for (int i = 0; i < input.num * input.channel * input.depth * input.height * input.width; i++) {
        input.data[i] = rand() % 100;
    }

    int kernel_size = 3; int stride_d = 2; int stride_h = 2; int stride_w = 2; int padding = 1;
    tensor output = maxpool3d(input, kernel_size, stride_d, stride_h, stride_w, padding);
    // print output as vec
    write_tensor_to_file("output.txt", output);
}

// test concat
void testconcat() {
    tensor input1;
    input1.num = 1; input1.channel = 1; input1.depth = 8; input1.height = 8; input1.width = 8;
    input1.data = (float *) malloc(input1.num * input1.channel * input1.depth * input1.height * input1.width * sizeof(float));
    for (int i = 0; i < input1.num * input1.channel * input1.depth * input1.height * input1.width; i++) {
        input1.data[i] = 1;
    }

    tensor input2;
    input2.num = 1; input2.channel = 1; input2.depth = 8; input2.height = 8; input2.width = 8;
    input2.data = (float *) malloc(input2.num * input2.channel * input2.depth * input2.height * input2.width * sizeof(float));
    for (int i = 0; i < input2.num * input2.channel * input2.depth * input2.height * input2.width; i++) {
        input2.data[i] = 2;
    }

    tensor output = concat(input1, input2);
    // print output as vec
    write_tensor_to_file("output.txt", output);
}

// test create_tensor_from_file()
void test_create_tensor_from_file() {
    const char *filename = "savedWB2/1.txt"; //64 1 3 3 3
    int num = 64;int channel = 1;int depth = 3;int height = 3;int width = 3;
    // const char *filename = "savedWB2/0.txt"; //64
    // int num = 64;int channel = 1;int depth = 1;int height = 1;int width = 1;
    tensor my_tensor = create_tensor_from_file(filename, num, channel, depth, height, width);
    // print output as vec
    write_tensor_to_file("output.txt", my_tensor);
}

// test center_crop()
void test_center_crop() {
    tensor input;
    input.num = 1; input.channel = 1; input.depth = 14; input.height = 14; input.width = 14;
    input.data = (float *) malloc(input.num * input.channel * input.depth * input.height * input.width * sizeof(float));
    for (int i = 0; i < input.num * input.channel * input.depth * input.height * input.width; i++) {
        input.data[i] = 1;
    }

    int crop_d = 11; int crop_h = 5; int crop_w = 4;
    tensor output = center_crop(input, crop_d, crop_h, crop_w);
    // print output as vec
    write_tensor_to_file("output.txt", output);
}

// test conv+relu
void test_conv_relu() {
    // creat input tensor, shape (1,1,32,32,32)// data all 0.1
    tensor input;
    input.num = 1; input.channel = 1; input.depth = 32; input.height = 32; input.width = 32;
    input.data = (float *) malloc(input.num * input.channel * input.depth * input.height * input.width * sizeof(float));
    for (int i = 0; i < input.num * input.channel * input.depth * input.height * input.width; i++) {
        input.data[i] = 0.1;
    }
    // conv1 (in=1, out=64, kernel=3, stride=1, padding=1), _____________________________________________
    // create kernel1 tensor, shape (64,1,3,3,3)// read from file
    int num = 64;int channel = 1;int depth = 3;int height = 3;int width = 3;
    const char *filename = "savedWB2/1.txt"; //64 1 3 3 3
    tensor kernel1 = create_tensor_from_file(filename, num, channel, depth, height, width);
    // create bias1 tensor, shape (1,64,1,1,1)// read from file
    num = 1;channel = 64;depth = 1;height = 1;width = 1;
    filename = "savedWB2/0.txt";
    tensor bias1 = create_tensor_from_file(filename, num, channel, depth, height, width);
    
    int stride = 1; int padding = 1;
    tensor conv3d_1_out = conv3d(input, kernel1, bias1, stride, padding);
    tensor relu1_out = leakyrelu(conv3d_1_out, 0.01);
    save_tensor_to_npy(relu1_out, "test_convRelu.npy");
}

//test instance_norm3d()
void test_instance_norm3d() {
    // creat input tensor, shape (1,1,32,32,32)// data all 0.1
    // tensor input;
    // input.num = 1; input.channel = 1; input.depth = 8; input.height = 8; input.width = 8;
    // input.data = (float *) malloc(input.num * input.channel * input.depth * input.height * input.width * sizeof(float));
    // for (int i = 0; i < input.num * input.channel * input.depth * input.height * input.width; i++) {
    //     input.data[i] = 10000;
    // }

    int num = 1; int channel = 1; int depth = 2; int height = 2; int width = 2;
    const char *filename = "test_input.txt";
    tensor input = create_tensor_from_file(filename, num, channel, depth, height, width);


    tensor output = instance_norm3d(input, 1e-5, 0.1);
    write_tensor_to_file("input.txt", input);
    write_tensor_to_file("output.txt", output);
}

// small network
tensor net(){
    // creat input tensor, shape (1,1,32,32,32)// data all 0.1
    // tensor input;
    // input.num = 1; input.channel = 1; input.depth = 32; input.height = 32; input.width = 32;
    // input.data = (float *) malloc(input.num * input.channel * input.depth * input.height * input.width * sizeof(float));
    // for (int i = 0; i < input.num * input.channel * input.depth * input.height * input.width; i++) {
    //     input.data[i] = 0.1;
    // }
    tensor input; int innum = 1; int inch = 1; int indepth = 20; int inheight = 269; int inwidth = 244;
    const char *filenamein = "data.txt"; // from 0.npy
    input = create_tensor_from_file(filenamein, innum, inch, indepth, inheight, inwidth);
    // conv1 (in=1, out=64, kernel=3, stride=1, padding=1), _____________________________________________
    // create kernel1 tensor, shape (64,1,3,3,3)// read from file
    int num = 64;int channel = 1;int depth = 3;int height = 3;int width = 3;
    const char *filename = "savedWB2/1.txt";
    tensor kernel1 = create_tensor_from_file(filename, num, channel, depth, height, width);
    
    int num_b = 1;int channel_b = 64;int depth_b = 1;int height_b = 1;int width_b = 1;
    const char *filename1 = "savedWB2/0.txt";
    tensor bias1 = create_tensor_from_file(filename1, num_b, channel_b, depth_b, height_b, width_b);
    
    int stride = 1; int padding = 1;
    tensor conv3d_1_out = conv3d(input, kernel1, bias1, stride, padding);
    // tensor conv3d_1_out_norm = instance_norm3d(conv3d_1_out, eps, momentum);
    tensor conv3d_1_out_relu = leakyrelu(conv3d_1_out, 0.01);
    save_tensor_to_npy(conv3d_1_out_relu, "conv3d_1_out.npy"); // (1,64,64,64,64) // Okkkkkkkk

    // conv2 (in=64, out=64, kernel=3, stride=1, padding=1), _____________________________________________
    // create kernel2 tensor, shape (64,64,3,3,3)// read from file
    int num1 = 64;int channel1 = 64;int depth1 = 3;int height1 = 3;int width1 = 3;
    const char *filename2 = "savedWB2/3.txt";
    tensor kernel2 = create_tensor_from_file(filename2, num1, channel1, depth1, height1, width1);

    int num_b1 = 1;int channel_b1 = 64;int depth_b1 = 1;int height_b1 = 1;int width_b1 = 1;
    const char *filename3 = "savedWB2/2.txt";
    tensor bias2 = create_tensor_from_file(filename3, num_b1, channel_b1, depth_b1, height_b1, width_b1);

    int stride1 = 1; int padding1 = 1;
    tensor conv3d_2_out = conv3d(conv3d_1_out_relu, kernel2, bias2, stride1, padding1);
    // tensor conv3d_2_out_norm = instance_norm3d(conv3d_2_out, eps1, momentum1);
    tensor conv3d_2_out_relu = leakyrelu(conv3d_2_out, 0.01); // -> 1
    save_tensor_to_npy(conv3d_2_out_relu, "conv3d_2_out.npy"); // Okkkkkkkk

    // maxpool1 (in=64, out=64, kernel=3, stride=[1,2,2], padding=1), _____________________________________________
    int kernel_size1 = 3;int stride_d1 = 1; int stride_h1 = 2; int stride_w1 = 2; int padding_p1 = 1;
    tensor maxpool3d_1_out = maxpool3d(conv3d_2_out_relu, kernel_size1, stride_d1, stride_h1, stride_w1, padding_p1);
    save_tensor_to_npy(maxpool3d_1_out, "maxpool3d_1_out.npy"); //Okkkkk

    // conv3 (in=64, out=128, kernel=3, stride=1, padding=1), _____________________________________________
    // create kernel3 tensor, shape (128,64,3,3,3)// read from file
    int num2 = 128;int channel2 = 64;int depth2 = 3;int height2 = 3;int width2 = 3;
    const char *filename4 = "savedWB2/5.txt";
    tensor kernel3 = create_tensor_from_file(filename4, num2, channel2, depth2, height2, width2);

    int num_b2 = 1;int channel_b2 = 128;int depth_b2 = 1;int height_b2 = 1;int width_b2 = 1;
    const char *filename5 = "savedWB2/4.txt";
    tensor bias3 = create_tensor_from_file(filename5, num_b2, channel_b2, depth_b2, height_b2, width_b2);

    int stride2 = 1; int padding3 = 1;
    tensor conv3d_3_out = conv3d(maxpool3d_1_out, kernel3, bias3, stride2, padding3);
    // tensor conv3d_3_out_norm = instance_norm3d(conv3d_3_out, eps2, momentum2);
    tensor conv3d_3_out_relu = leakyrelu(conv3d_3_out, 0.01);
    save_tensor_to_npy(conv3d_3_out_relu, "conv3d_3_out.npy"); // Okkkkkkkk

    // conv4 (in=128, out=128, kernel=3, stride=1, padding=1), _____________________________________________
    // create kernel4 tensor, shape (128,128,3,3,3)// read from file
    int num3 = 128;int channel3 = 128;int depth3 = 3;int height3 = 3;int width3 = 3;
    const char *filename6 = "savedWB2/7.txt";
    tensor kernel4 = create_tensor_from_file(filename6, num3, channel3, depth3, height3, width3);

    int num_b3 = 1;int channel_b3 = 128;int depth_b3 = 1;int height_b3 = 1;int width_b3 = 1;
    const char *filename7 = "savedWB2/6.txt";
    tensor bias4 = create_tensor_from_file(filename7, num_b3, channel_b3, depth_b3, height_b3, width_b3);

    int stride3 = 1; int padding4 = 1;
    tensor conv3d_4_out = conv3d(conv3d_3_out_relu, kernel4, bias4, stride3, padding4);
    // tensor conv3d_4_out_norm = instance_norm3d(conv3d_4_out, eps3, momentum3);
    tensor conv3d_4_out_relu = leakyrelu(conv3d_4_out, 0.01); // -> 2
    save_tensor_to_npy(conv3d_4_out_relu, "conv3d_4_out.npy");

    // maxpool2 (in=128, out=128, kernel=3, stride=2, padding=1), _____________________________________________
    int kernel_size2 = 3; int stride_d2 = 2; int stride_h2 = 2; int stride_w2 = 2; int padding_p2 = 1;
    tensor maxpool3d_2_out = maxpool3d(conv3d_4_out_relu, kernel_size2, stride_d2, stride_h2, stride_w2, padding_p2);
    save_tensor_to_npy(maxpool3d_2_out, "maxpool3d_2_out.npy");

    // conv5 (in=128, out=256, kernel=3, stride=1, padding=1), _____________________________________________
    // create kernel5 tensor, shape (256,128,3,3,3)// read from file
    int num4 = 256;int channel4 = 128;int depth4 = 3;int height4 = 3;int width4 = 3;
    const char *filename8 = "savedWB2/9.txt";
    tensor kernel5 = create_tensor_from_file(filename8, num4, channel4, depth4, height4, width4);

    int num_b4 = 1;int channel_b4 = 256;int depth_b4 = 1;int height_b4 = 1;int width_b4 = 1;
    const char *filename9 = "savedWB2/8.txt";
    tensor bias5 = create_tensor_from_file(filename9, num_b4, channel_b4, depth_b4, height_b4, width_b4);

    int stride4 = 1; int padding6 = 1;
    tensor conv3d_5_out = conv3d(maxpool3d_2_out, kernel5, bias5, stride4, padding6);
    // tensor conv3d_5_out_norm = instance_norm3d(conv3d_5_out, eps4, momentum4);
    tensor conv3d_5_out_relu = leakyrelu(conv3d_5_out, 0.01);
    save_tensor_to_npy(conv3d_5_out_relu, "conv3d_5_out.npy");

    // conv6 (in=256, out=256, kernel=3, stride=1, padding=1), _____________________________________________
    // create kernel6 tensor, shape (256,256,3,3,3)// read from file
    int num5 = 256;int channel5 = 256;int depth5 = 3;int height5 = 3;int width5 = 3;
    const char *filename10 = "savedWB2/11.txt";
    tensor kernel6 = create_tensor_from_file(filename10, num5, channel5, depth5, height5, width5);

    int num_b5 = 1;int channel_b5 = 256;int depth_b5 = 1;int height_b5 = 1;int width_b5 = 1;
    const char *filename11 = "savedWB2/10.txt";
    tensor bias6 = create_tensor_from_file(filename11, num_b5, channel_b5, depth_b5, height_b5, width_b5);

    int stride5 = 1; int padding7 = 1; float eps5 = 1e-5; float momentum5 = 0.1;
    tensor conv3d_6_out = conv3d(conv3d_5_out_relu, kernel6, bias6, stride5, padding7);
    // tensor conv3d_6_out_norm = instance_norm3d(conv3d_6_out, eps5, momentum5);
    tensor conv3d_6_out_relu = leakyrelu(conv3d_6_out, 0.01); // -> 3

    // maxpool3 (in=256, out=256, kernel=3, stride=2, padding=1), _____________________________________________
    int kernel_size3 = 3; int stride_d3 = 2; int stride_h3 = 2; int stride_w3 = 2; int padding_p3 = 1;
    tensor maxpool3d_3_out = maxpool3d(conv3d_6_out_relu, kernel_size3, stride_d3, stride_h3, stride_w3, padding_p3);
    save_tensor_to_npy(maxpool3d_3_out, "maxpool3d_3_out.npy");

    // conv7 (in=256, out=512, kernel=3, stride=1, padding=1), _____________________________________________
    // create kernel7 tensor, shape (512,256,3,3,3)// read from file
    int num6 = 512;int channel6 = 256;int depth6 = 3;int height6 = 3;int width6 = 3;
    const char *filename12 = "savedWB2/13.txt";
    tensor kernel7 = create_tensor_from_file(filename12, num6, channel6, depth6, height6, width6);

    int num_b6 = 1;int channel_b6 = 512;int depth_b6 = 1;int height_b6 = 1;int width_b6 = 1;
    const char *filename13 = "savedWB2/12.txt";
    tensor bias7 = create_tensor_from_file(filename13, num_b6, channel_b6, depth_b6, height_b6, width_b6);

    int stride6 = 1; int padding9 = 1; float eps6 = 1e-5; float momentum6 = 0.1;
    tensor conv3d_7_out = conv3d(maxpool3d_3_out, kernel7, bias7, stride6, padding9);
    // tensor conv3d_7_out_norm = instance_norm3d(conv3d_7_out, eps6, momentum6);
    tensor conv3d_7_out_relu = leakyrelu(conv3d_7_out, 0.01);
    save_tensor_to_npy(conv3d_7_out_relu, "conv3d_7_out.npy");

    // conv8 (in=512, out=512, kernel=3, stride=1, padding=1), _____________________________________________
    // create kernel8 tensor, shape (512,512,3,3,3)// read from file
    int num7 = 512;int channel7 = 512;int depth7 = 3;int height7 = 3;int width7 = 3;
    const char *filename14 = "savedWB2/15.txt";
    tensor kernel8 = create_tensor_from_file(filename14, num7, channel7, depth7, height7, width7);

    int num_b7 = 1;int channel_b7 = 512;int depth_b7 = 1;int height_b7 = 1;int width_b7 = 1;
    const char *filename15 = "savedWB2/14.txt";
    tensor bias8 = create_tensor_from_file(filename15, num_b7, channel_b7, depth_b7, height_b7, width_b7);

    int stride7 = 1; int padding10 = 1; float eps7 = 1e-5; float momentum7 = 0.1;
    tensor conv3d_8_out = conv3d(conv3d_7_out_relu, kernel8, bias8, stride7, padding10);
    // tensor conv3d_8_out_norm = instance_norm3d(conv3d_8_out, eps7, momentum7);
    tensor conv3d_8_out_relu = leakyrelu(conv3d_8_out, 0.01); // -> 4
    save_tensor_to_npy(conv3d_8_out_relu, "conv3d_8_out.npy");

    // maxpool4 (in=512, out=512, kernel=3, stride=[1,2,2], padding=1), _____________________________________________
    int kernel_size4 = 3; int stride_d4 = 1; int stride_h4 = 2; int stride_w4 = 2; int padding_p4 = 1;
    tensor maxpool3d_4_out = maxpool3d(conv3d_8_out_relu, kernel_size4, stride_d4, stride_h4, stride_w4, padding_p4);
    save_tensor_to_npy(maxpool3d_4_out, "maxpool3d_4_out.npy");

    // conv9 (in=512, out=1024, kernel=3, stride=1, padding=1), _____________________________________________
    // create kernel9 tensor, shape (1024,512,3,3,3)// read from file
    int num8 = 1024;int channel8 = 512;int depth8 = 3;int height8 = 3;int width8 = 3;
    const char *filename16 = "savedWB2/17.txt";
    tensor kernel9 = create_tensor_from_file(filename16, num8, channel8, depth8, height8, width8);

    int num_b8 = 1;int channel_b8 = 1024;int depth_b8 = 1;int height_b8 = 1;int width_b8 = 1;
    const char *filename17 = "savedWB2/16.txt";
    tensor bias9 = create_tensor_from_file(filename17, num_b8, channel_b8, depth_b8, height_b8, width_b8);

    int stride8 = 1; int padding12 = 1; float eps8 = 1e-5; float momentum8 = 0.1;
    tensor conv3d_9_out = conv3d(maxpool3d_4_out, kernel9, bias9, stride8, padding12);
    // tensor conv3d_9_out_norm = instance_norm3d(conv3d_9_out, eps8, momentum8);
    tensor conv3d_9_out_relu = leakyrelu(conv3d_9_out, 0.01);
    save_tensor_to_npy(conv3d_9_out_relu, "conv3d_9_out.npy");

    // conv10 (in=1024, out=1024, kernel=3, stride=1, padding=1), _____________________________________________
    // create kernel10 tensor, shape (1024,1024,3,3,3)// read from file
    int num9 = 1024;int channel9 = 1024;int depth9 = 3;int height9 = 3;int width9 = 3;
    const char *filename18 = "savedWB2/19.txt";
    tensor kernel10 = create_tensor_from_file(filename18, num9, channel9, depth9, height9, width9);

    int num_b9 = 1;int channel_b9 = 1024;int depth_b9 = 1;int height_b9 = 1;int width_b9 = 1;
    const char *filename19 = "savedWB2/18.txt";
    tensor bias10 = create_tensor_from_file(filename19, num_b9, channel_b9, depth_b9, height_b9, width_b9);

    int stride9 = 1; int padding13 = 1; 
    tensor conv3d_10_out = conv3d(conv3d_9_out_relu, kernel10, bias10, stride9, padding13);
    // tensor conv3d_10_out_norm = instance_norm3d(conv3d_10_out, eps9, momentum9);
    tensor conv3d_10_out_relu = leakyrelu(conv3d_10_out, 0.01); // shape (1, 1024, 8, 2, 2)
    save_tensor_to_npy(conv3d_10_out_relu, "conv3d_10_out.npy");

    // deconv1 (in=1024, out=512, kernel=2, stride=[1,2,2], padding=0), _____________________________________________
    // create kernel11 tensor, shape (1024,512,2,2,2)// read from file
    int num10 = 1024;int channel10 = 512;int depth10 = 2;int height10 = 2;int width10 = 2;
    const char *filename20 = "savedWB2/21.txt";
    tensor kernel11 = create_tensor_from_file(filename20, num10, channel10, depth10, height10, width10);
    // deconv: bias.num == kernel.chanel
    int num_b10 = 512;int channel_b10 = 1;int depth_b10 = 1;int height_b10 = 1;int width_b10 = 1;
    const char *filename21 = "savedWB2/20.txt";
    tensor bias11 = create_tensor_from_file(filename21, num_b10, channel_b10, depth_b10, height_b10, width_b10);

    int stride_d11 = 1; int stride_h11 = 2; int stride_w11 = 2; int padding_d11 = 0; int padding_h11 = 0; int padding_w11 = 0;
    tensor deconv3d_11_out = deconv3d(conv3d_10_out_relu, kernel11, bias11, stride_d11, stride_h11, stride_w11, padding_d11, padding_h11, padding_w11);
    save_tensor_to_npy(deconv3d_11_out, "deconv3d_11_out.npy"); // should be (1, 512, 9, 4, 4)
    // center crop: 
    int crop_d1 = conv3d_8_out_relu.depth; int crop_h1 = conv3d_8_out_relu.height; int crop_w1 = conv3d_8_out_relu.width; // 4.dhw
    tensor deconv3d_11_out_crop = center_crop(deconv3d_11_out, crop_d1, crop_h1, crop_w1);
    save_tensor_to_npy(deconv3d_11_out_crop, "deconv3d_11_out_crop.npy");// should be (1, 512, 8, 4, 4) Okkkkkkkkkkkkkkkkkkkkkkkkk
    // no relu and norm for deconv

    // concat12 (in=[512,512], out=1024), _____________________________________________
    tensor concat12_out = concat(conv3d_8_out_relu, deconv3d_11_out_crop); // <- 4 conv3d_8_out_relu

    // conv13 (in=1024, out=512, kernel=3, stride=1, padding=1), _____________________________________________
    // create kernel12 tensor, shape (512,1024,3,3,3)// read from file
    int num11 = 512;int channel11 = 1024;int depth11 = 3;int height11 = 3;int width11 = 3;
    const char *filename22 = "savedWB2/23.txt";
    tensor kernel12 = create_tensor_from_file(filename22, num11, channel11, depth11, height11, width11);

    int num_b11 = 1;int channel_b11 = 512;int depth_b11 = 1;int height_b11 = 1;int width_b11 = 1;
    const char *filename23 = "savedWB2/22.txt";
    tensor bias12 = create_tensor_from_file(filename23, num_b11, channel_b11, depth_b11, height_b11, width_b11);

    int stride10 = 1; int padding14 = 1; 
    tensor conv3d_13_out = conv3d(concat12_out, kernel12, bias12, stride10, padding14);
    // tensor conv3d_13_out_norm = instance_norm3d(conv3d_13_out, eps10, momentum10); no norm
    tensor conv3d_13_out_relu = leakyrelu(conv3d_13_out, 0.01);
    save_tensor_to_npy(conv3d_13_out_relu, "conv3d_13_out_relu.npy"); // okkkkk

    // conv14 (in=512, out=512, kernel=3, stride=1, padding=1), _____________________________________________
    // create kernel13 tensor, shape (512,512,3,3,3)// read from file
    int num12 = 512;int channel12 = 512;int depth12 = 3;int height12 = 3;int width12 = 3;
    const char *filename24 = "savedWB2/25.txt";
    tensor kernel13 = create_tensor_from_file(filename24, num12, channel12, depth12, height12, width12);

    int num_b12 = 1;int channel_b12 = 512;int depth_b12 = 1;int height_b12 = 1;int width_b12 = 1;
    const char *filename25 = "savedWB2/24.txt";
    tensor bias13 = create_tensor_from_file(filename25, num_b12, channel_b12, depth_b12, height_b12, width_b12);

    int stride11 = 1; int padding15 = 1; 
    tensor conv3d_14_out = conv3d(conv3d_13_out_relu, kernel13, bias13, stride11, padding15);
    tensor conv3d_14_out_relu = leakyrelu(conv3d_14_out, 0.01);
    save_tensor_to_npy(conv3d_14_out_relu, "conv3d_14_out_relu.npy"); // okkkkk

    // deconv2 (in=512, out=256, kernel=2, stride=2, padding=0), _____________________________________________
    // create kernel14 tensor, shape (512,256,2,2,2)// read from file
    int num13 = 512;int channel13 = 256;int depth13 = 2;int height13 = 2;int width13 = 2;
    const char *filename26 = "savedWB2/27.txt";
    tensor kernel14 = create_tensor_from_file(filename26, num13, channel13, depth13, height13, width13);

    int num_b13 = 256;int channel_b13 = 1;int depth_b13 = 1;int height_b13 = 1;int width_b13 = 1;
    const char *filename27 = "savedWB2/26.txt";
    tensor bias14 = create_tensor_from_file(filename27, num_b13, channel_b13, depth_b13, height_b13, width_b13);

    int stride_d13 = 2; int stride_h13 = 2; int stride_w13 = 2; int padding_d13 = 0; int padding_h13 = 0; int padding_w13 = 0;
    tensor deconv3d_2_out = deconv3d(conv3d_14_out_relu, kernel14, bias14, stride_d13, stride_h13, stride_w13, padding_d13, padding_h13, padding_w13);
    // center crop
    int crop_d2 = conv3d_6_out_relu.depth; int crop_h2 = conv3d_6_out_relu.height; int crop_w2 = conv3d_6_out_relu.width;
    tensor deconv3d_2_out_crop = center_crop(deconv3d_2_out, crop_d2, crop_h2, crop_w2);
    save_tensor_to_npy(deconv3d_2_out_crop, "deconv3d_2_out_crop.npy"); // okkkkk, mse=1.5180946e-14

    // concat2 (in=[256,256], out=512), _____________________________________________
    tensor concat13_out = concat(conv3d_6_out_relu, deconv3d_2_out_crop); // <- 3 conv3d_6_out_relu

    // conv15 (in=512, out=256, kernel=3, stride=1, padding=1), _____________________________________________
    // create kernel15 tensor, shape (256,512,3,3,3)// read from file
    int num14 = 256;int channel14 = 512;int depth14 = 3;int height14 = 3;int width14 = 3;
    const char *filename28 = "savedWB2/29.txt";
    tensor kernel15 = create_tensor_from_file(filename28, num14, channel14, depth14, height14, width14);

    int num_b14 = 1;int channel_b14 = 256;int depth_b14 = 1;int height_b14 = 1;int width_b14 = 1;
    const char *filename29 = "savedWB2/28.txt";
    tensor bias15 = create_tensor_from_file(filename29, num_b14, channel_b14, depth_b14, height_b14, width_b14);

    int stride12 = 1; int padding16 = 1;
    tensor conv3d_15_out = conv3d(concat13_out, kernel15, bias15, stride12, padding16);
    tensor conv3d_15_out_relu = leakyrelu(conv3d_15_out, 0.01);
    save_tensor_to_npy(conv3d_15_out_relu, "conv3d_15_out_relu.npy"); // okkkkk, mse=6.145547e-14

    // conv16 (in=256, out=256, kernel=3, stride=1, padding=1), _____________________________________________
    // create kernel16 tensor, shape (256,256,3,3,3)// read from file
    int num15 = 256;int channel15 = 256;int depth15 = 3;int height15 = 3;int width15 = 3;
    const char *filename30 = "savedWB2/31.txt";
    tensor kernel16 = create_tensor_from_file(filename30, num15, channel15, depth15, height15, width15);

    int num_b15 = 1;int channel_b15 = 256;int depth_b15 = 1;int height_b15 = 1;int width_b15 = 1;
    const char *filename31 = "savedWB2/30.txt";
    tensor bias16 = create_tensor_from_file(filename31, num_b15, channel_b15, depth_b15, height_b15, width_b15);

    int stride13 = 1; int padding17 = 1;
    tensor conv3d_16_out = conv3d(conv3d_15_out_relu, kernel16, bias16, stride13, padding17);
    tensor conv3d_16_out_relu = leakyrelu(conv3d_16_out, 0.01);
    save_tensor_to_npy(conv3d_16_out_relu, "conv3d_16_out_relu.npy"); // okkkkk, mse=9.6983924e-14

    // deconv3 (in=256, out=128, kernel=2, stride=2, padding=0), _____________________________________________
    // create kernel17 tensor, shape (256,128,2,2,2)// read from file
    int num16 = 256;int channel16 = 128;int depth16 = 2;int height16 = 2;int width16 = 2;
    const char *filename32 = "savedWB2/33.txt";
    tensor kernel17 = create_tensor_from_file(filename32, num16, channel16, depth16, height16, width16);

    int num_b16 = 128;int channel_b16 = 1;int depth_b16 = 1;int height_b16 = 1;int width_b16 = 1;
    const char *filename33 = "savedWB2/32.txt";
    tensor bias17 = create_tensor_from_file(filename33, num_b16, channel_b16, depth_b16, height_b16, width_b16);

    int stride_d14 = 2; int stride_h14 = 2; int stride_w14 = 2; int padding_d14 = 0; int padding_h14 = 0; int padding_w14 = 0;
    tensor deconv3d_3_out = deconv3d(conv3d_16_out_relu, kernel17, bias17, stride_d14, stride_h14, stride_w14, padding_d14, padding_h14, padding_w14);
    // center crop
    int crop_d3 = conv3d_4_out_relu.depth; int crop_h3 = conv3d_4_out_relu.height; int crop_w3 = conv3d_4_out_relu.width;
    tensor deconv3d_3_out_crop = center_crop(deconv3d_3_out, crop_d3, crop_h3, crop_w3);
    save_tensor_to_npy(deconv3d_3_out_crop, "deconv3d_3_out_crop.npy"); // okkk, mse=7.121739e-15

    // concat3 (in=[128,128], out=256), _____________________________________________
    tensor concat14_out = concat(conv3d_4_out_relu, deconv3d_3_out_crop); // <- 2 conv3d_4_out_relu

    // conv17 (in=256, out=128, kernel=3, stride=1, padding=1), _____________________________________________
    // create kernel18 tensor, shape (128,256,3,3,3)// read from file
    int num17 = 128;int channel17 = 256;int depth17 = 3;int height17 = 3;int width17 = 3;
    const char *filename34 = "savedWB2/35.txt";
    tensor kernel18 = create_tensor_from_file(filename34, num17, channel17, depth17, height17, width17);

    int num_b17 = 1;int channel_b17 = 128;int depth_b17 = 1;int height_b17 = 1;int width_b17 = 1;
    const char *filename35 = "savedWB2/34.txt";
    tensor bias18 = create_tensor_from_file(filename35, num_b17, channel_b17, depth_b17, height_b17, width_b17);

    int stride14 = 1; int padding18 = 1;
    tensor conv3d_17_out = conv3d(concat14_out, kernel18, bias18, stride14, padding18);
    tensor conv3d_17_out_relu = leakyrelu(conv3d_17_out, 0.01);
    save_tensor_to_npy(conv3d_17_out_relu, "conv3d_17_out_relu.npy"); // okkk, mse=7.402755e-14

    // conv18 (in=128, out=128, kernel=3, stride=1, padding=1), _____________________________________________
    // create kernel19 tensor, shape (128,128,3,3,3)// read from file
    int num18 = 128;int channel18 = 128;int depth18 = 3;int height18 = 3;int width18 = 3;
    const char *filename36 = "savedWB2/37.txt";
    tensor kernel19 = create_tensor_from_file(filename36, num18, channel18, depth18, height18, width18);

    int num_b18 = 1;int channel_b18 = 128;int depth_b18 = 1;int height_b18 = 1;int width_b18 = 1;
    const char *filename37 = "savedWB2/36.txt";
    tensor bias19 = create_tensor_from_file(filename37, num_b18, channel_b18, depth_b18, height_b18, width_b18);

    int stride15 = 1; int padding19 = 1;
    tensor conv3d_18_out = conv3d(conv3d_17_out_relu, kernel19, bias19, stride15, padding19);
    tensor conv3d_18_out_relu = leakyrelu(conv3d_18_out, 0.01);
    save_tensor_to_npy(conv3d_18_out_relu, "conv3d_18_out_relu.npy"); // okkk, mse=2.0867748e-13

    // deconv4 (in=128, out=64, kernel=2, stride=2, padding=0), _____________________________________________
    // create kernel20 tensor, shape (128,64,2,2,2)// read from file
    int num19 = 128;int channel19 = 64;int depth19 = 2;int height19 = 2;int width19 = 2;
    const char *filename38 = "savedWB2/39.txt";
    tensor kernel20 = create_tensor_from_file(filename38, num19, channel19, depth19, height19, width19);

    int num_b19 = 64;int channel_b19 = 1;int depth_b19 = 1;int height_b19 = 1;int width_b19 = 1;
    const char *filename39 = "savedWB2/38.txt";
    tensor bias20 = create_tensor_from_file(filename39, num_b19, channel_b19, depth_b19, height_b19, width_b19);

    int stride_d15 = 1; int stride_h15 = 2; int stride_w15 = 2; int padding_d15 = 0; int padding_h15 = 0; int padding_w15 = 0;
    tensor deconv3d_4_out = deconv3d(conv3d_18_out_relu, kernel20, bias20, stride_d15, stride_h15, stride_w15, padding_d15, padding_h15, padding_w15);
    // center crop
    int crop_d4 = conv3d_2_out_relu.depth; int crop_h4 = conv3d_2_out_relu.height; int crop_w4 = conv3d_2_out_relu.width;
    tensor deconv3d_4_out_crop = center_crop(deconv3d_4_out, crop_d4, crop_h4, crop_w4);
    save_tensor_to_npy(deconv3d_4_out_crop, "deconv3d_4_out_crop.npy"); // okkk, mse=4.8204598e-14

    // concat4 (in=[64,64], out=128), _____________________________________________
    tensor concat15_out = concat(conv3d_2_out_relu, deconv3d_4_out_crop); // <- 1 conv3d_2_out_relu

    // conv19 (in=128, out=64, kernel=3, stride=1, padding=1), _____________________________________________
    // create kernel21 tensor, shape (64,128,3,3,3)// read from file
    int num20 = 64;int channel20 = 128;int depth20 = 3;int height20 = 3;int width20 = 3;
    const char *filename40 = "savedWB2/41.txt";
    tensor kernel21 = create_tensor_from_file(filename40, num20, channel20, depth20, height20, width20);

    int num_b20 = 1;int channel_b20 = 64;int depth_b20 = 1;int height_b20 = 1;int width_b20 = 1;
    const char *filename41 = "savedWB2/40.txt";
    tensor bias21 = create_tensor_from_file(filename41, num_b20, channel_b20, depth_b20, height_b20, width_b20);

    int stride16 = 1; int padding20 = 1;
    tensor conv3d_19_out = conv3d(concat15_out, kernel21, bias21, stride16, padding20);
    tensor conv3d_19_out_relu = leakyrelu(conv3d_19_out, 0.01);
    save_tensor_to_npy(conv3d_19_out_relu, "conv3d_19_out_relu.npy"); // okkkk, mse=1.3523487e-13

    // conv20 (in=64, out=64, kernel=3, stride=1, padding=1), _____________________________________________
    // create kernel22 tensor, shape (64,64,3,3,3)// read from file
    int num21 = 64;int channel21 = 64;int depth21 = 3;int height21 = 3;int width21 = 3;
    const char *filename42 = "savedWB2/43.txt";
    tensor kernel22 = create_tensor_from_file(filename42, num21, channel21, depth21, height21, width21);

    int num_b21 = 1;int channel_b21 = 64;int depth_b21 = 1;int height_b21 = 1;int width_b21 = 1;
    const char *filename43 = "savedWB2/42.txt";
    tensor bias22 = create_tensor_from_file(filename43, num_b21, channel_b21, depth_b21, height_b21, width_b21);

    int stride17 = 1; int padding21 = 1;
    tensor conv3d_20_out = conv3d(conv3d_19_out_relu, kernel22, bias22, stride17, padding21);
    tensor conv3d_20_out_relu = leakyrelu(conv3d_20_out, 0.01);
    save_tensor_to_npy(conv3d_20_out_relu, "conv3d_20_out_relu.npy");

    // conv21 (in=64, out=4, kernel=1, stride=1, padding=0), _____________________________________________
    // create kernel23 tensor, shape (4,64,1,1,1)// read from file
    int num22 = 4;int channel22 = 64;int depth22 = 1;int height22 = 1;int width22 = 1;
    const char *filename44 = "savedWB2/45.txt";
    tensor kernel23 = create_tensor_from_file(filename44, num22, channel22, depth22, height22, width22);

    int num_b22 = 1;int channel_b22 = 4;int depth_b22 = 1;int height_b22 = 1;int width_b22 = 1;
    const char *filename45 = "savedWB2/44.txt";
    tensor bias23 = create_tensor_from_file(filename45, num_b22, channel_b22, depth_b22, height_b22, width_b22);

    int stride18 = 1; int padding22 = 0;
    tensor conv3d_21_out = conv3d(conv3d_20_out_relu, kernel23, bias23, stride18, padding22);
    save_tensor_to_npy(conv3d_21_out, "conv3d_21_out.npy");
    // no relu

    return conv3d_21_out;
}
    


int main() {
    // testconv3d(); // correct
    // testdeconv3d(); // correct
    // testmaxpool3d(); // correct
    // testconcat(); // correct
    // test_create_tensor_from_file(); // correct
    // test_center_crop(); // correct
    // tensor conv3d_21_out = net();
    // test_conv_relu();
    test_instance_norm3d();

    return 0;
}

/*
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1       [-1, 64, 64, 64, 64]           1,792
         LeakyReLU-2       [-1, 64, 64, 64, 64]               0
            Conv3d-3       [-1, 64, 64, 64, 64]         110,656
         LeakyReLU-4       [-1, 64, 64, 64, 64]               0
         MaxPool3d-5       [-1, 64, 64, 32, 32]               0
            Conv3d-6      [-1, 128, 64, 32, 32]         221,312
         LeakyReLU-7      [-1, 128, 64, 32, 32]               0
            Conv3d-8      [-1, 128, 64, 32, 32]         442,496
         LeakyReLU-9      [-1, 128, 64, 32, 32]               0
        MaxPool3d-10      [-1, 128, 32, 16, 16]               0
           Conv3d-11      [-1, 256, 32, 16, 16]         884,992
        LeakyReLU-12      [-1, 256, 32, 16, 16]               0
           Conv3d-13      [-1, 256, 32, 16, 16]       1,769,728
        LeakyReLU-14      [-1, 256, 32, 16, 16]               0
        MaxPool3d-15        [-1, 256, 16, 8, 8]               0
           Conv3d-16        [-1, 512, 16, 8, 8]       3,539,456
        LeakyReLU-17        [-1, 512, 16, 8, 8]               0
           Conv3d-18        [-1, 512, 16, 8, 8]       7,078,400
        LeakyReLU-19        [-1, 512, 16, 8, 8]               0
        MaxPool3d-20        [-1, 512, 16, 4, 4]               0
           Conv3d-21       [-1, 1024, 16, 4, 4]      14,156,800
        LeakyReLU-22       [-1, 1024, 16, 4, 4]               0
           Conv3d-23       [-1, 1024, 16, 4, 4]      28,312,576
        LeakyReLU-24       [-1, 1024, 16, 4, 4]               0
  ConvTranspose3d-25        [-1, 512, 17, 8, 8]       4,194,816
concat_with LeakyReLU-19       [-1, 1024, 16, 8, 8]               0
           Conv3d-27        [-1, 512, 16, 8, 8]      14,156,288
        LeakyReLU-28        [-1, 512, 16, 8, 8]               0
           Conv3d-29        [-1, 512, 16, 8, 8]       7,078,400
        LeakyReLU-30        [-1, 512, 16, 8, 8]               0
  ConvTranspose3d-31      [-1, 256, 32, 16, 16]       1,048,832
concat_with LeakyReLU-14      [-1, 512, 32, 16, 16]               0
           Conv3d-33      [-1, 256, 32, 16, 16]       3,539,200
        LeakyReLU-34      [-1, 256, 32, 16, 16]               0
           Conv3d-35      [-1, 256, 32, 16, 16]       1,769,728
        LeakyReLU-36      [-1, 256, 32, 16, 16]               0
  ConvTranspose3d-37      [-1, 128, 64, 32, 32]         262,272
concat_with LeakyReLU-9      [-1, 256, 64, 32, 32]               0
           Conv3d-39      [-1, 128, 64, 32, 32]         884,864
        LeakyReLU-40      [-1, 128, 64, 32, 32]               0
           Conv3d-41      [-1, 128, 64, 32, 32]         442,496
        LeakyReLU-42      [-1, 128, 64, 32, 32]               0
  ConvTranspose3d-43       [-1, 64, 65, 64, 64]          65,600
concat_with LeakyReLU-4      [-1, 128, 64, 64, 64]               0
           Conv3d-45       [-1, 64, 64, 64, 64]         221,248
        LeakyReLU-46       [-1, 64, 64, 64, 64]               0
           Conv3d-47       [-1, 64, 64, 64, 64]         110,656
        LeakyReLU-48       [-1, 64, 64, 64, 64]               0
           Conv3d-49        [-1, 4, 64, 64, 64]             260
================================================================
Total params: 90,292,868
Trainable params: 90,292,868
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 1.00
Forward/backward pass size (MB): 2401.25
Params size (MB): 344.44
Estimated Total Size (MB): 2746.69
----------------------------------------------------------------
*/


