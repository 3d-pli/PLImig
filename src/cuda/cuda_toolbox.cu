/*
    MIT License

    Copyright (c) 2021 Forschungszentrum Jülich / Jan André Reuter.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
 */

#include "cuda/cuda_toolbox.h"

__device__ void shellSort(float* array, uint low, uint high) {
    // Using the Ciura, 2001 sequence for best performance
    uint gaps[8] = {1, 4, 10, 23, 57, 132, 301, 701};
    if(low < high) {
        float* subArr = array + low;
        uint n = high - low;
        for (int pos = 7; pos > 0; --pos) {
            uint gap = gaps[pos];
            // Do a gapped insertion sort for this gap size.
            // The first gap elements a[0..gap-1] are already in gapped order
            // keep adding one more element until the entire array is
            // gap sorted
            for (uint i = gap; i < n; i += 1) {
                // add a[i] to the elements that have been gap sorted
                // save a[i] in temp and make a hole at position i
                float temp = subArr[i];

                // shift earlier gap-sorted elements up until the correct
                // location for a[i] is found
                uint j;
                for (j = i; j >= gap && subArr[j - gap] > temp; j -= gap) {
                    subArr[j] = subArr[j - gap];
                }

                // put temp (the original a[i]) in its correct location
                subArr[j] = temp;
            }
        }
    }
}

__global__ void medianFilterKernel(const float* image, int image_stride,
                                   float* result_image, int result_image_stride,
                                   int2 imageDims) {
    // Calculate actual position in image based on thread number and block number
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    // The valid values will be counted to ensure that the median will be calculated correctly
    uint validValues = 0;
    int cy_bound;
    // Median filter buffer
    float buffer[4 * MEDIAN_KERNEL_SIZE * MEDIAN_KERNEL_SIZE];

    // Only try to calculate the median of pixels within the non-padded image
    if(x >= MEDIAN_KERNEL_SIZE && x < imageDims.x - MEDIAN_KERNEL_SIZE && y >= MEDIAN_KERNEL_SIZE && y < imageDims.y - MEDIAN_KERNEL_SIZE) {
        // Transfer image pixels to our kernel for median filtering application
        for (int cx = -MEDIAN_KERNEL_SIZE; cx < MEDIAN_KERNEL_SIZE; ++cx) {
            // The median filter kernel is round. Therefore calculate the valid y-positions based on our x-position in the kernel
            cy_bound = sqrtf(MEDIAN_KERNEL_SIZE * MEDIAN_KERNEL_SIZE - cx * cx);
            for (int cy = -cy_bound; cy <= cy_bound; ++cy) {
                // Save values in buffer
                buffer[validValues] = image[x + cx + (y + cy) * image_stride];
                ++validValues;
            }
        }
        shellSort(buffer, 0, validValues);
        // Get middle value as our median
        result_image[x + y * result_image_stride] = buffer[validValues / 2];
    }
}

__global__ void medianFilterMaskedKernel(const float* image, int image_stride,
                                         float* result_image, int result_image_stride,
                                         const uchar* mask, int mask_stride,
                                         int2 imageDims) {
    // Calculate actual position in image based on thread number and block number
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    // The valid values will be counted to ensure that the median will be calculated correctly
    uint validValues = 0;
    int cy_bound;
    // Median filter buffer
    float buffer[4 * MEDIAN_KERNEL_SIZE * MEDIAN_KERNEL_SIZE];

    // Only try to calculate the median of pixels within the non-padded image
    if(x >= MEDIAN_KERNEL_SIZE && x < imageDims.x - MEDIAN_KERNEL_SIZE && y >= MEDIAN_KERNEL_SIZE && y < imageDims.y - MEDIAN_KERNEL_SIZE) {
        // Transfer image pixels to our kernel for median filtering application
        for (int cx = -MEDIAN_KERNEL_SIZE; cx < MEDIAN_KERNEL_SIZE; ++cx) {
            // The median filter kernel is round. Therefore calculate the valid y-positions based on our x-position in the kernel
            cy_bound = sqrtf(MEDIAN_KERNEL_SIZE * MEDIAN_KERNEL_SIZE - cx * cx);
            for (int cy = -cy_bound; cy <= cy_bound; ++cy) {
                // If the pixel in the kernel matches the current pixel on the gray / white mask
                if (mask[x + y * mask_stride] == mask[x + cx + (y + cy) * image_stride]) {
                    // Save values in buffer
                    buffer[validValues] = image[x + cx + (y + cy) * image_stride];
                    ++validValues;
                }
            }
        }
        // Depending on the number of valid values, calculate the median, save the pixel value itself or save zero
        if (validValues > 1) {
            shellSort(buffer, 0, validValues);
            result_image[x + y * result_image_stride] = buffer[validValues / 2];
        } else if (validValues == 1) {
            result_image[x + y * result_image_stride] = buffer[0];
        } else {
            result_image[x + y * result_image_stride] = 0;
        }
    }
}

__global__ void connectedComponentsInitializeMask(const uchar* image, int image_stride,
                                                  uint* mask, int mask_stride,
                                                  int line_width) {
    // Calculate actual position in image based on thread number and block number
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if(image[x + y * image_stride] != 0) {
        mask[x + y * mask_stride] = y * uint(line_width) + x + 1;
    } else {
        mask[x + y * mask_stride] = 0;
    }
}

__global__ void connectedComponentsIteration(uint* mask, int mask_stride, int2 maskDims, volatile bool* changeOccured) {
    // Calculate actual position in image based on thread number and block number
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    uint minVal;
    if(mask[x + y * mask_stride] != 0) {
        minVal = mask[x + y * mask_stride];

        if(int(x - 1) >= 0 && mask[x-1 + y * mask_stride] != 0) {
            minVal = min(minVal, mask[x-1 + y * mask_stride]);
        }
        if(int(x + 1) < maskDims.x && mask[x+1 + y * mask_stride] != 0) {
            minVal = min(minVal, mask[x+1 + y * mask_stride]);
        }
        if(int(y - 1) >= 0 && mask[x + (y-1) * mask_stride] != 0) {
            minVal = min(minVal, mask[x + (y-1) * mask_stride]);
        }
        if(int(y + 1) < maskDims.y && mask[x + (y+1) * mask_stride] != 0) {
            minVal = min(minVal, mask[x + (y+1) * mask_stride]);
        }

        if(minVal != mask[x + y * mask_stride]) {
            mask[x + y * mask_stride] = minVal;
            *changeOccured = true;
        }
    }
}

__global__ void connectedComponentsReduceComponents(uint* mask, int mask_stride,
                                                    const uint* lutKeys,
                                                    const uint lutSize) {
    // Calculate actual position in image based on thread number and block number
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    for (uint i = 0; i < lutSize; ++i) {
        if(mask[x + y * mask_stride] == lutKeys[i]) {
            mask[x + y * mask_stride] = i;
            break;
        }
    }
}

cv::Mat PLImg::cuda::labeling::callCUDAConnectedComponents(const cv::Mat& image) {
    cv::Mat result = cv::Mat(image.rows, image.cols, CV_32SC1);

    uchar* deviceImage;
    uint* deviceMask;
    bool* deviceChangeOccured;
    bool changeOccured;

    CHECK_CUDA(cudaMalloc(&deviceImage, image.total() * sizeof(uchar)));
    CHECK_CUDA(cudaMemcpy(deviceImage, image.data, image.total() * sizeof(uchar), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&deviceMask, image.total() * sizeof(uint)));
    CHECK_CUDA(cudaMalloc(&deviceChangeOccured, sizeof(bool)));

    dim3 threadsPerBlock, numBlocks;
    threadsPerBlock = dim3(1, 1);
    numBlocks = dim3(ceil(float(image.cols) / threadsPerBlock.x), ceil(float(image.rows) / threadsPerBlock.y));

    connectedComponentsInitializeMask<<<numBlocks, threadsPerBlock>>>(deviceImage, image.cols, deviceMask, image.cols, image.cols);
    CHECK_CUDA(cudaFree(deviceImage));
    uint i = 0;
    do {
        std::cout << std::to_string(++i) + "th step..." << std::endl;
        CHECK_CUDA(cudaMemset(deviceChangeOccured, false, sizeof(bool)));
        connectedComponentsIteration<<<numBlocks, threadsPerBlock>>>(deviceMask, image.cols, {image.cols, image.rows},
                                                                     deviceChangeOccured);
        CHECK_CUDA(cudaMemcpy(&changeOccured, deviceChangeOccured, sizeof(bool), cudaMemcpyDeviceToHost));
    } while(changeOccured);
    CHECK_CUDA(cudaFree(deviceChangeOccured));

    uint* deviceUniqueMask;
    CHECK_CUDA(cudaMalloc(&deviceUniqueMask, image.total() * sizeof(uint)));
    CHECK_CUDA(cudaMemcpy(deviceUniqueMask, deviceMask, image.total() * sizeof(uint), cudaMemcpyDeviceToDevice));
    thrust::sort(thrust::device, deviceUniqueMask, deviceUniqueMask + image.total());
    uint* newEnd = thrust::unique(thrust::device, deviceUniqueMask, deviceUniqueMask + image.total());

    connectedComponentsReduceComponents<<<numBlocks, threadsPerBlock>>>(deviceMask, image.cols,
                                                                        deviceUniqueMask,
                                                                        thrust::distance(deviceUniqueMask, newEnd));
    CHECK_CUDA(cudaFree(deviceUniqueMask));

    // Copy result from GPU back to CPU
    CHECK_CUDA(cudaMemcpy(result.data, deviceMask, image.total() * sizeof(uint), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(deviceMask));

    return result;
}

std::shared_ptr<cv::Mat> PLImg::cuda::filters::callCUDAmedianFilter(const std::shared_ptr<cv::Mat>& image) {
    // Create a result image with the same dimensions as our input image
    cv::Mat result = cv::Mat(image->rows, image->cols, image->type());
    // Expand borders of input image inplace to ensure that the median algorithm can run correcly
    cv::copyMakeBorder(*image, *image, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, cv::BORDER_REPLICATE);

    // The image might be too large to be saved completely in the video memory.
    // Therefore chunks will be used if the amount of memory is too small.
    uint numberOfChunks = 1;
    // Check the free video memory
    ulong freeMem;
    CHECK_CUDA(cudaMemGetInfo(&freeMem, nullptr));
    // If the total free memory is smaller than the estimated amount of memory, calculate the number of chunks
    // with the power of four (1, 4, 16, 256, 1024, ...)
    if(double(image->total()) * image->elemSize() * 2.1 > double(freeMem)) {
        numberOfChunks = fmax(1, pow(4.0, ceil(log(image->total() * image->elemSize() * 2.1 / double(freeMem)) / log(4))));
    }
    // Each dimensions will get the same number of chunks. Calculate them by using the square root.
    uint chunksPerDim = fmax(1, sqrtf(numberOfChunks));

    float* deviceImage, *deviceResult;
    uint xMin, xMax, yMin, yMax;
    ulong nSrcStep, nResStep;
    // Apply median filter
    // Calculate offsets for image and result. Starting at the edge would result in errors because we would
    // go out of bounds.
    dim3 threadsPerBlock, numBlocks;

    // We've increased the image dimensions earlier. Save the original image dimensions for further calculations.
    int2 realImageDims = {image->cols - 2 * MEDIAN_KERNEL_SIZE, image->rows - 2 * MEDIAN_KERNEL_SIZE};
    int2 subImageDims;

    cv::Mat subImage, subResult, croppedImage;
    // For each chunk
    for(uint it = 0; it < numberOfChunks; ++it) {
        std::cout << "\rCurrent chunk: " << it+1 << "/" << numberOfChunks;
        std::flush(std::cout);
        // Calculate image boarders
        xMin = (it % chunksPerDim) * realImageDims.x / chunksPerDim;
        xMax = fmin((it % chunksPerDim + 1) * realImageDims.x / chunksPerDim, realImageDims.x);
        yMin = (it / chunksPerDim) * realImageDims.y / chunksPerDim;
        yMax = fmin((it / chunksPerDim + 1) * realImageDims.y / chunksPerDim, realImageDims.y);

        // Get chunk of our image and result. Apply padding to the result to ensure that the median filter will run correctly.
        croppedImage = cv::Mat(*image, cv::Rect(xMin, yMin, xMax - xMin + 2 * MEDIAN_KERNEL_SIZE, yMax - yMin + 2 * MEDIAN_KERNEL_SIZE));
        croppedImage.copyTo(subImage);
        croppedImage = cv::Mat(result, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
        croppedImage.copyTo(subResult);
        cv::copyMakeBorder(subResult, subResult, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, cv::BORDER_REPLICATE);

        // Allocate GPU memory for the original image and its result
        CHECK_CUDA(cudaMalloc((void **) &deviceImage, subImage.total() * subImage.elemSize()));
        // Length of columns
        nSrcStep = subImage.cols;

        CHECK_CUDA(cudaMalloc((void **) &deviceResult, subImage.total() * subImage.elemSize()));
        // Length of columns
        nResStep = subResult.cols;

        // Copy image from CPU to GPU
        CHECK_CUDA(cudaMemcpy(deviceImage, subImage.data, subImage.total() * subImage.elemSize(), cudaMemcpyHostToDevice));

        // Apply median filter
        subImageDims = {subImage.cols, subImage.rows};
        threadsPerBlock = dim3(CUDA_KERNEL_NUM_THREADS, CUDA_KERNEL_NUM_THREADS);
        numBlocks = dim3(ceil(float(subImageDims.x) / threadsPerBlock.x), ceil(float(subImageDims.y) / threadsPerBlock.y));
        // Run median filter
        medianFilterKernel<<<numBlocks, threadsPerBlock>>>(deviceImage, nSrcStep,
                                                           deviceResult, nResStep,
                                                           subImageDims);

        // Copy result from GPU back to CPU
        CHECK_CUDA(cudaMemcpy(subResult.data, deviceResult, subImage.total() * subImage.elemSize(), cudaMemcpyDeviceToHost));

        // Free reserved memory
        CHECK_CUDA(cudaFree(deviceImage));
        CHECK_CUDA(cudaFree(deviceResult));
        CHECK_CUDA(cudaDeviceSynchronize());

        // Calculate the range where the median filter was applied and where the chunk will be placed.
        cv::Rect srcRect = cv::Rect(MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, xMax - xMin, yMax - yMin);
        cv::Rect dstRect = cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin);
        subResult(srcRect).copyTo(result(dstRect));
    }
    // Fix output after \r
    std::cout << std::endl;
    // Revert the padding of the original image
    croppedImage = cv::Mat(*image, cv::Rect(MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, image->cols - 2*MEDIAN_KERNEL_SIZE, image->rows - 2*MEDIAN_KERNEL_SIZE));
    croppedImage.copyTo(*image);

    // Return resulting median filtered image
    return std::make_shared<cv::Mat>(result);
}

std::shared_ptr<cv::Mat> PLImg::cuda::filters::callCUDAmedianFilterMasked(const std::shared_ptr<cv::Mat>& image, const std::shared_ptr<cv::Mat>& mask) {
    // Copy the result back to the CPU
    cv::Mat result = cv::Mat(image->rows, image->cols, image->type());
    cv::copyMakeBorder(*image, *image, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(*mask, *mask, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, cv::BORDER_REPLICATE);

    // The image might be too large to be saved completely in the video memory.
    // Therefore chunks will be used if the amount of memory is too small.
    uint numberOfChunks = 1;
    ulong freeMem;
    // Check the free video memory
    CHECK_CUDA(cudaMemGetInfo(&freeMem, nullptr));
    // If the total free memory is smaller than the estimated amount of memory, calculate the number of chunks
    // with the power of four (1, 4, 16, 256, 1024, ...)
    if(double(image->total()) * image->elemSize() * 3.1 > double(freeMem)) {
        numberOfChunks = fmax(1, pow(4.0, ceil(log(image->total() * image->elemSize() * 3.1 / double(freeMem)) / log(4))));
    }
    // Each dimensions will get the same number of chunks. Calculate them by using the square root.
    uint chunksPerDim = fmax(1, sqrtf(numberOfChunks));

    float* deviceImage, *deviceResult;
    uchar* deviceMask;
    uint xMin, xMax, yMin, yMax;
    ulong nSrcStep, nMaskStep, nResStep;
    // Apply median filter
    // Calculate offsets for image and result. Starting at the edge would result in errors because we would
    // go out of bounds.
    dim3 threadsPerBlock, numBlocks;

    // We've increased the image dimensions earlier. Save the original image dimensions for further calculations.
    int2 realImageDims = {image->cols - 2 * MEDIAN_KERNEL_SIZE, image->rows - 2 * MEDIAN_KERNEL_SIZE};
    int2 subImageDims;

    cv::Mat subImage, subMask, subResult, croppedImage;
    for(uint it = 0; it < numberOfChunks; ++it) {
        std::cout << "\rCurrent chunk: " << it+1 << "/" << numberOfChunks;
        std::flush(std::cout);
        // Calculate image boarders
        xMin = (it % chunksPerDim) * realImageDims.x / chunksPerDim;
        xMax = fmin((it % chunksPerDim + 1) * realImageDims.x / chunksPerDim, realImageDims.x);
        yMin = (it / chunksPerDim) * realImageDims.y / chunksPerDim;
        yMax = fmin((it / chunksPerDim + 1) * realImageDims.y / chunksPerDim, realImageDims.y);

        // Get chunk of our image, mask and result. Apply padding to the result to ensure that the median filter will run correctly.
        croppedImage = cv::Mat(*image, cv::Rect(xMin, yMin, xMax - xMin + 2 * MEDIAN_KERNEL_SIZE, yMax - yMin + 2 * MEDIAN_KERNEL_SIZE));
        croppedImage.copyTo(subImage);
        croppedImage = cv::Mat(*mask, cv::Rect(xMin, yMin, xMax - xMin + 2 * MEDIAN_KERNEL_SIZE, yMax - yMin + 2 * MEDIAN_KERNEL_SIZE));
        croppedImage.copyTo(subMask);
        croppedImage = cv::Mat(result, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
        croppedImage.copyTo(subResult);
        cv::copyMakeBorder(subResult, subResult, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, cv::BORDER_REPLICATE);

        // Allocate GPU memory for the original image, mask and its result
        CHECK_CUDA(cudaMalloc((void **) &deviceImage, subImage.total() * subImage.elemSize()));
        // Length of columns
        nSrcStep = subImage.cols;

        CHECK_CUDA(cudaMalloc((void **) &deviceMask, subMask.total() * subMask.elemSize()));
        // Length of columns
        nMaskStep = subMask.cols;

        CHECK_CUDA(cudaMalloc((void **) &deviceResult, subImage.total() * subImage.elemSize()));
        // Length of columns
        nResStep = subResult.cols;

        // Copy image from CPU to GPU
        CHECK_CUDA(cudaMemcpy(deviceImage, subImage.data, subImage.total() * subImage.elemSize(), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(deviceMask, subMask.data, subMask.total() * subMask.elemSize(), cudaMemcpyHostToDevice));

        // Apply median filter
        subImageDims = {subImage.cols, subImage.rows};
        threadsPerBlock = dim3(CUDA_KERNEL_NUM_THREADS, CUDA_KERNEL_NUM_THREADS);
        numBlocks = dim3(ceil(float(subImageDims.x) / threadsPerBlock.x), ceil(float(subImageDims.y) / threadsPerBlock.y));
        // Run median filter
        medianFilterMaskedKernel<<<numBlocks, threadsPerBlock>>>(deviceImage, nSrcStep,
                                                                 deviceResult, nResStep,
                                                                 deviceMask, nMaskStep,
                                                                 subImageDims);

        CHECK_CUDA(cudaMemcpy(subResult.data, deviceResult, subImage.total() * subImage.elemSize(), cudaMemcpyDeviceToHost));

        // Free reserved memory
        CHECK_CUDA(cudaFree(deviceImage));
        CHECK_CUDA(cudaFree(deviceResult));
        CHECK_CUDA(cudaFree(deviceMask));
        CHECK_CUDA(cudaDeviceSynchronize());

        cv::Rect srcRect = cv::Rect(MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, subResult.cols - 2*MEDIAN_KERNEL_SIZE, subResult.rows - 2*MEDIAN_KERNEL_SIZE);
        cv::Rect dstRect = cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin);

        subResult(srcRect).copyTo(result(dstRect));
    }
    // Fix output after \r
    std::cout << std::endl;

    croppedImage = cv::Mat(*image, cv::Rect(MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, image->cols - 2*MEDIAN_KERNEL_SIZE, image->rows - 2*MEDIAN_KERNEL_SIZE));
    croppedImage.copyTo(*image);
    croppedImage = cv::Mat(*mask, cv::Rect(MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, mask->cols - 2*MEDIAN_KERNEL_SIZE, mask->rows - 2*MEDIAN_KERNEL_SIZE));
    croppedImage.copyTo(*mask);
    return std::make_shared<cv::Mat>(result);
}


