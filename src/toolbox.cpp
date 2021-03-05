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

#include "toolbox.h"

int PLImg::Histogram::peakWidth(cv::Mat hist, int peakPosition, float direction, float targetHeight) {
    float height = hist.at<float>(peakPosition) * targetHeight;
    int i = peakPosition;
    if(direction > 0) {
        while(i < hist.rows && hist.at<float>(i) > height) {
            ++i;
        }
        return i - peakPosition;
    } else {
        while(i > 0 && hist.at<float>(i) > height) {
            --i;
        }
        return peakPosition - i;
    }
}

float PLImg::Histogram::maxCurvature(cv::Mat hist, float histLow, float histHigh, float direction, int start, int stop) {
    float stepSize = (histHigh - histLow) / float(hist.rows);
    if(stop - start > 3) {
        auto maxIterator = std::max_element(hist.begin<float>() + start, hist.begin<float>() + stop);
        int maxPos = maxIterator - hist.begin<float>();
        int width = fmax(1.0f, peakWidth(hist, maxPos, direction));

        int roiStart, roiEnd;
        if(direction > 0) {
            roiStart = maxPos;
            roiEnd = std::min(maxPos + 10 * width, stop);
        } else {
            roiStart = std::max(start, maxPos - 10 * width);
            roiEnd = maxPos;
        }
        cv::Mat kappa;
        if(roiEnd - roiStart > 3) {
            kappa = cv::Mat(roiEnd - roiStart, 1, CV_32FC1);
            float d1, d2;
            #pragma omp parallel for private(d1, d2)
            for (int i = 1; i < kappa.rows - 1; ++i) {
                d1 = (hist.at<float>(roiStart + i + 1) - hist.at<float>(roiStart + i)) / stepSize;
                d2 = (hist.at<float>(roiStart + i + 1) - 2 * hist.at<float>(roiStart + i) +
                      hist.at<float>(roiStart + i - 1)) / pow(stepSize, 2.0f);
                kappa.at<float>(i) = d2 / pow(1 + pow(d1, 2.0f), 3.0f / 2.0f);
            }
        } else {
            return histLow + float(roiStart + 1) * stepSize;
        }
        auto minKappa = std::max_element(kappa.begin<float>()+1, kappa.end<float>()-1);
        int minPos = minKappa - kappa.begin<float>();
        return histLow + float(roiStart + minPos) * stepSize;
    } else {
        return histLow + float(start) * stepSize;
    }
}

std::vector<unsigned> PLImg::Histogram::peaks(cv::Mat hist, int start, int stop, float minSignificance) {
    std::vector<unsigned> peaks;

    int posAhead;
    // find all peaks
    for(int pos = start + 1; pos < stop-1; ++pos) {
        if(hist.at<int>(pos) - hist.at<int>(pos-1) > 0) {
            posAhead = pos + 1;

            while(posAhead < hist.rows && hist.at<int>(pos) == hist.at<int>(posAhead)) {
                ++posAhead;
            }

            if(hist.at<int>(pos) - hist.at<int>(posAhead) > 0) {
                peaks.push_back((pos + posAhead - 1) / 2);
            }
        }
    }

    float maxElem = *std::max_element(hist.begin<float>(), hist.end<float>());

    // filter peaks by prominence
    for(int i = peaks.size()-1; i >= 0; --i) {
        float left_min = hist.at<float>(peaks.at(i));
        if(left_min == maxElem) {
            continue;
        }
        int left_i = peaks.at(i) - 1;
        while(left_i > 0 && hist.at<float>(left_i) <= hist.at<float>(peaks.at(i))) {
            if(hist.at<float>(left_i) < left_min) {
                left_min = hist.at<float>(left_i);
            }
            --left_i;
        }

        float right_min = hist.at<float>(peaks.at(i));
        int right_i = peaks.at(i) + 1;
        while(right_i < hist.rows && hist.at<float>(right_i) <= hist.at<float>(peaks.at(i))) {
            if(hist.at<float>(right_i) < right_min) {
                right_min = hist.at<float>(right_i);
            }
            ++right_i;
        }

        float prominence = hist.at<float>(peaks.at(i)) - fmax(left_min, right_min);
        if(prominence < minSignificance) {
            peaks.erase(peaks.begin() + i);
        }
    }

    return peaks;
}

cv::Mat PLImg::Image::largestAreaConnectedComponents(const cv::Mat& image, cv::Mat mask, float percentPixels) {
    float pixelThreshold;
    if(mask.empty()) {
        pixelThreshold = float(image.cols) * float(image.rows) * percentPixels / 100;
        mask = cv::Mat::ones(image.rows, image.cols, CV_8UC1);
    } else {
        std::cout << float(cv::countNonZero(mask)) << " " << float(image.cols) * float(image.rows) << std::endl;
        pixelThreshold = float(cv::countNonZero(mask)) * percentPixels / 100;
    }

    int channels[] = {0};
    float histBounds[] = {0.0f, 1.0f};
    const float* histRange = { histBounds };
    int histSize = MAX_NUMBER_OF_BINS;

    cv::Mat hist;
    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    uint front_bin = hist.rows - 1;
    uint pixelSum = 0;
    while(pixelSum < 2 * pixelThreshold && front_bin > 0) {
        pixelSum += uint(hist.at<float>(front_bin));
        --front_bin;
    }

    cv::Mat cc_mask, labels;
    std::pair<cv::Mat, int> component;

    uint front_bin_max = front_bin;
    uint front_bin_min = 0;

    while(int(front_bin_max) - int(front_bin_min) > 0) {
        cc_mask = (image > float(front_bin)/MAX_NUMBER_OF_BINS) & mask;
        if(float(cv::countNonZero(cc_mask)) > pixelThreshold) {
            labels = PLImg::cuda::labeling::connectedComponents(cc_mask);
            cc_mask.release();
            component = PLImg::cuda::labeling::largestComponent(labels);
            labels.release();

            std::cout << component.second << " " << pixelThreshold << std::endl;
            std::flush(std::cout);

            if (component.second < pixelThreshold * 0.9) {
                front_bin_max = front_bin;
                front_bin = fmin(front_bin - float(front_bin_max - front_bin_min) / 2, front_bin - 1);
            } else if (component.second > pixelThreshold * 1.1) {
                front_bin_min = front_bin;
                front_bin = fmax(front_bin + 1, front_bin + float(front_bin_max - front_bin_min) / 2);
            } else {
                std::cout << std::endl;
                return component.first;
            }
        } else {
            front_bin = front_bin - 1;
        }
        std::cout << "front bin = " << front_bin / MAX_NUMBER_OF_BINS << std::endl;
    }
    // No search result during the while loop
    if (component.first.empty()) {
        return cv::Mat::ones(image.rows, image.cols, CV_8UC1);
    } else {
        return component.first;
    }
}

bool PLImg::cuda::runCUDAchecks() {
    static bool didRunCudaChecks = false;
    if(!didRunCudaChecks) {
        cudaError_t err;

        printf("Checking if CUDA is running as expected.\n");

        const NppLibraryVersion *libVer = nppGetLibVersion();
        printf("NPP  Library Version: %d.%d.%d\n", libVer->major, libVer->minor,
               libVer->build);

        int driverVersion, runtimeVersion;
        err = cudaDriverGetVersion(&driverVersion);
        if(err != cudaSuccess) {
            std::cerr << "Could not get driver version! \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        printf("CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
               (driverVersion % 100) / 10);

        err = cudaRuntimeGetVersion(&runtimeVersion);
        if(err != cudaSuccess) {
            std::cerr << "Could not get runtime version! \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
               (runtimeVersion % 100) / 10);

        // Min spec is SM 1.0 devices
        cudaDeviceProp deviceProperties{};
        err = cudaGetDeviceProperties(&deviceProperties, 0);
        if(err != cudaSuccess) {
            std::cerr << "Could not get device properties! \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        printf("Compute capability: %d,%d\n", deviceProperties.major, deviceProperties.minor);
        printf("Total memory: %.3f MiB\n", deviceProperties.totalGlobalMem / 1024.0 / 1024.0);
        didRunCudaChecks = true;
    }
    return true;

}

ulong PLImg::cuda::getFreeMemory() {
    PLImg::cuda::runCUDAchecks();
    ulong free;
    cudaError_t err;
    err = cudaMemGetInfo(&free, nullptr);
    if(err != cudaSuccess) {
        std::cerr << "Could not get free memory! \n";
        std::cerr << cudaGetErrorName(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    return free;
}

ulong PLImg::cuda::getTotalMemory() {
    PLImg::cuda::runCUDAchecks();
    ulong total;
    cudaError_t err;
    err = cudaMemGetInfo(nullptr, &total);
    if(err != cudaSuccess) {
        std::cerr << "Could not get total memory! \n";
        std::cerr << cudaGetErrorName(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    return total;
}

std::shared_ptr<cv::Mat> PLImg::cuda::filters::medianFilter(const std::shared_ptr<cv::Mat>& image) {
    PLImg::cuda::runCUDAchecks();
    return callCUDAmedianFilter(image);

}

std::shared_ptr<cv::Mat> PLImg::cuda::filters::medianFilterMasked(const std::shared_ptr<cv::Mat>& image,
                                                            const std::shared_ptr<cv::Mat>& mask) {
    PLImg::cuda::runCUDAchecks();
    return callCUDAmedianFilterMasked(image, mask);
}

cv::Mat PLImg::cuda::labeling::connectedComponents(const cv::Mat &image) {
    PLImg::cuda::runCUDAchecks();
    cv::Mat result = cv::Mat(image.rows, image.cols, CV_32SC1);

    // Error objects
    cudaError_t err;
    NppStatus errCode;

    // Calculate the number of chunks for the Connected Components algorithm
    Npp32u numberOfChunks = 1;
    Npp32u chunksPerDim;
    Npp32f predictedMemoryUsage = float(image.total()) * float(image.elemSize()) + 2 * float(image.total()) * float(sizeof(Npp32u))
                                    + float(size_t(image.rows) * size_t(image.cols) * 9);
    if (predictedMemoryUsage > double(PLImg::cuda::getFreeMemory())) {
        numberOfChunks = fmax(numberOfChunks, pow(4, ceil(log(predictedMemoryUsage / double(PLImg::cuda::getFreeMemory())) / log(4))));
    }
    chunksPerDim = fmax(1, numberOfChunks/sqrt(numberOfChunks));

    // Chunked connected components algorithm.
    // Labels right on the edges will be wrong. This will be fixed in the next step.
    Npp32u *deviceResult;
    Npp8u *deviceBuffer, *deviceImage;
    Npp32s nSrcStep, nDstStep, pSrcOffset, pDstOffset;
    NppiSize roi;
    Npp32s xMin, xMax, yMin, yMax;
    Npp32s nextLabelNumber = 0;
    Npp32s maxLabelNumber = 0;
    cv::Mat subImage, subResult, subMask, croppedImage;
    for(Npp32u it = 0; it < numberOfChunks; ++it) {
        //std::cout << "\rCurrent chunk: " << it+1 << "/" << numberOfChunks;
        std::flush(std::cout);
        // Calculate image boarders
        xMin = (it % chunksPerDim) * image.cols / chunksPerDim;
        xMax = fmin((it % chunksPerDim + 1) * image.cols / chunksPerDim, image.cols);
        yMin = (it / chunksPerDim) * image.rows / chunksPerDim;
        yMax = fmin((it / chunksPerDim + 1) * image.rows / chunksPerDim, image.rows);

        croppedImage = cv::Mat(image, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
        croppedImage.copyTo(subImage);
        croppedImage = cv::Mat(result, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
        croppedImage.copyTo(subResult);
        croppedImage.release();

        cv::copyMakeBorder(subImage, subImage, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
        cv::copyMakeBorder(subResult, subResult, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);

        // Reserve memory on GPU for image and result image
        // Image
        err = cudaMalloc((void **) &deviceImage, subImage.total() * subImage.elemSize());
        if (err != cudaSuccess) {
            std::cerr << "Could not allocate enough memory for original mask \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        // Length of columns
        nSrcStep = sizeof(Npp8u) * subImage.cols;

        // Result
        err = cudaMalloc((void **) &deviceResult, subImage.total() * sizeof(Npp32u));
        if (err != cudaSuccess) {
            std::cerr << "Could not allocate enough memory for result \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        // Length of columns
        nDstStep = sizeof(Npp32u) * subImage.cols;

        // Copy image from CPU to GPU
        err = cudaMemcpy(deviceImage, subImage.data, subImage.total() * subImage.elemSize(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Could not copy image from host to device \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        roi = {subImage.cols - 2, subImage.rows - 2};
        // Calculate offsets for image and result. Starting at the edge would result in errors because we would
        // go out of bounds.
        pSrcOffset = 1 + 1 * nSrcStep / sizeof(Npp8u);
        pDstOffset = 1 + 1 * nDstStep / sizeof(Npp32u);

        err = cudaMalloc((void **) &deviceBuffer, size_t(roi.width) * size_t(roi.height) * 9);
        if (err != cudaSuccess) {
            std::cerr << "Could not generate buffer for Connected Components application. Error code is: ";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        errCode = nppiLabelMarkersUF_8u32u_C1R(deviceImage + pSrcOffset, nSrcStep, deviceResult + pDstOffset, nDstStep, roi, nppiNormInf, deviceBuffer);
        if (errCode != NPP_SUCCESS) {
            printf("NPP error: Could not create labeling : %d\n", errCode);
            exit(EXIT_FAILURE);
        }
        cudaFree(deviceImage);

        errCode = nppiCompressMarkerLabels_32u_C1IR(deviceResult + pDstOffset, nDstStep, roi, roi.height * roi.width, &maxLabelNumber, deviceBuffer);
        if (errCode != NPP_SUCCESS) {
            printf("NPP error: Could not compress label markers : %d\n", errCode);
            exit(EXIT_FAILURE);
        }

        // Copy the result back to the CPU
        err = cudaMemcpy(subResult.data, deviceResult, subImage.total() * sizeof(Npp32u), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Could not copy image from device to host \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        // Increase label number according to the previous chunk. Set background back to 0
        subMask = subResult == 0;
        subResult = subResult + cv::Scalar(nextLabelNumber, 0, 0);
        subResult.setTo(0, subMask);
        nextLabelNumber = nextLabelNumber + maxLabelNumber;

        // Free reserved memory
        cudaFree(deviceResult);
        cudaFree(deviceBuffer);

        cv::Rect srcRect = cv::Rect(1, 1, subResult.cols - 2, subResult.rows - 2);
        cv::Rect dstRect = cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin);
        subResult(srcRect).copyTo(result(dstRect));
    }

    // Iterate along the borders of each chunk to check if any labels overlap there. If that's the case
    // replace the higher numbered label by the lower numbered label. Only apply if more than one chunk is present.
    if(numberOfChunks > 1) {
        bool somethingDidChange = true;
        int minVal;
        while(somethingDidChange) {
            somethingDidChange = false;
            for (uint chunk = 0; chunk < numberOfChunks; ++chunk) {
                xMin = (chunk % chunksPerDim) * result.cols / chunksPerDim;
                xMax = fmin((chunk % chunksPerDim + 1) * result.cols / chunksPerDim, result.cols-1);
                yMin = (chunk / chunksPerDim) * image.rows / chunksPerDim;
                yMax = fmin((chunk / chunksPerDim + 1) * result.rows / chunksPerDim, result.rows-1);

                // Check upper and lower border
                for (uint x = xMin; x < xMax; ++x) {
                    if (result.at<int>(yMin, x) > 0 && yMin - 1 >= 0) {
                        if (result.at<int>(yMin - 1, x) > 0 && result.at<int>(yMin, x) != result.at<int>(yMin - 1, x)) {
                            minVal = fmin(result.at<int>(yMin, x), result.at<int>(yMin - 1, x));
                            result.setTo(minVal, result == result.at<int>(yMin, x));
                            result.setTo(minVal, result == result.at<int>(yMin - 1, x));
                            somethingDidChange = true;
                        }
                    }

                    if (result.at<int>(yMax, x) > 0 && yMax + 1 < result.rows) {
                        if (result.at<int>(yMax + 1, x) > 0 && result.at<int>(yMax, x) != result.at<int>(yMax + 1, x)) {
                            minVal = fmin(result.at<int>(yMax, x), result.at<int>(yMax + 1, x));
                            result.setTo(minVal, result == result.at<int>(yMax, x));
                            result.setTo(minVal, result == result.at<int>(yMax + 1, x));
                            somethingDidChange = true;
                        }
                    }
                }

                // Check left and right border
                for (uint y = yMin; y < yMax; ++y) {
                    if (result.at<int>(y, xMin) > 0 && xMin - 1 >= 0) {
                        if (result.at<int>(y, xMin - 1) > 0 && result.at<int>(y, xMin) != result.at<int>(y, xMin - 1)) {
                            minVal = fmin(result.at<int>(y, xMin),result.at<int>(y, xMin - 1));
                            result.setTo(minVal, result == result.at<int>(y, xMin));
                            result.setTo(minVal, result == result.at<int>(y, xMin - 1));
                            somethingDidChange = true;
                        }
                    }

                    if (result.at<int>(y, xMax) > 0 && xMax + 1 < result.cols) {
                        if (result.at<int>(y, xMax+1) > 0 && result.at<int>(y, xMax+1) != result.at<int>(y, xMax)) {
                            minVal = fmin(result.at<int>(y, xMax), result.at<int>(y, xMax+1));
                            result.setTo(minVal, result == result.at<int>(y, xMax));
                            result.setTo(minVal, result == result.at<int>(y, xMax+1));
                            somethingDidChange = true;
                        }
                    }
                }
            }
        }
    }
    return result;
}

std::pair<cv::Mat, int> PLImg::cuda::labeling::largestComponent(const cv::Mat &connectedComponentsImage) {
    PLImg::cuda::runCUDAchecks();

    // Check how many labels are present in the given image.
    uint numLabels = 0;
    #pragma omp parallel reduction(max:numLabels) shared(connectedComponentsImage)
    {
        uint numThreads = omp_get_num_threads();
        uint myThread = omp_get_thread_num();
        uint numElements = std::distance(connectedComponentsImage.begin<int>(), connectedComponentsImage.end<int>());
        uint myStart = numElements / numThreads * myThread;
        uint myEnd = fmin(numElements, numElements / numThreads * (myThread + 1));
        numLabels = *std::max_element(connectedComponentsImage.begin<int>() + myStart, connectedComponentsImage.begin<int>() + myEnd);
    }

    // If more than one label is present, continue to find the largest component
    if(numLabels > 1) {
        // Error objects
        cudaError_t err;
        NppStatus errCode;

        // Calculate the number of chunks for the Connected Components algorithm
        Npp32u numberOfChunks = 1;
        Npp32u chunksPerDim;
        Npp32f predictedMemoryUsage =
                1.1f * float(connectedComponentsImage.total()) * float(connectedComponentsImage.elemSize());
        if (predictedMemoryUsage > double(PLImg::cuda::getFreeMemory())) {
            numberOfChunks = fmax(1, pow(4, ceil(log(predictedMemoryUsage / double(PLImg::cuda::getFreeMemory())) /
                                                 log(4))));
        }
        chunksPerDim = fmax(1, numberOfChunks / sqrt(numberOfChunks));

        // Setup histograms and bins for nppi execution
        std::vector<Npp32s> localHist = std::vector<Npp32s>(numLabels, 0);
        std::vector<Npp32s> globalHist = std::vector<Npp32s>(numLabels, 0);
        std::vector<Npp32f> bins = std::vector<float>(numLabels + 1, 1);

        // Fill bin values with (0, 1, 2, ...)
        std::iota(bins.begin(), bins.end(), 0);

        Npp32s *histBuffer;
        Npp32f *binBuffer;
        Npp32s xMin, xMax, yMin, yMax;
        cv::Mat subImage, croppedImage;
        Npp32f *deviceImage;
        Npp8u *deviceBuffer;
        NppiSize deviceROI;
        Npp32s nSrcStep, bufferSize;

        // Allocate memory for the bins
        err = cudaMalloc((void **) &binBuffer, bins.size() * sizeof(Npp32f));
        if (err != cudaSuccess) {
            std::cerr << "Could not allocate enough memory for bins of histogram \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        // Copy bins from CPU to GPU
        err = cudaMemcpy(binBuffer, bins.data(), bins.size() * sizeof(Npp32f), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Could not copy bins from host to device \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        // Allocate memory for the histogram
        err = cudaMalloc((void **) &histBuffer, localHist.size() * sizeof(Npp32s));
        if (err != cudaSuccess) {
            std::cerr << "Could not allocate enough memory bins of histogram \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        for (Npp32u it = 0; it < numberOfChunks; ++it) {
            // Calculate image boarders
            xMin = (it % chunksPerDim) * connectedComponentsImage.cols / chunksPerDim;
            xMax = fmin((it % chunksPerDim + 1) * connectedComponentsImage.cols / chunksPerDim,
                        connectedComponentsImage.cols);
            yMin = (it / chunksPerDim) * connectedComponentsImage.rows / chunksPerDim;
            yMax = fmin((it / chunksPerDim + 1) * connectedComponentsImage.rows / chunksPerDim,
                        connectedComponentsImage.rows);

            croppedImage = cv::Mat(connectedComponentsImage, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
            croppedImage.copyTo(subImage);
            croppedImage.release();
            subImage.convertTo(subImage, CV_32FC1);

            // Reserve memory on GPU for image and result image
            // Image
            err = cudaMalloc((void **) &deviceImage, subImage.total() * subImage.elemSize());
            if (err != cudaSuccess) {
                std::cerr << "Could not allocate enough memory for original image \n";
                std::cerr << cudaGetErrorName(err) << std::endl;
                exit(EXIT_FAILURE);
            }

            // Copy image from CPU to GPU
            err = cudaMemcpy(deviceImage, subImage.data, subImage.total() * subImage.elemSize(),
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                std::cerr << "Could not copy image from host to device \n";
                std::cerr << cudaGetErrorName(err) << std::endl;
                exit(EXIT_FAILURE);
            }

            // Length of columns
            nSrcStep = sizeof(Npp32f) * subImage.cols;
            deviceROI = {subImage.cols, subImage.rows};

            // Get buffer size for the histogram calculation
            nppiHistogramRangeGetBufferSize_32f_C1R(deviceROI, numLabels + 1, &bufferSize);
            err = cudaMalloc(&deviceBuffer, bufferSize);
            if (err != cudaSuccess) {
                std::cerr << "Could not allocate enough memory for buffer \n";
                std::cerr << cudaGetErrorName(err) << std::endl;
                exit(EXIT_FAILURE);
            }

            // Calculate the histogram based on our input image and bins.
            // The largest histogram bin will be the largest component in our image.
            errCode = nppiHistogramRange_32f_C1R(deviceImage, nSrcStep, deviceROI, histBuffer, binBuffer, numLabels + 1,
                                                 deviceBuffer);
            if (errCode != NPP_SUCCESS) {
                printf("NPP error: Could not calculate histogram : %d\n", errCode);
                exit(EXIT_FAILURE);
            }

            // Copy image from CPU to GPU
            err = cudaMemcpy(localHist.data(), histBuffer, localHist.size() * sizeof(Npp32s), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                std::cerr << "Could not copy image from device to host \n";
                std::cerr << cudaGetErrorName(err) << std::endl;
                exit(EXIT_FAILURE);
            }

            // If our image was chunked we still need to create our full histogram based on our small histograms.
            // The needed values are added here.
            #pragma omp parallel for default(shared)
            for (uint i = 0; i < globalHist.size(); ++i) {
                globalHist.at(i) += localHist.at(i);
            }
            cudaFree(deviceImage);
            cudaFree(deviceBuffer);
        }
        cudaFree(binBuffer);
        cudaFree(histBuffer);

        int maxLabel;
        // Get number of threads for next step
        uint numThreads;
        #pragma omp parallel
        numThreads = omp_get_num_threads();
        // Create vector of maxima to get the maximum of maxima
        std::vector<std::pair<int, int>> threadMaxLabels(numThreads);
        #pragma omp parallel private(maxLabel)
        {
            uint myThread = omp_get_thread_num();
            uint numElements = globalHist.end() - globalHist.begin() - 1;
            uint myStart = numElements / numThreads * myThread;
            uint myEnd = fmin(numElements, numElements / numThreads * (myThread + 1));
            maxLabel = std::distance(globalHist.begin(), std::max_element(globalHist.begin() + 1 + myStart, globalHist.begin() + 1 + myEnd));
            std::pair<int, int> myMaxLabel = std::pair<int, int>(maxLabel, globalHist.at(maxLabel));
            threadMaxLabels.at(myThread) = myMaxLabel;
        }

        maxLabel = 0;
        for(uint i = 0; i < numThreads; ++i) {
            if(threadMaxLabels.at(i).second >= threadMaxLabels.at(maxLabel).second) {
                maxLabel = i;
            }
        }
        maxLabel = threadMaxLabels.at(maxLabel).first;
        return std::pair<cv::Mat, int>(connectedComponentsImage == maxLabel, globalHist.at(maxLabel));
    } else if(numLabels == 1){
        return std::pair<cv::Mat, int>(connectedComponentsImage == 1, cv::countNonZero(connectedComponentsImage));
    } else {
        return std::pair<cv::Mat, int>(cv::Mat(), 0);
    }

}
