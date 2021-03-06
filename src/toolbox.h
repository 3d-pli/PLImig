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
#ifndef PLIMG_TOOLBOX_H
#define PLIMG_TOOLBOX_H

#include "cuda/cuda_toolbox.h"
#include "cuda/define.h"
#include "cuda/exceptions.h"
#include <chrono>
#include <numeric>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <random>
#include <set>
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#include <vector>

/// Minimum number of bins used for the calculation of tRet() and tTra()
constexpr auto MIN_NUMBER_OF_BINS = 64;
/// Maximum number of bins used for the calculation of tRet() and tTra(). This value will also be used as the default number of bins for all other operations.
constexpr auto MAX_NUMBER_OF_BINS = 256;

/**
 * @file
 * @brief PLImg histogram toolbox functions
 */
namespace PLImg {
    namespace Histogram {
        /**
         * @brief Determine the one-sided peak width of a given peak position based on its height.
         * @param hist Histogram which was calculated using OpenCV functions
         * @param peakPosition Peak position in histogram of which the width shall be calculated
         * @param direction +1 -> ++peakPosition , -1 -> --peakPosition
         * @param targetHeight Target height for the peak width. 0.5 = 50%.
         * @return Integer number describing the width in bins
         */
        int peakWidth(cv::Mat hist, int peakPosition, float direction, float targetHeight = 0.5f);
        
        /**
         * This method calculates the floating point value of the curvature in a given histogram.
         * To achieve this, the curvature is determined using numerical differentiation with the following formula
         * \f[ \kappa = \frac{y^{''}}{(1+(y^{'})^2)^{3/2}} \f]
         * The maximum position of \f$ \kappa \f$ is our maximum curvature and will be returned.
         * @brief Calculate the maximum curvature floating point value of a histogram
         * @param hist OpenCV calculated histogram
         */
         cv::Mat curvature(cv::Mat hist, float histHigh, float histLow);

        /**
         * Histograms can be interpreted as a discrete function with extrema like any normal function. This method
         * allows to calculate the peak positions in between an interval (start, stop) with an additional threshold
         * which is the prominence of detected peaks.
         * The implementation is similar to the implementation of peak finding in SciPy.
         * @brief Calculate the peak positions within a histogram filtered with prominence values
         * @param hist OpenCV histogram
         * @param start Start bin
         * @param stop End bin
         * @param minSignificance Minimal prominence value for a peak to be considered as such.
         * @return Vector with the peak positions in between start and stop
         */
        std::vector<unsigned> peaks(cv::Mat hist, int start, int stop, float minSignificance = 0.01f);
    }

    namespace Image {
        std::array<cv::Mat, 2> randomizedModalities(std::shared_ptr<cv::Mat>& transmittance, std::shared_ptr<cv::Mat>& retardation, float scalingValue=0.25f);
        unsigned long long maskCountNonZero(const cv::Mat& mask);
    }

    namespace cuda {
        /**
         * @brief Execute some CUDA checks to ensure that the rest of the program should run as expected.
         * @return true if all checks were run successfully
         */
        bool runCUDAchecks();
        /**
         * @brief Get the total amount of memory in bytes.
         * @return Total amount of VRAM in bytes.
         */
        size_t getTotalMemory();
        /**
         * @brief Get the free amount of memory in bytes.
         * @return Total amount of free VRAM in bytes.
         */
        size_t getFreeMemory();

        size_t getHistogramMemoryEstimation(const cv::Mat& image, uint numBins);
        cv::Mat histogram(const cv::Mat& image, float minLabel, float maxLabel, uint numBins);

        namespace filters {
            size_t getMedianFilterMemoryEstimation(const std::shared_ptr<cv::Mat>& image);
            size_t getMedianFilterMaskedMemoryEstimation(const std::shared_ptr<cv::Mat>& image, const std::shared_ptr<cv::Mat>& mask);
            /**
             * This method applies a circular median filter with a radius of 10 to the given image.
             * @brief Apply circular median filter with a radius of 10 to the image
             * @param image Image on which the median filter will be applied.
             * @return Shared pointer of the filtered image.
             */
            std::shared_ptr<cv::Mat> medianFilter(const std::shared_ptr<cv::Mat>& image);
            /**
             * This method applies a circular median filter with a radius of 10 to the given image. In addition
             * only masked pixels will be filtered.
             * Let's take the following example for a small mask:
             * | 1 | 1 | 1 |
             * | 0 | 1 | 0 |
             * | 0 | 0 | 0 |
             * When we are at a pixel which is masked with a 1, we will only look at pixels within the radius which is
             * also a 1. When we are at a pixel which is masked with a 0, we will only look at pixels within the radius
             * which is also a 0.
             * @brief Apply circular median filter with a radius of 10 to the image while applying a separation mask.
             * @param image Image on which the masked median filter will be applied.
             * @param mask 8-bit mask for the median filter.
             * @return Shared pointer of the filtered image.
             */
            std::shared_ptr<cv::Mat> medianFilterMasked(const std::shared_ptr<cv::Mat>& image, const std::shared_ptr<cv::Mat>& mask);
        }

        namespace labeling {
            size_t getLargestAreaConnectedComponentsMemoryEstimation(const cv::Mat& image);
            size_t getConnectedComponentsMemoryEstimation(const cv::Mat& image);
            size_t getConnectedComponentsLargestComponentMemoryEstimation(const cv::Mat& image);

            /**
             * This method allows to search the largest connected component in an image. This connected component will
             * represent the largest area with the highest image values consisting of at least
             * \f$ percentPixels / 100 * image.size() \f$ pixels. The threshold is determined through the number of image bins
             * which need to be used to find a valid mask with enough pixels.
             * @brief Search for the largest connected components area which fills at least percentPixels of the image size.
             * @param image OpenCV image which will be used for the connected components algorithm.
             * @param mask
             * @param percentPixels Percent of pixels which are needed for the algorithm to succeed.
             * @return OpenCV matrix masking the connected components area with the largest pixels
             */
            cv::Mat largestAreaConnectedComponents(const cv::Mat& image, cv::Mat mask = cv::Mat(), float percentPixels = 0.01f);
            
            /**
             * Execute the connected components algorithm on an 8-bit image. This method will use the CUDA NPP library
             * to detect connected regions and will return a mask with the resulting labels. If the input image is too
             * large this method will use chunks to reduce the memory load. However, this will increase the computing
             * time significantly.
             * @brief Run connected components algorithm on an 8-bit image
             * @param image 8-bit OpenCV matrix
             * @return OpenCV matrix with the resulting labels of the input image
             */
            cv::Mat connectedComponents (const cv::Mat& image);
            /**
             * If the original image is too large for the connected component algorithm on the GPU the image will be
             * split into chunks to allow the execution. However, labels on the edges of the chunks might be
             * wrong because they are split due to the chunk choice. This method fixes the wrong labels by checking
             * border regions for labels on both sides and creating a lookup table to fix those labels.
             * The input image itself will be altered in this operation. Please keep this in mind.
             * @param image Chunked image which contains possible wrong labeling
             * @param numberOfChunks Number of chunks which were used to generate the image.
             */
            void connectedComponentsMergeChunks(cv::Mat& image, int numberOfChunks);
            /**
             * connectedComponents (const cv::Mat& image) will return a labeled image which can be further analyzed.
             * This functions allows to find the largest region and will return a mask of it in combination with its
             * size as an integer value. This method will use the CUDA NPP library to create a histogram of the labels
             * @brief Get mask and size of the largest component from connected components mask
             * @param connetedComponentsImage Output image of connectedComponents (const cv::Mat& image)
             * @return Pair of the largest region mask and the number of pixels in the mask.
             */
            std::pair<cv::Mat, int> largestComponent(const cv::Mat& connectedComponentsImage);
        }
    }
}

#endif //PLIMG_TOOLBOX_H
