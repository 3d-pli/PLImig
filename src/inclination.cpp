//
// Created by jreuter on 03.12.20.
//

#include "inclination.h"
#include <iostream>

PLImg::Inclination::Inclination() : m_transmittance(), m_retardation(), m_blurredMask(), m_whiteMask(), m_grayMask() {
    m_im = nullptr,
    m_ic = nullptr;
    m_rmaxWhite = nullptr;
    m_rmaxGray = nullptr;
    m_regionGrowingMask = nullptr;
}

PLImg::Inclination::Inclination(sharedMat transmittance, sharedMat retardation,
                                sharedMat blurredMask, sharedMat whiteMask, sharedMat grayMask) :
                                m_transmittance(std::move(transmittance)), m_retardation(std::move(retardation)), m_blurredMask(std::move(blurredMask)),
                                m_whiteMask(std::move(whiteMask)), m_grayMask(std::move(grayMask)) {
    m_im = nullptr,
    m_ic = nullptr;
    m_rmaxWhite = nullptr;
    m_rmaxGray = nullptr;
    m_regionGrowingMask = nullptr;
    m_inclination = nullptr;
}

void PLImg::Inclination::setModalities(sharedMat transmittance, sharedMat retardation,
                                       sharedMat blurredMask, sharedMat whiteMask, sharedMat grayMask) {
    m_transmittance = std::move(transmittance);
    m_retardation = std::move(retardation);
    m_blurredMask = std::move(blurredMask);
    m_whiteMask = std::move(whiteMask);
    m_grayMask = std::move(grayMask);

    m_im = nullptr,
    m_ic = nullptr;
    m_rmaxWhite = nullptr;
    m_rmaxGray = nullptr;
    m_regionGrowingMask = nullptr;
    m_inclination = nullptr;
}

void PLImg::Inclination::set_ic(float ic) {
    m_ic = std::make_unique<float>(ic);
    m_inclination = nullptr;
}

void PLImg::Inclination::set_im(float im) {
    m_im = std::make_unique<float>(im);
    m_inclination = nullptr;
}

void PLImg::Inclination::set_rmaxGray(float rmaxGray) {
    m_rmaxGray = std::make_unique<float>(rmaxGray);
    m_inclination = nullptr;
}

void PLImg::Inclination::set_rmaxWhite(float rmaxWhite) {
    m_rmaxWhite = std::make_unique<float>(rmaxWhite);
    m_inclination = nullptr;
}

float PLImg::Inclination::ic() {
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    if(!m_ic) {
        cv::Mat selection = *m_grayMask & *m_blurredMask < 0.01;

        int channels[] = {0};
        float histBounds[] = {0.0f, 1.0f};
        const float* histRange = { histBounds };
        int histSize = 1000;

        cv::Mat hist;
        cv::calcHist(&(*m_transmittance), 1, channels, selection, hist, 1, &histSize, &histRange, true, false);

        int max_pos = std::max_element(hist.begin<float>(), hist.end<float>()) - hist.begin<float>();
        m_ic = std::make_unique<float>(float(max_pos) / float(histSize));
    }
    return *m_ic;
}

float PLImg::Inclination::im() {
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    if(!m_im) {
        if(!m_regionGrowingMask) {
            m_regionGrowingMask = std::make_unique<cv::Mat>(PLImg::imageRegionGrowing(*m_retardation));
        }
        m_im = std::make_unique<float>(cv::mean(*m_transmittance, *m_regionGrowingMask)[0]);
    }
    return *m_im;
}

float PLImg::Inclination::rmaxGray() {
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    if(!m_rmaxGray) {
        int channels[] = {0};
        float histBounds[] = {0.0f, 1.0f};
        const float* histRange = { histBounds };
        int histSize = NUMBER_OF_BINS;

        // Generate histogram
        cv::Mat hist;
        cv::calcHist(&(*m_retardation), 1, channels, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

        // Create kernel for convolution of histogram
        int kernelSize = histSize/20;
        cv::Mat kernel(kernelSize, 1, CV_32FC1);
        kernel.setTo(cv::Scalar(1.0f/float(kernelSize)));
        cv::filter2D(hist, hist, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, CV_32F);

        // TODO: Peaksuche

        m_rmaxGray = std::make_unique<float>(histogramPlateau(hist, -kernelSize / (2.0f * NUMBER_OF_BINS),
                                                                1.0f - kernelSize / (2.0f * NUMBER_OF_BINS),
                                                                1, 0, NUMBER_OF_BINS/2));
    }
    return *m_rmaxGray;
}

float PLImg::Inclination::rmaxWhite() {
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    if(!m_rmaxWhite) {
        if(!m_regionGrowingMask) {
            m_regionGrowingMask = std::make_unique<cv::Mat>(PLImg::imageRegionGrowing(*m_retardation));
        }
        m_rmaxWhite = std::make_unique<float>(cv::mean(*m_retardation, *m_regionGrowingMask)[0]);
    }
    return *m_rmaxWhite;
}

sharedMat PLImg::Inclination::inclination() {
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    if(!m_inclination) {
        std::cout << __PRETTY_FUNCTION__ << std::endl;
        m_inclination = std::make_shared<cv::Mat>(m_retardation->rows, m_retardation->cols, CV_32FC1);
        float tmpVal;
        float blurredMaskVal;
        float asinWRmax = asin(rmaxWhite());
        float asinGRMax = asin(rmaxGray());
        float logIcIm = log(ic() / im());
        #pragma omp parallel for default(shared) private(tmpVal, blurredMaskVal)
        for(unsigned y = 0; y < m_inclination->rows; ++y) {
            for(unsigned x = 0; x < m_inclination->cols; ++x) {
                blurredMaskVal = m_blurredMask->at<float>(y, x);
                if(blurredMaskVal < 0.01) {
                    blurredMaskVal = 0;
                }
                tmpVal = sqrt(
                        blurredMaskVal *
                                (
                                        asin(m_retardation->at<float>(y, x)) /
                                        asinWRmax *
                                        logIcIm /
                                        fmax(1e-15, log(*m_ic / m_transmittance->at<float>(y, x)))
                                )
                            + (1 - blurredMaskVal) *
                            asin(m_retardation->at<float>(y, x)) /
                            asinGRMax
                        );
                if(tmpVal > 1) {
                    tmpVal = 1;
                } else if(tmpVal < -1) {
                    tmpVal = -1;
                }
                m_inclination->at<float>(y, x) = acos(tmpVal) * 180 / M_PI;
            }
        }
    }
    return m_inclination;
}