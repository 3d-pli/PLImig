/*
    MIT License

    Copyright (c) 2020 Forschungszentrum Jülich / Jan André Reuter.

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

#ifndef PLIMG_INCLINATION_H
#define PLIMG_INCLINATION_H

#include <cmath>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "toolbox.h"

typedef std::shared_ptr<cv::Mat> sharedMat;

namespace PLImg {
    class Inclination {
    public:
        Inclination();
        Inclination(sharedMat transmittance, sharedMat retardation, sharedMat blurredMask,
                    sharedMat whiteMask, sharedMat grayMask);
        void setModalities(sharedMat transmittance, sharedMat retardation, sharedMat blurredMask, sharedMat whiteMask,
                           sharedMat grayMask);

        float im();
        float ic();
        float rmaxGray();
        float rmaxWhite();

        void set_im(float im);
        void set_ic(float ic);
        void set_rmaxGray(float rmaxGray);
        void set_rmaxWhite(float rmaxWhite);

        sharedMat inclination();
        sharedMat saturation();
    private:
        std::unique_ptr<float> m_im, m_ic, m_rmaxGray, m_rmaxWhite;
        std::unique_ptr<cv::Mat> m_regionGrowingMask;
        sharedMat m_transmittance, m_retardation, m_inclination, m_saturation;
        sharedMat m_blurredMask, m_whiteMask, m_grayMask;

    };
}


#endif //PLIMG_INCLINATION_H
