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

#ifndef PLIMG_READER_H
#define PLIMG_READER_H

#include <filesystem>
#include <vector>
#include <hdf5.h>
#include <nifti/nifti1_io.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sys/stat.h>

#include <PLIHDF5/plihdf5.h>
#include <PLIHDF5/exceptions.h>

/**
 * @file
 * @brief PLImg::Reader class
 */
namespace PLImg {
    class Reader {
    public:
        /**
         * Checks if the given file exists
         * @param filename Path to the file to check
         * @return True if file exists, otherwise false
         */
        static bool fileExists(const std::string& filename);
        /**
         * Opens and reads an image with file ending .h5, .nii or .tiff
         * @param filename Path to the file which shall be opened.
         * @param dataset HDF5 dataset from which the image shall be read.
         * @return OpenCV Matrix containing the image.
         */
        static cv::Mat imread(const std::string& filename, const std::string& dataset="/Image");
        /**
         * Returns a list of all readable datasets within the given HDF5 file
         * @param filename Path to the file which shall be opened.
         * @return Vector of readable datasets
         */
        static std::vector<std::string> datasets(const std::string& filename);

        static std::string attribute(const std::string& filename, const std::string attributeName);
    private:
        /**
         * Opens and reads an image with file ending .h5
         * @param filename Path to the file which shall be opened.
         * @param dataset HDF5 dataset from which the image shall be read.
         * @return OpenCV Matrix containing the image.
         */
        static cv::Mat readHDF5(const std::string& filename, const std::string& dataset="/Image");
        /**
         * Opens and reads an image with file ending .tiff
         * @param filename Path to the file which shall be opened.
         * @return OpenCV Matrix containing the image.
         */
        static cv::Mat readTiff(const std::string& filename);
        /**
         * Opens and reads an image with file ending .nii
         * @param filename Path to the file which shall be opened.
         * @return OpenCV Matrix containing the image.
         */
        static cv::Mat readNIFTI(const std::string& filename);

        static std::vector<std::string> datasets(hid_t group_id);
    };

}


#endif //PLIMG_READER_H
