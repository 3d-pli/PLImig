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

#ifndef PLIMG_WRITER_H
#define PLIMG_WRITER_H

#include <filesystem>
#include <H5Cpp.h>
#include <opencv2/core.hpp>
#include <string>
#ifdef __GNUC__
    #include <unistd.h>
    #include <pwd.h>
#else
    #define NOMINMAX
    #include <Windows.h>
    #include <lmcons.h>
#endif

#include "plim/AttributeHandler.h"
#include "reader.h"
#include "version.h"

constexpr hsize_t hdf5_writer_chunk_dimensions[2] = {2048, 2048};

/**
 * @file
 * @brief PLImg::HDF5Writer class
 */
namespace PLImg {
    /**
     * The HDF5Writer class is the main class to write results from PLImig to files while preserving all information
     * like generated parameters or modality information. PLIM is used to extract the information from the original
     * transmittance and retardation. This class offers functionality to write numerical datasets and attributes
     * (float, double, int + String for attributes).
     * @brief The HDF5Writer class
     */
    class HDF5Writer {
    public:
        /**
         * @brief HDF5Writer Default constructor for the HDF5Writer
         * Default constructor for the HDF5Writer. This class will be used to write HDF5 files.
         */
        HDF5Writer();
        /**
         * @brief path Get currently set path
         * @return Currently set path via the set_path(const std::string& filename) method
         */
        std::string path();
        /**
         * Set the desired path of the HDF5 file which will be written. The file will be created if it doesn't exist.
         * If the file already exists, open it in append mode.
         * @brief set_path Set HDF5 path. If path exists, open the file
         * @param filename Path of the file which will be written
         */
        void set_path(const std::string& filename);
        /**
         * HDF5 files support attributes in addition to storing raw data. This methods allows to set attributes for
         * a given dataset. Here, a float attribute will be written to the parameter_name in the dataset.
         * @param dataset Existing dataset which the attribute will be written to
         * @param parameter_name Parameter name for the attribute in the dataset
         * @param value Value that will be written to the dataset
         */
        void write_attribute(const std::string& dataset, const std::string& parameter_name, float value);
        /**
         * HDF5 files support attributes in addition to storing raw data. This methods allows to set attributes for
         * a given dataset. Here, a double attribute will be written to the parameter_name in the dataset.
         * @param dataset Existing dataset which the attribute will be written to
         * @param parameter_name Parameter name for the attribute in the dataset
         * @param value Value that will be written to the dataset
         */
        void write_attribute(const std::string& dataset, const std::string& parameter_name, double value);
        /**
         * HDF5 files support attributes in addition to storing raw data. This methods allows to set attributes for
         * a given dataset. Here, an int attribute will be written to the parameter_name in the dataset.
         * @param dataset Existing dataset which the attribute will be written to
         * @param parameter_name Parameter name for the attribute in the dataset
         * @param value Value that will be written to the dataset
         */
        void write_attribute(const std::string& dataset, const std::string& parameter_name, int value);
        /**
         * HDF5 files support attributes in addition to storing raw data. This methods allows to set attributes for
         * a given dataset. Here, a string attribute will be written to the parameter_name in the dataset.
         * @param dataset Existing dataset which the attribute will be written to
         * @param parameter_name Parameter name for the attribute in the dataset
         * @param value Value that will be written to the dataset
         */
        void write_attribute(const std::string& dataset, const std::string& parameter_name, std::string value);
        /**
         * This method allows to write an OpenCV matrix to a HDF5 file. The image will be written to a given dataset.
         * If the dataset doesn't exist yet the method will create the dataset and write the data.
         * However, if there's already a dataset with the same name this method will check if the datatype and image
         * dimensions match. If that's the case, the data in the HDF5 file will be overwritten. Otherwise, an exception
         * is thrown.
         * @brief Write OpenCV image to a dataset in the HDF5 file
         * @param dataset Destination within the HDF5 file.
         * @param image OpenCV image which will be written.
         */
        void write_dataset(const std::string& dataset, const cv::Mat& image, bool create_softlink=false);
        /**
         * This method allows the recursive creation of groups within a HDF5 file.
         * @brief Create group within HDF5 file
         * @param group Group which shall be created.
         */
        void create_group(const std::string& group);
        /**
         * @brief close Closes currently opened file.
         */
        void close();
        /**
         *
         * @param transmittance_path Original transmittance path used for the call of the program
         * @param retardation_path Original retardation path used for the call of the program
         * @param output_dataset Dataset which will be written to
         * @param input_dataset Dataset which will be used to copy most attributes
         * @param modality Name of the modality which will be written (NTransmittance, Retardation, ...)
         * @param argc Number of arguments when calling the program
         * @param argv Arguments when calling the program
         */
        void writePLIMAttributes(const std::vector<std::string>& reference_maps,
                                 const std::string& output_dataset, const std::string& input_dataset,
                                 const std::string& modality, int argc, char** argv);
    private:
        /**
         * @brief Opens current HDF5 path
         */
        void open();

        static void createDirectoriesIfMissing(const std::string& filename);

        void write_type_attribute(const std::string& dataset, const std::string& parameter_name, const H5::AtomType& datatype, void* value);

        static void writePLIMReference(plim::AttributeHandler& handler, std::initializer_list<plim::AttributeHandler> reference_handler);

        ///
        std::string m_filename;
        ///
        H5::H5File m_hdf5file;
    };
}
#endif //PLIMG_WRITER_H
