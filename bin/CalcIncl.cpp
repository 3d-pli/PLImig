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

#include "reader.h"
#include "writer.h"
#include "inclination.h"
#include "CLI/CLI.hpp"
#include <opencv2/core.hpp>

#include <vector>
#include <string>
#include <iostream>

int main(int argc, char** argv) {
    CLI::App app;

    // Get the number of threads for all following steps
    int numThreads;
    #pragma omp parallel
    numThreads = omp_get_num_threads();
    cv::setNumThreads(numThreads);

    std::vector<std::string> transmittance_files;
    std::vector<std::string> retardation_files;
    std::vector<std::string> mask_files;
    std::string output_folder;
    std::string dataset;
    float im, ic, rmaxWhite, rmaxGray, transmittance_normalization;
    bool detailed = false;

    auto required = app.add_option_group("Required parameters");
    required->add_option("--itra", transmittance_files, "Input transmittance files")
            ->required()
            ->check(CLI::ExistingFile);
    required->add_option("--iret", retardation_files, "Input retardation files")
            ->required()
            ->check(CLI::ExistingFile);
    required->add_option("--imask", mask_files, "Input mask files from PLImg")
            ->required()
            ->check(CLI::ExistingFile);
    required->add_option("-o, --output", output_folder, "Output folder")
            ->required()
            ->check(CLI::ExistingDirectory);

    auto optional = app.add_option_group("Optional parameters");
    optional->add_option("-d, --dataset", dataset, "HDF5 dataset")
            ->default_val("/Image");
    optional->add_flag("--detailed", detailed);
    optional->add_option("--im, --tm", im)->default_val(-1);
    optional->add_option("--ic, --tc", ic)->default_val(-1);
    optional->add_option("--rmaxWhite, --rrefhm", rmaxWhite)->default_val(-1);
    optional->add_option("--rmaxGray, --rreflm", rmaxGray)->default_val(-1);
    optional->add_option("--transmittance_normalization", transmittance_normalization, "Normalization value to convert Transmittance to NTransmittance. Has to be provided if not present in the HDF5 file.")
              ->default_val(-1);
    CLI11_PARSE(app, argc, argv);

    PLImg::HDF5Writer writer;
    PLImg::Inclination inclination;
    std::string transmittance_basename, inclination_basename;
    std::string transmittance_path, retardation_path, mask_path;

    for(unsigned i = 0; i < transmittance_files.size(); ++i) {
        transmittance_path = transmittance_files.at(i);
        retardation_path = retardation_files.at(i);
        mask_path = mask_files.at(i);
        std::cout << transmittance_path << std::endl;
        std::cout << retardation_path << std::endl;
        std::cout << mask_path << std::endl;

        #ifdef WIN32
                unsigned long long int endPosition = transmittance_path.find_last_of('\\');
        #else
                unsigned long long int endPosition = transmittance_path.find_last_of('/');
        #endif
        if(endPosition != std::string::npos) {
            transmittance_basename = transmittance_path.substr(endPosition+1);
        } else {
            transmittance_basename = transmittance_path;
        }
        for(std::string extension : std::array<std::string, 5> {".h5", ".tiff", ".tif", ".nii.gz", ".nii"}) {
            endPosition = transmittance_basename.rfind(extension);
            if(endPosition != std::string::npos) {
                transmittance_basename = transmittance_basename.substr(0, endPosition);
            }
        }

        // Get name of retardation and check if transmittance has median filer applied.
        inclination_basename = std::string(transmittance_basename);
        auto pos = inclination_basename.find("median");
        if (pos != std::string::npos) {
            int length = 6;
            while(std::isdigit(inclination_basename.at(pos + length))) {
                ++length;
            }
            inclination_basename = inclination_basename.replace(pos, length, "");
        }
        if (inclination_basename.find("NTransmittance") != std::string::npos) {
            inclination_basename = inclination_basename.replace(inclination_basename.find("NTransmittance"), 14, "Inclination");
        }
        if (inclination_basename.find("Transmittance") != std::string::npos) {
            inclination_basename = inclination_basename.replace(inclination_basename.find("Transmittance"), 13, "Inclination");
        }

        // Read all files.
        std::shared_ptr<cv::Mat> transmittance = std::make_shared<cv::Mat>(PLImg::Reader::imread(transmittance_path, dataset));
        std::shared_ptr<cv::Mat> retardation = std::make_shared<cv::Mat>(PLImg::Reader::imread(retardation_path, dataset));
        std::shared_ptr<cv::Mat> mask = std::make_shared<cv::Mat>(PLImg::Reader::imread(mask_path, dataset));
        std::shared_ptr<cv::Mat> blurredMask = std::make_shared<cv::Mat>(PLImg::Reader::imread(mask_path, "/Probability"));
        std::cout << "Files read" << std::endl;

        // Get normalization value for transmittance
        if(transmittance_normalization == -1) {
            std::string attribute_value = PLImg::Reader::attribute(transmittance_path, "normalization_value");
            if(attribute_value == "") {
                std::cerr << "Attribute normalization_value was not found and no normalization value was given through the command line!" << std::endl;
                return 1;
            }
            transmittance_normalization = std::stof(attribute_value);
        }
        *transmittance = *transmittance / transmittance_normalization;

        std::shared_ptr<cv::Mat> medTransmittance;
        // If our given transmittance isn't already median filtered (based on it's file name)
        if (transmittance_path.find("median") == std::string::npos) {
            // Generate med10Transmittance
            medTransmittance = PLImg::cuda::filters::medianFilterMasked(transmittance, mask);
            std::cout << "Filtered transmittance generated" << std::endl;
        } else {
            medTransmittance = transmittance;
        }
        transmittance = nullptr;

        // Set our read parameters
        inclination.setModalities(medTransmittance, retardation, blurredMask, mask);
        // If manual parameters were given, apply them here
        if(im >= 0) {
            inclination.set_TM(im);
        }
        if(ic >= 0) {
            inclination.set_Tc(ic);
        }
        if(rmaxWhite >= 0) {
            inclination.set_RrefHM(rmaxWhite);
        }
        if(rmaxGray >= 0) {
            inclination.set_RrefLM(rmaxGray);
        }
        // Create file and dataset. Write the inclination afterwards.
        writer.set_path(output_folder+ "/" + inclination_basename + ".h5");
        writer.write_dataset("/Image", *inclination.inclination());
        writer.write_attribute("/Image", "im", inclination.T_c());
        writer.write_attribute("/Image", "ic", inclination.T_M());
        writer.write_attribute("/Image", "rmax_white", inclination.R_refHM());
        writer.write_attribute("/Image", "rmax_gray", inclination.R_refLM());
        // writer.write_attribute("/Image", "version", PLImg::Version::versionHash() + ", " + PLImg::Version::timeStamp());

        writer.writePLIMAttributes({transmittance_path, retardation_path, mask_path}, "/Image", "/Image", "Inclination", argc, argv);
        std::cout << "Inclination generated and written" << std::endl;
        writer.close();

        if(detailed) {
            auto saturation_basename = std::string(inclination_basename);
            if (inclination_basename.find("Inclination") != std::string::npos) {
                saturation_basename = saturation_basename.replace(saturation_basename.find("Inclination"), 11, "Saturation");
            }
            // Create file and dataset. Write the inclination afterwards.
            writer.set_path(output_folder+ "/" + saturation_basename + ".h5");

            writer.write_dataset("/Image", *inclination.saturation());
            writer.write_attribute("/Image", "im", inclination.T_c());
            writer.write_attribute("/Image", "ic", inclination.T_M());
            writer.write_attribute("/Image", "rmax_white", inclination.R_refHM());
            writer.write_attribute("/Image", "rmax_gray", inclination.R_refLM());
            // writer.write_attribute("/Image", "version", PLImg::Version::versionHash() + ", " + PLImg::Version::timeStamp());

            writer.writePLIMAttributes({transmittance_path, retardation_path, mask_path}, "/Image", "/Image", "Inclination Saturation", argc, argv);
            std::cout << "Saturation image generated and written" << std::endl;
            writer.close();
        }

        std::cout << std::endl;
    }

    return EXIT_SUCCESS;
}
