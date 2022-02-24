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
#include "maskgeneration.h"
#include "inclination.h"
#include "CLI/CLI.hpp"

#include <vector>
#include <string>
#include <iostream>

#ifdef TIME_MEASUREMENT
    #pragma message("Time measurement enabled.")
    #include <chrono>
#endif

int main(int argc, char** argv) {
    #ifdef TIME_MEASUREMENT
        auto start = std::chrono::high_resolution_clock::now();
    #endif
    CLI::App app;

    // Get the number of threads for all following steps
    int numThreads;
    #pragma omp parallel
    numThreads = omp_get_num_threads();
    cv::setNumThreads(numThreads);

    std::vector<std::string> transmittance_files;
    std::vector<std::string> retardation_files;
    std::string output_folder;
    std::string dataset;
    bool detailed = false;

    float tmin, tmax, tret, ttra, transmittance_normalization;

    auto required = app.add_option_group("Required parameters");
    required->add_option("--itra", transmittance_files, "Input transmittance files")
            ->required()
            ->check(CLI::ExistingFile);
    required->add_option("--iret", retardation_files, "Input retardation files")
            ->required()
            ->check(CLI::ExistingFile);
    required->add_option("-o, --output", output_folder, "Output folder")
                    ->required()
                    ->check(CLI::ExistingDirectory);

    auto optional = app.add_option_group("Optional parameters");
    optional->add_option("-d, --dataset", dataset, "HDF5 dataset")
                    ->default_val("/Image");
    optional->add_flag("--detailed", detailed);
    auto parameters = optional->add_option_group("Parameters", "Control the generated masks by setting parameters manually");
    parameters->add_option("--ilower, --tthres", ttra, "Average transmittance value of brightest retardation values")
              ->default_val(-1);
    parameters->add_option("--rthres", tret, "Plateau in retardation histogram")
              ->default_val(-1);
    parameters->add_option("--irmax, --tref", tmin, "Average transmittance value of brightest retardation values")
              ->default_val(-1);
    parameters->add_option("--iupper, --tback", tmax, "Separator of gray matter and background")
              ->default_val(-1);
    parameters->add_option("--transmittance_normalization", transmittance_normalization, "Normalization value to convert Transmittance to NTransmittance. Has to be provided if not present in the HDF5 file.")
              ->default_val(-1);
    CLI11_PARSE(app, argc, argv);

    PLImg::HDF5Writer writer;
    PLImg::MaskGeneration generation;
    PLImg::Inclination inclination;

    std::string transmittance_basename, mask_basename, inclination_basename;
    std::string retardation_path, mask_path, transmittance_path, median_transmittance_path;

    for(unsigned i = 0; i < transmittance_files.size(); ++i) {
        transmittance_path = transmittance_files.at(i);
        retardation_path = retardation_files.at(i);
        std::cout << transmittance_path << std::endl;
        std::cout << retardation_path << std::endl;

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
        for(const std::string& extension : std::array<std::string, 5> {".h5", ".tiff", ".tif", ".nii.gz", ".nii"}) {
            endPosition = transmittance_basename.rfind(extension);
            if(endPosition != std::string::npos) {
                transmittance_basename = transmittance_basename.substr(0, endPosition);
            }
        }

        // Get name of retardation and check if transmittance has median filer applied.
        mask_basename = std::string(transmittance_basename);
        auto pos = mask_basename.find("median");
        if (pos != std::string::npos) {
            int length = 6;
            while(std::isdigit(mask_basename.at(pos + length))) {
                ++length;
            }
            mask_basename = mask_basename.replace(pos, length, "");
        }
        if (mask_basename.find("NTransmittance") != std::string::npos) {
            mask_basename = mask_basename.replace(mask_basename.find("NTransmittance"), 14, "Mask");
        }
        if (mask_basename.find("Transmittance") != std::string::npos) {
            mask_basename = mask_basename.replace(mask_basename.find("Transmittance"), 13, "Mask");
        }
        inclination_basename = std::string(mask_basename);
        if (mask_basename.find("Mask") != std::string::npos) {
            inclination_basename = inclination_basename.replace(inclination_basename.find("Mask"), 4, "Inclination");
        }

        std::shared_ptr<cv::Mat> transmittance = std::make_shared<cv::Mat>(
                PLImg::Reader::imread(transmittance_path, dataset));
        std::shared_ptr<cv::Mat> retardation = std::make_shared<cv::Mat>(
                PLImg::Reader::imread(retardation_path, dataset));


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

        std::cout << "Files read" << std::endl;

        std::shared_ptr<cv::Mat> medTransmittance;
        if (transmittance_path.find("median") == std::string::npos) {
            // Generate median transmittance
            medTransmittance = PLImg::cuda::filters::medianFilter(transmittance);
            // Set output file name
            std::string medianName = "median"+std::to_string(MEDIAN_KERNEL_SIZE)+"NTransmittance";
            std::string median_transmittance_basename(mask_basename);
            median_transmittance_basename.replace(mask_basename.find("Mask"), 4, medianName);
            median_transmittance_path = output_folder + "/" + median_transmittance_basename + ".h5";
            // Set and write file
            writer.set_path(median_transmittance_path);
            writer.write_dataset("/Image", *medTransmittance, true);
            writer.write_attribute("/Image", "median_kernel_size", int(MEDIAN_KERNEL_SIZE));
            writer.writePLIMAttributes({transmittance_path}, "/Image", "/Image", "NTransmittance", argc, argv);
            writer.close();
            std::cout << "Median-Transmittance generated" << std::endl;
        } else {
            medTransmittance = transmittance;
        }

        generation.setModalities(retardation, medTransmittance);
        if(ttra >= 0) {
            generation.set_tthres(ttra);
        }
        if(tret >= 0) {
            generation.set_rthres(tret);
        }
        if(tmin >= 0) {
            generation.set_tref(tmin);
        }
        if(tmax >= 0) {
            generation.set_tback(tmax);
        }
        generation.removeBackground();

        mask_path = output_folder + "/" + mask_basename + ".h5";
        writer.set_path(mask_path);

        writer.write_dataset("Image", *generation.fullMask(), true);
        writer.writePLIMAttributes({transmittance_path, retardation_path}, "/Image", "/Image", "Mask", argc, argv);
        writer.write_attribute("/Image", "i_lower", generation.T_thres());
        writer.write_attribute("/Image", "r_thres", generation.R_thres());
        writer.write_attribute("/Image", "i_rmax", generation.T_ref());
        writer.write_attribute("/Image", "i_upper", generation.T_back());
        // writer.write_attribute("/Image", "version", PLImg::Version::versionHash() + ", " + PLImg::Version::timeStamp());
        std::cout << "Mask generated and written" << std::endl;

        writer.write_dataset("/Probability", *generation.probabilityMask());
        std::cout << "Probability mask generated and written" << std::endl;

        if (detailed) {
            writer.write_dataset("/NoNerveFibers", *generation.noNerveFiberMask());
            std::cout << "Detailed masks generated and written" << std::endl;
        }
        writer.close();

        if (transmittance_path.find("median") == std::string::npos) {
            // Generate med10Transmittance
            medTransmittance = PLImg::cuda::filters::medianFilterMasked(transmittance, generation.fullMask());
            transmittance = nullptr;
        } else {
            medTransmittance = transmittance;
        }
        std::cout << "Median filtered and masked transmittance generated" << std::endl;

        // Set our read parameters
        inclination.setModalities(medTransmittance, retardation, generation.probabilityMask(), generation.fullMask());

        // Create file and dataset. Write the inclination afterwards.
        writer.set_path(output_folder+ "/" + inclination_basename + ".h5");
        writer.write_dataset("/Image", *inclination.inclination(), true);
        writer.write_attribute("/Image", "im", inclination.T_c());
        writer.write_attribute("/Image", "ic", inclination.T_M());
        writer.write_attribute("/Image", "rmax_white", inclination.R_refHM());
        writer.write_attribute("/Image", "rmax_gray", inclination.R_refLM());
        // writer.write_attribute("/Image", "version", PLImg::Version::versionHash() + ", " + PLImg::Version::timeStamp());

        writer.writePLIMAttributes(std::vector<std::string> {transmittance_path, retardation_path, mask_path}, "/Image", "/Image", "Inclination", argc, argv);
        std::cout << "Inclination generated and written" << std::endl;
        writer.close();

        if(detailed) {
            auto saturation_basename = std::string(mask_basename);
            if (mask_basename.find("Mask") != std::string::npos) {
                saturation_basename = saturation_basename.replace(saturation_basename.find("Mask"), 4, "Saturation");
            }
            // Create file and dataset. Write the inclination afterwards.
            writer.set_path(output_folder+ "/" + saturation_basename + ".h5");
            writer.write_dataset("/Image", *inclination.saturation(), true);
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

    #ifdef TIME_MEASUREMENT
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        std::cout << "Runtime was " << duration.count() << std::endl;
    #endif
    return EXIT_SUCCESS;
}
