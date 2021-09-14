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
#include "CLI/CLI.hpp"
#include "version.h"

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
    bool blurred = false;

    float tmin, tmax, tret, ttra;

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
    optional->add_flag("--probability", blurred);
    auto parameters = optional->add_option_group("Parameters", "Control the generated masks by setting parameters manually");
    parameters->add_option("--ilower", ttra, "Average transmittance value of brightest retardation values")
              ->default_val(-1);
    parameters->add_option("--rthres", tret, "Plateau in retardation histogram")
              ->default_val(-1);
    parameters->add_option("--irmax", tmin, "Average transmittance value of brightest retardation values")
              ->default_val(-1);
    parameters->add_option("--iupper", tmax, "Separator of gray matter and background")
              ->default_val(-1);
    CLI11_PARSE(app, argc, argv);

    PLImg::HDF5Writer writer;
    PLImg::MaskGeneration generation;

    std::string transmittance_basename, mask_basename;
    std::string transmittance_path, retardation_path;
    // Output paths
    std::string median_transmittance_path, mask_path;

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
        for(std::string& extension : std::array<std::string, 5> {".h5", ".tiff", ".tif", ".nii.gz", ".nii"}) {
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

        std::shared_ptr<cv::Mat> transmittance = std::make_shared<cv::Mat>(
                PLImg::Reader::imread(transmittance_path, dataset));
        std::shared_ptr<cv::Mat> retardation = std::make_shared<cv::Mat>(
                PLImg::Reader::imread(retardation_path, dataset));
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
        transmittance = nullptr;

        generation.setModalities(retardation, medTransmittance);
        if(ttra >= 0) {
            generation.set_tTra(ttra);
        }
        if(tret >= 0) {
            generation.set_tRet(tret);
        }
        if(tmin >= 0) {
            generation.set_tMin(tmin);
        }
        if(tmax >= 0) {
            generation.set_tMax(tmax);
        }

        writer.set_path(output_folder + "/" + mask_basename + ".h5");
        writer.write_dataset("/Image", *generation.fullMask(), true);
        writer.writePLIMAttributes({median_transmittance_path, retardation_path}, "/Image", "/Image", "Mask", argc, argv);
        writer.write_attribute("/Image", "i_lower", generation.tTra());
        writer.write_attribute("/Image", "r_thres", generation.tRet());
        writer.write_attribute("/Image", "i_rmax", generation.tMin());
        writer.write_attribute("/Image", "i_upper", generation.tMax());
        // writer.write_attribute("/Image", "version", PLImg::Version::versionHash() + ", " + PLImg::Version::timeStamp());
        std::cout << "Full mask generated and written" << std::endl;

        if (blurred) {
            writer.write_dataset("/Probability", *generation.probabilityMask());
            std::cout << "Probability mask generated and written" << std::endl;
        }
        if (detailed) {
            writer.write_dataset("/NoNerveFibers", *generation.noNerveFiberMask());
            std::cout << "Detailed masks generated and written" << std::endl;
        }
        writer.close();
        std::cout << std::endl;
    }

    #ifdef TIME_MEASUREMENT
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        std::cout << "Runtime was " << duration.count() << std::endl;
    #endif
    return EXIT_SUCCESS;
}
