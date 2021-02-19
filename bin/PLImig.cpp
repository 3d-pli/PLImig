#include "reader.h"
#include "writer.h"
#include "maskgeneration.h"
#include "inclination.h"
#include "CLI/CLI.hpp"

#include <vector>
#include <string>
#include <iostream>

int main(int argc, char** argv) {
    CLI::App app;

    std::vector<std::string> transmittance_files;
    std::vector<std::string> retardation_files;
    std::string output_folder;
    std::string dataset;
    bool detailed = false;

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
    auto parameters = optional->add_option_group("Parameters", "Control the generated masks by setting parameters manually");
    parameters->add_option("--ttra", ttra, "Average transmittance value of brightest retardation values")
              ->default_val(-1);
    parameters->add_option("--tret", tret, "Plateau in retardation histogram")
              ->default_val(-1);
    parameters->add_option("--tmin", tmin, "Average transmittance value of brightest retardation values")
              ->default_val(-1);
    parameters->add_option("--tmax", tmax, "Separator of gray matter and background")
              ->default_val(-1);
    CLI11_PARSE(app, argc, argv);

    PLImg::HDF5Writer writer;
    PLImg::MaskGeneration generation;
    PLImg::Inclination inclination;

    std::string transmittance_basename, retardation_basename, mask_basename, inclination_basename;
    std::string retardation_path, mask_path;
    bool retardation_found;

    for(const auto& transmittance_path : transmittance_files) {
        std::cout << transmittance_path << std::endl;

        unsigned long long int endPosition = transmittance_path.find_last_of('/');
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
        retardation_basename = std::string(transmittance_basename);
        if (retardation_basename.find("median10") != std::string::npos) {
            retardation_basename = retardation_basename.replace(retardation_basename.find("median10"), 8, "");
        }
        if (retardation_basename.find("NTransmittance") != std::string::npos) {
            retardation_basename = retardation_basename.replace(retardation_basename.find("NTransmittance"), 14, "Retardation");
        }
        if (retardation_basename.find("Transmittance") != std::string::npos) {
            retardation_basename = retardation_basename.replace(retardation_basename.find("Transmittance"), 13, "Retardation");
        }
        retardation_found = false;
        for(auto & retardation_file : retardation_files) {
            if(retardation_file.find(retardation_basename) != std::string::npos) {
                retardation_found = true;
                retardation_path = retardation_file;
                break;
            }
        }
        if(retardation_found) {
            mask_basename = std::string(retardation_basename);
            if (mask_basename.find("Retardation") != std::string::npos) {
                mask_basename = mask_basename.replace(mask_basename.find("Retardation"), 11, "Mask");
            }
            inclination_basename = std::string(mask_basename);
            if (mask_basename.find("Mask") != std::string::npos) {
                inclination_basename = inclination_basename.replace(inclination_basename.find("Mask"), 4, "Inclination");
            }

            std::shared_ptr<cv::Mat> transmittance = std::make_shared<cv::Mat>(
                    PLImg::Reader::imread(transmittance_path, dataset));
            std::shared_ptr<cv::Mat> retardation = std::make_shared<cv::Mat>(
                    PLImg::Reader::imread(retardation_path, dataset));
            std::cout << "Files read" << std::endl;

            std::shared_ptr<cv::Mat> medTransmittance = transmittance;
            if (transmittance_path.find("median10") == std::string::npos) {
                // Generate med10Transmittance
                medTransmittance = PLImg::cuda::filters::medianFilter(transmittance);
                // Write it to a file
                std::string medTraName(mask_basename);
                medTraName.replace(mask_basename.find("Mask"), 4, "median10NTransmittance");
                // Set file
                writer.set_path(output_folder + "/" + medTraName + ".h5");
                // Set dataset
                std::string group = dataset.substr(0, dataset.find_last_of('/'));
                // Create group and dataset
                writer.create_group(group);
                writer.write_dataset(dataset + "/", *medTransmittance);
                writer.close();
            } else {
                medTransmittance = transmittance;
            }
            std::cout << "Med10Transmittance generated" << std::endl;

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
            writer.create_group(dataset);
            writer.write_attributes("/", generation.tTra(), generation.tRet(), generation.tMin(), generation.tMax());
            std::cout << "Attributes generated and written" << std::endl;
            writer.write_dataset(dataset + "/White", *generation.whiteMask());
            std::cout << "White mask generated and written" << std::endl;
            writer.write_dataset(dataset + "/Gray", *generation.grayMask());
            std::cout << "Gray mask generated and written" << std::endl;
            writer.write_dataset(dataset + "/Blurred", *generation.blurredMask());
            std::cout << "Blurred mask generated and written" << std::endl;

            if (detailed) {
                writer.write_dataset(dataset + "/Full", *generation.fullMask());
                writer.write_dataset(dataset + "/NoNerveFibers", *generation.noNerveFiberMask());
                std::cout << "Detailed masks generated and written" << std::endl;
            }
            writer.close();

            if (transmittance_path.find("median10") == std::string::npos) {
                // Write it to a file
                std::string medTraName(mask_basename);
                medTraName.replace(mask_basename.find("Mask"), 4, "median10NTransmittanceMasked");
                // Set file
                writer.set_path(output_folder + "/" + medTraName + ".h5");
                // Set dataset
                std::string group = dataset.substr(0, dataset.find_last_of('/'));
                // Create group and dataset
                writer.create_group(group);

                // Generate med10Transmittance
                medTransmittance = PLImg::cuda::filters::medianFilterMasked(transmittance, generation.grayMask());
                writer.write_dataset(dataset + "/", *medTransmittance);
                transmittance = nullptr;
                writer.close();
            } else {
                medTransmittance = transmittance;
            }
            std::cout << "Median10 filtered and masked transmittance generated and written" << std::endl;

            // Set our read parameters
            inclination.setModalities(medTransmittance, retardation, generation.blurredMask(), generation.whiteMask(), generation.grayMask());
            // If manual parameters were given, apply them here

            inclination.set_im(generation.tMin());
            inclination.set_rmaxGray(generation.tRet());

            // Create file and dataset. Write the inclination afterwards.
            std::cout << output_folder+ "/" + inclination_basename + ".h5" << std::endl;
            writer.set_path(output_folder+ "/" + inclination_basename + ".h5");
            writer.create_group(dataset);
            writer.write_dataset(dataset+"/Inclination", *inclination.inclination());
            std::cout << "Inclination generated and written" << std::endl;

            if(detailed) {
                writer.write_dataset(dataset+"/Saturation", *inclination.saturation());
                std::cout << "Saturation image generated and written" << std::endl;
            }

            writer.close();
            std::cout << std::endl;
        }
    }

    return EXIT_SUCCESS;
}