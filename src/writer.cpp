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

#include "writer.h"
#include <iostream>

PLImg::HDF5Writer::HDF5Writer() {
    m_filename = "";
}

std::string PLImg::HDF5Writer::path() {
    return this->m_filename;
}

void PLImg::HDF5Writer::set_path(const std::string& filename) {
    if(this->m_filename != filename) {
        this->m_filename = filename;
        this->open();
    }
}

void PLImg::HDF5Writer::write_dataset(const std::string& dataset, const cv::Mat& image) {
    PLI::HDF5::Dataset dset;
    PLI::HDF5::Type dtype(H5T_NATIVE_FLOAT);
    switch(image.type()) {
        case CV_32FC1:
        dtype = PLI::HDF5::Type(H5T_NATIVE_FLOAT);
        case CV_32SC1:
        dtype = PLI::HDF5::Type(H5T_NATIVE_INT);
        case CV_8UC1:
        dtype = PLI::HDF5::Type(H5T_NATIVE_UCHAR);
    }

    // Try to open the dataset.
    // This will throw an exception if the dataset doesn't exist.
    if(PLI::HDF5::Dataset::exists(this->m_hdf5file, dataset)) {
        dset = PLI::HDF5::openDataset(this->m_hdf5file, dataset);
    } else {
        // Create dataset normally
        // Check for the datatype from the OpenCV mat to determine the HDF5 datatype
        switch(image.type()) {
            case CV_32FC1:
            dtype = PLI::HDF5::Type(H5T_NATIVE_FLOAT);
            case CV_32SC1:
            dtype = PLI::HDF5::Type(H5T_NATIVE_INT);
            case CV_8UC1:
            dtype = PLI::HDF5::Type(H5T_NATIVE_UCHAR);
        }
        dset = PLI::HDF5::createDataset(this->m_hdf5file, dataset, {hsize_t(image.rows), hsize_t(image.cols)}, {hdf5_writer_chunk_dimensions[0], hdf5_writer_chunk_dimensions[1]}, dtype);

    }
    dset.write(image.data, {0u, 0u}, {hsize_t(image.rows), hsize_t(image.cols)}, dtype);
    dset.close();
    m_hdf5file.flush();
}

void PLImg::HDF5Writer::create_group(const std::string& group) {
    std::stringstream ss(group);
    std::string token;
    std::string groupString;

    PLI::HDF5::Group grp;
    // Create groups recursively if the group doesn't exist.
    while (std::getline(ss, token, '/')) {
        groupString.append("/").append(token);
        if(!token.empty()) {
            try {
                grp = PLI::HDF5::createGroup(this->m_hdf5file, groupString);
                grp.close();
            } catch(...){}
        }
    }
}

void PLImg::HDF5Writer::close() {
    m_hdf5file.close();
}

void PLImg::HDF5Writer::open() {
    createDirectoriesIfMissing(m_filename);
    // If the file doesn't exist open it with Read-Write.
    // Otherwise open it with appending so that existing content will not be deleted.
    if(PLI::HDF5::File::fileExists(m_filename)) {
        try {
            m_hdf5file = PLI::HDF5::openFile(m_filename, PLI::HDF5::File::ReadWrite);
        }  catch (...) {
            exit(EXIT_FAILURE);
        }
    } else {
        m_hdf5file = PLI::HDF5::createFile(m_filename);
    }
    #ifdef __GNUC__
        sleep(1);
    #else
        Sleep(1000);
    #endif
}

void PLImg::HDF5Writer::createDirectoriesIfMissing(const std::string &filename) {
    // Get folder name
    auto pos = filename.find_last_of('/');
    if(pos != std::string::npos) {
        std::string folder_name = filename.substr(0, filename.find_last_of('/'));
        std::error_code err;
        std::filesystem::create_directory(folder_name, err);
        if(err.value() != 0) {
            throw std::runtime_error("Output folder " + folder_name + " could not be created! Please check your path and permissions");
        }
    }
}

void PLImg::HDF5Writer::writePLIMAttributes(const std::vector<std::string>& reference_maps,
                                            const std::string& output_dataset, const std::string& input_dataset,
                                            const std::string& modality, const int argc, char** argv) {
    PLI::HDF5::Group grp;
    PLI::HDF5::Dataset dset;
    PLI::HDF5::AttributeHandler attrHandler;
    try {
        grp = PLI::HDF5::openGroup(this->m_hdf5file, output_dataset);
        attrHandler.setPtr(grp);
    } catch(...) {
        dset = PLI::HDF5::openDataset(this->m_hdf5file, output_dataset);
        attrHandler.setPtr(dset);
    }
    PLI::PLIM plim(attrHandler);

    if(attrHandler.attributeExists("image_modality")) {
        attrHandler.deleteAttribute("image_modality");
    }
    attrHandler.createAttribute<std::string>("image_modality", modality);

    plim.addCreator();
    if(attrHandler.attributeExists("creation_time")) {
        attrHandler.deleteAttribute("creation_time");
    }
    attrHandler.createAttribute<std::string>("creation_time", Version::timeStamp());


    plim.addSoftware(std::filesystem::path(argv[0]).filename());
    plim.addSoftwareRevision(Version::versionHash());
    std::string software_parameters;
    for(int i = 1; i < argc; ++i) {
        software_parameters += std::string(argv[i]) + " ";
    }
    plim.addSoftwareParameters(software_parameters);

    std::vector<PLI::HDF5::File> reference_files;
    std::vector<PLI::HDF5::Dataset> reference_datasets;
    std::vector<PLI::HDF5::AttributeHandler> reference_modalities;
    for(auto& reference : reference_maps) {
        if(reference.find(".h5") != std::string::npos) {
            try {
                reference_files.emplace_back();
                reference_files.back().open(reference, PLI::HDF5::File::ReadOnly);
                reference_datasets.push_back(PLI::HDF5::openDataset(reference_files.back(), input_dataset));
                PLI::HDF5::AttributeHandler handler(reference_datasets.back());
                reference_modalities.push_back(handler);

                handler.copyAllTo(attrHandler, {});
            } catch (PLI::HDF5::Exceptions::HDF5RuntimeException& e) {
                std::cerr << e.what() << std::endl;
            } catch (PLI::HDF5::Exceptions::IdentifierNotValidException& e) {
                std::cerr << e.what() << std::endl;
            }
        }
    }

    plim.addReference(reference_modalities);
    plim.addID({});

    for(auto& reference : reference_datasets) {
        reference.close();
    }
    for(auto& reference : reference_files) {
        reference.close();
    }

    grp.close();
    dset.close();
}
