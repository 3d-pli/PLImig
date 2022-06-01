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

bool PLImg::Reader::fileExists(const std::string& filename) {
    std::filesystem::path file{ filename };
    return std::filesystem::exists(file);
}

cv::Mat PLImg::Reader::imread(const std::string& filename, const std::string& dataset) {
    // Check if file exists
    if(fileExists(filename)) {
        // Opening the file has to be handeled differently depending on the file ending.
        // This will be done here.
        if(filename.substr(filename.size()-2) == "h5") {
            return readHDF5(filename, dataset);
        } else if(filename.substr(filename.size()-3) == "nii" || filename.substr(filename.size()-6) == "nii.gz"){
            return readNIFTI(filename);
        } else {
            return readTiff(filename);
        }
    } else {
        throw std::filesystem::filesystem_error("File not found: " + filename, std::error_code(10, std::generic_category()));
    }
}

cv::Mat PLImg::Reader::readHDF5(const std::string &filename, const std::string &dataset) {
    PLI::HDF5::File file = PLI::HDF5::openFile(filename);
    PLI::HDF5::Dataset dset = PLI::HDF5::openDataset(file, dataset);

    if(dset.ndims() > 2) {
        throw std::runtime_error("Expected 2D input image!");
    }
    auto dims = dset.dims();
    // Create OpenCV mat and copy content from dataset to mat
    cv::Mat image(dims[0], dims[1], CV_32FC1);
    auto flattenedImage = dset.readFullDataset<float>();

    std::copy(flattenedImage.begin(), flattenedImage.end(), image.begin<float>());

    dset.close();
    file.close();
    return image;
}

cv::Mat PLImg::Reader::readNIFTI(const std::string &filename) {
    nifti_image * img = nifti_image_read(filename.c_str(), 1);
    // Get image dimensions
    uint width = img->nx;
    uint height = img->ny;
    // Convert NIFTI datatype to OpenCV datatype
    uint datatype = img->datatype;
    uint cv_type;
    switch(datatype) {
        case 16:
            cv_type = CV_32FC1;
            break;
        case 8:
            cv_type = CV_32SC1;
            break;
        case 4:
            cv_type = CV_16SC1;
            break;
        case 2:
            cv_type = CV_8SC1;
        break;
        default:
            throw std::runtime_error("Did expect 32-bit floating point or 8/16/32-bit integer image!");
    }
    // Create OpenCV image with the image data
    cv::Mat image(height, width, cv_type);
    image.data = (uchar*) img->data;
    return image;
}

cv::Mat PLImg::Reader::readTiff(const std::string &filename) {
    return cv::imread(filename, cv::IMREAD_ANYDEPTH);
}

std::vector<std::string> PLImg::Reader::datasets(const std::string &filename) {
    std::vector<std::string> names;
    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    names = datasets(file);
    H5Fclose(file);
    return names;
}

std::vector<std::string> PLImg::Reader::datasets(hid_t group_id) {
    std::vector<std::string> names;

    char memb_name[1024];
    int id_type;
    herr_t error;

    hsize_t number_of_objects;
    error = H5Gget_num_objs(group_id, &number_of_objects);
    if(error) return names;

    for(unsigned i = 0; i < number_of_objects; ++i) {
        H5Gget_objname_by_idx(group_id, i, memb_name, 1024);
        id_type = H5Gget_objtype_by_idx(group_id, i);

        if(id_type == H5G_DATASET) {
            // Get the dimensions of our image dataset
            std::string dataset_string = ""; //[";

            /* dataset = H5Dopen(group_id, memb_name, H5P_DEFAULT);
            hid_t dataspace = H5Dget_space(dataset);
            hsize_t ndims = H5Sget_simple_extent_ndims(dataspace);
            hsize_t* dims = new hsize_t[ndims];
            H5Sget_simple_extent_dims(dataspace, dims, nullptr);
            for(unsigned dim = 0; dim < ndims; ++dim) {
                dataset_string += std::to_string(dims[dim]);
                if(dim + 1 < ndims) dataset_string += ", ";
            }
            dataset_string += "]";
            H5Sclose(dataspace);
            H5Dclose(dataset);
            delete [] dims;*/

            dataset_string = std::string(memb_name); //+ " " + dataset_string;
            names.push_back(dataset_string);
        } else if(id_type == H5G_GROUP) {
            hid_t group = H5Gopen(group_id, memb_name, H5P_DEFAULT);
            auto recursive_names = datasets(group);
            for(auto& name: recursive_names) {
                name = std::string(memb_name) + "/" + name;
            }
            names.insert(names.end(), recursive_names.begin(), recursive_names.end());
            H5Gclose(group);
        }
    }

    return names;
}

std::string PLImg::Reader::attribute(const std::string &filename, const std::string attributeName) {
    if(filename.substr(filename.size()-2) != "h5") {
        return "";
    }
    hid_t hdf5file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t image_dataset = H5Dopen(hdf5file, "/Image", H5P_DEFAULT);
    PLI::HDF5::AttributeHandler attribute_handler(image_dataset);

    if(!attribute_handler.attributeExists(attributeName)) {
        return "";
    }

    PLI::HDF5::Type type = attribute_handler.attributeType(attributeName);
    std::string return_value = "";

    if(type == H5T_NATIVE_FLOAT) {
        return_value = std::to_string(attribute_handler.getAttribute<float>(attributeName)[0]);
    } else if (type == H5T_NATIVE_INT) {
        return_value = std::to_string(attribute_handler.getAttribute<int>(attributeName)[0]);
    } else if (type == H5T_NATIVE_DOUBLE) {
        return_value = std::to_string(attribute_handler.getAttribute<double>(attributeName)[0]);
    } else if (type == H5T_NATIVE_UINT) {
        return_value = std::to_string(attribute_handler.getAttribute<unsigned int>(attributeName)[0]);
    }

    H5Dclose(image_dataset);
    H5Fclose(hdf5file);
    return return_value;
}
