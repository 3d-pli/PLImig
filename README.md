# PLImig (PLI Mask and Inclination Generation)

- [Introduction](#introduction)
- [Functionality of this toolbox](#functionality-of-this-toolbox)
- [System requirements](#system-requirements)
- [Required programs and packages](#required-programs-and-packages)
- [Optional programs and packages](#optional-programs-and-packages)
- [Install instructions](#install-instructions)
  - [Clone repository](#clone-repository)
  - [Compile the program](#compile-the-program)
  - [Install program](#install-program)
  - [Changing options](#changing-options)
- [Run the program](#run-the-program)
- [Performance measurements](#performance-measurements)

# Introduction
To better understand the brain's function and treat neurological diseases, a detailed reconstruction of the intricate and densely grown nerve fiber network is needed. The method 3D polarized light imaging (3D-PLI) has been developed a decade ago to analyze the three-dimensional course of nerve fibre pathways in brain tissue with micrometer resolution ([Axer et al. 2011](https://doi.org/10.1016/j.neuroimage.2010.08.075)). While the in-plane fiber directions can be determined with high accuracy, the computation of the out-of-plane fiber inclinations is more challenging because they are derived from the amplitude of the birefringence signals (retardation), which depends e.g. on the amount of nerve fibers. 
Here, we introduce an automated, optimized computation of the fiber inclinations, allowing for a much faster, reproducible determination of fiber orientations in 3D-PLI. Depending on the degree of myelination, the algorithm uses different models (transmittance-weighted, unweighted, or a linear combination), allowing to account for regionally specific behavior. As the algorithm is parallelized and GPU optimized, and uses images from standard 3D-PLI (retardation and transmittance), it can be applied to large data sets, also from previous measurements.


The software uses the (normalized) transmittance and retardation images from 3D-PLI measurements as input and computes parameter maps showing the regions with high myelination (HM), the regions with low myelination (LM), a high myelination probability map (P_HM) and the inclination. Common image formats (TIFF, NIfTI, HDF5) are supported. The computation runs completely automatically and requires no additional parameters. For testing purposes and to study different scenarios (e.g. unweighted vs. transmittance-weighted model), the user can define specific parameters for different steps in the computation.

The software uses CUDA for the GPU implementation, OpenCV as image container, and OpenMP (supporting multi-platform shared-memory multiprocessing programming). 

Further information about the program and the corresponding study can be found [here](https://arxiv.org/abs/2111.13783v1).


# Functionality of the toolbox

PLImig is designed as a standalone tool, but can also be used in other projects for preparation or processing steps. 
The three command line tool allows basic processing functionalities and can be used freely to process standard 3D-PLI measurements.
Please keep in mind that only **NTransmittance** files can be processed as non-normalized transmittance files could result in erroneous parameters and therefore also wrong inclination angles.
Normalized transmittance images are defined as the transmittance image divided by the transmittance of a measurement without sample. 

Applying a median filter before calling the tool is optional as **PLImig** does include a basic median filter functionality using CUDA.
The filter kernel size can only be changed by setting `MEDIAN_KERNEL_SIZE` before compilation. The default parameter is `5`. This value was chosen because it reduces artifacts of lower median kernels while keeping the basic structure of the tissue. Higher median kernels will result in stronger clouding artifacts in the inclination.

## Generation of masks

Both the mask and inclination require specific parameters which are determined by the histograms in both the median filtered transmittance
and retardation. For reference, both histograms are shown below:

**Retardation:**
![](./img/hist_retardation_256bins.png)
**Median10Transmittance:**
![](./img/hist_transmittance_256bins.png)

The histograms show similar features for most measurements. The retardation and transmittance show a distinct peak at the front and the back of the histogram, respectively. We use those peaks to define the background and separate regions with low and high myelination.
The separation of low and high myelination is then expanded by further parameters to include crossing and highly inclined regions of the tissue.

If a non median filtered NTransmittance is used as input, **PLImig** will generate the median5NTransmittance automatically and save it as **[...]\_median5NTransmittance\_[...].h5**. The dataset will match the original dataset of the input files or is set by the `--dataset` parameter when starting the program.

### T_ref
`T_ref` is considered as the average value of the transmittance within a connected region with the highest retardation values.
As the highest retardation values represent mostly flat white matter fibers, this can be used to get a first estimation of the white matter values in the transmittance.

To get the average transmittance value, masks based on a difference value are generated. There, the connected components algorithm
is executed.

![](./img/tMinExample.png)

If the number of pixels in the connected region is large enough (0.01% of the number of pixels in the image), the resulting mask will be used to calculate the mean transmittance value. Otherwise, the difference will be reduced by one bin size 1/256 = 0.00390625.

### T_back

All other parameters rely on the same procedure to calculate the value. As seen in the histograms above, both the transmittance and retardation show a somewhat smooth curve. Our interest is the point of maximum curvature which separates the desired regions from each other.

The curvature formula can be found [here](https://tutorial.math.lamar.edu/classes/calciii/curvature.aspx) for example and is adapted to be used on non-continous data.

As we are not able to calculate the maxima of the curvatures directly, we can choose the highest value of the curvatures for our point of maximum curvature. However, this might result in issues when there are slight variations in the histogram curve caused by higher bin sizes. To circumvent this issue, we can instead look at the number of peaks in our curvature plot and choose the first peak.

`T_back` separates the background of the transmittance from the tissue. 
The background is discernible by a clearly visible peak in the latter half of the histogram. The peak itself and all pixels with a value above the 
point of maximum curvature represent the background and will not be visible in the mask for low/high myelinated regions.

To find the maximum curvature, a search interval between the second half of the histogram and absolute maximum value is selected.
In between this interval, the next local minima in the left direction from the absolute maximum is selected as the left bound for the maximum curvature. 

An example for the resulting mask is shown below.

![](./img/tMaxExample.png)

### R_thres

After most of the necessary parameters are generated on the transmittance, one parameter in the retardation is needed to generate the desired masks for low/high myelinated regions. The general procedure follows the algorithm used for `T_back`. 

The resulting mask can be seen below. This mask generally gets most of the regions with high myelination, but might still miss a few areas. Those will be filled in combination with `T_thres`.

![](./img/tRetExample.png)


### T_thres

While `T_ref` is a good estimation in the transmittance, some fine fibers might not be caught by simply using the mean transmittance value. Therefore, we use the curvature formula again to estimate a point which contains more finer fibers without including too much of the gray matter. Our range will be limited by the next peak starting from `T_ref`.
If not enough values are present to calculate the curvatures, `T_ref` will be used as our value.

### Additional considerations for the algorithm
To ensure that our resulting thresholding parameters are not influenced by small interferences or more than one peak, we change the algorithm a bit.

We start using a histogram of only 64 bins to get a first estimation of our thresholding parameters. In each following iteration, we increase the number of
bins by a factor of 2 up to 256 bins. In each iteration, we take the last estimation and choose a interval around the last estimation for our current one. This ensures that we do not end in a small dip which becomes visible with higher bin counts.

In addition, if there is more than one peak in our interval, we start at the last peak. This is chosen because there might be a background peak resulting in erroneous parameters.

To ensure that our parameters are not influenced by the background, we implement a last quality improvement. We calculate `T_back` as our first parameter. As `T_back` is our only parameter which separates the tissue from our background, we can use this value to mask the tissue and evaluate all other parameters only with the tissue. This helps with an invalid calculation of `R_thres` for example, just because too many background pixels resulted in an invalid background peak. 


### HM-mask (regions with high myelination)
After generating all of our parameters, we can finally build our masks which separate low and high myelinated regions.
The formula for both masks as well as an example are shown below:

Mask[HM] = (0 < T < T_thres) or (R > R_thres)
![](./img/WhiteMaskExample.png)

### LM-mask (regions with low myelination)
Mask[LM] = (T_thres <= T <= T_back) or (R < R_thres)
![](./img/GrayMaskExample.png)

### HM-probability mask
While the parameters generated above are finite and will not change in subsequent executions, statistic errors like camera noise changes the results significantly. In addition, just using the HM-mask for our inclination calculation will result in sharp edges which do not represent the reality. Therefore the HM-probability mask is calculated.

Currently, 100 iterations on 25\% of the image size is used as our kernel. 
In each iteration, `T_ref` and `T_back` are fix values. A retardation and transmittance image will be generated by choosing random pixels from our initial retardation and median filtered transmittance. Pixels can be chosen multiple times. 

After the generation of a random image, we calculate `R_thres` and `T_thres` with our normal procedures and save values which differ from our initial values.

After our number of iterations, we calculate each pixel of our probability mask with the formula given in the paper. An example of such a probability mask is shown below. All values are in the range of $`[0, 1]`$.

![](./img/BlurredMaskExample.png)

### No nerve fiber mask
The LM-regions do not have as many fibers as the HM-regions. When calculating the inclination, some parts might be wrong because no fibers are present. This mask gives an esimation which parts of the LM-regions might not have any fibers. To achieve this, the mean and standard deviation of the background are used. Regions in the LM-regions with a value below mean + 2*stddev are considered as a region without any fibers.

![](./img/NoNerveFiberExample.png)

## Generation of the inclination

### T_M
The `T_M` value matches the calculation of `T_ref` in our mask generation.

### ic
`T_c` is considered as the mode of the transmittance in the gray substance. The gray substance is defined by our probability mask with values below 0.01. 

### R_ref,LM
The `R_ref,LM` calculation matches the calculation of `R_thres` in our mask generation.
This value will be used as our maximum value which is present in the gray substance.

### R_ref,HM
`R_ref,HM` will be calculated using the region growing algorithm described above. However, instead of using the resulting mask
to calculate the mean transmittance value like in `T_ref`, the mean retardation value of the mask is calculated.

### Inclination

The inclination formula will convert the transmittance and retardation values to an angle between 0° and 90° depending on our chosen parameters. Different formulas will be used for HM- and LM-regions. The inclination angles in regions that have values between zero and one in the HM-probability mask will be computed using a linear interplation of both formulas.

The formula for the inclination can be found in the paper.

![](./img/InclinationExample.png)

### Saturation map
The saturation map is an optional parameter map which is generated after the inclination. Here, all pixels with a value of 0° or below and 90° or above are marked by a value between 0 and 4.

The numbers can be interpreted like this:

- 0 -- No saturated pixel
- 1 -- Saturated pixel. Inclination angle is 0° or below. The retardation was higher than `R_ref,HM`
- 2 -- Saturated pixel. Inclination angle is 90° or above. The retardation was higher than `R_ref,HM`
- 3 -- Saturated pixel. Inclination angle is 0° or below. The retardation was lower than `R_ref,HM`
- 4 -- Saturated pixel. Inclination angle is 90° or above. The retardation was lower than `R_ref,HM`

![](./img/SaturationExample.png)


# System requirements
**Minimal requirements:**

* CPU: multicore processor recommended but not necessary.
* Memory: 8 GiB (32+ GiB recommended for large measurements)
* GPU: CUDA 9.0+ capable with 4+ GiB VRAM

# Required programs and packages
* CMake 3.14+
* C++-17 capable compiler (with OpenMP)
* Make
* OpenCV
* HDF5
* libNIFTI
* CUDA v10 or newer

# Optional programs and packages
For testing purposes:
* gcovr
* gcov
* Google Test

# Install instructions
Install all needed dependencies using your package manager or by compiling them from source.

## Setting up the program for execution
The following install instructions are exemplary for Ubuntu 20.04. Please replace `focal` with the code name or your Ubuntu distribution. The install instructions may vary based on your system.

### Install dependencies
```bash
apt-get update -qq && apt-get upgrade -y
apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common wget
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
apt-get update -qq
apt-get install -y gcc g++ cmake make build-essential file git
apt-get install -y libopencv-dev libhdf5-dev libnifti-dev
```

## Setting up the program for development
In addition to the install instructions for the exeuction of PLImig, some other packages are needed for tests and documentation.

Example using Ubuntu or Debian:
```bash
sudo apt-get install -y gcc g++ cmake make build-essential file git gcovr libgtest-dev doxygen
cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make
sudo cp ./lib/libgtest*.a /usr/lib
cd - 
```

### Clone the project
```bash
git clone git@github.com:3d-pli/PLImig.git
cd PLImig
```

## Compile the program
Execute the following commands in the project folder:
```bash
mkdir build
cd build/
cmake ..
make && make test
```

If everything ran successfully, the generated programs are located at `PLImig/build/bin/` and can be started from there.

## Changing options
By default, the following options are set:
```
BUILD_TESTING = ON
CMAKE_BUILD_TYPE = Release
CMAKE_INSTALL_PREFIX = /usr/local
```
You are able to change this options with `ccmake` or by defining them when calling `cmake`.

# Run the program
## PLIMaskGeneration
```
PLIMaskGeneration --itra [input-ntransmittance] --iret [input-retardation] --output [output-folder] [[parameters]]
```
### Required Arguments
| Argument      | Function                                                                    |
| ------------------- | --------------------------------------------------------------------------- |
| `--itra`  | One or more normalized transmittance files with the file format `.h5`, `.nii`, `.nii.gz` or `.tiff` |
| `--iret` | One or more retardation files with the file format `.h5`, `.nii`, `.nii.gz` or `.tiff` |
| `-o, --output` | Output folder for mask and median10NTransmittance, if generated. |

### Optional Arguments
| Argument      | Function                                                                    |
| -------------- | --------------------------------------------------------------------------- |
| `--dataset` | Read and write from/to the given dataset instead of `/Image` |
| `--tthres`  | Transmittance threshold. This threshold is near `T_ref` and will be set to the point of maximum curvature between `T_ref` and `T_back` |
| `--rthres` | Set the point of maximum curvature in the retardation histogram |
| `--tref` | Set the mean value of the transmittance in a connected region of the largest retardation values |
| `--tback` | Set the point of maximum curvature near the absolute maximum in the transmittance histogram |
| `--detailed` | Using this parameter will add two more parameter maps to the output file. This will include a full mask of both the HM- and LM-regions as well as a mask showing an approximation of regions without any nerve fibers. | 
| `--probability` | Create a floating point mask (HM-probability map) indicating regions that can be considered as the transition zone between HM- and LM-regions. This will be used to calculate the inclination image. |

## PLIInclination
```
PLIInclination --itra [input-ntransmittance] --iret [input-retardation] --imask [input-masks] --output [output-folder] [[parameters]]
```
### Required Arguments
| Argument      | Function                                                                    |
| ------------------- | --------------------------------------------------------------------------- |
| `--itra`  | One or more normalized transmittance files with the file format `.h5`, `.nii`, `.nii.gz` or `.tiff` |
| `--iret` | One or more retardation files with the file format `.h5`, `.nii`, `.nii.gz` or `.tiff` |
| `--imask` | One or more mask files generated by PLIMaskGeneration |
| `-o, --output` | Output folder for mask, inclination and median10NTransmittance, if generated. |

### Optional Arguments
| Argument      | Function                                                                    |
| -------------- | --------------------------------------------------------------------------- |
| `--dataset` | Read and write from/to the given dataset instead of `/Image` |
| `--tm`  | Mean value in the transmittance based on the highest retardation value|
| `--tc` | Maximum value in the LM-regions of the transmittance where the HM-probability is below 0.01 |
| `--rrefhm` | Mean value in the retardation based on the highest retardation values |
| `--rreflm` | Point of maximum curvature in the LM-regions of the retardation |
| `--detailed` | Add saturation map to the inclination HDF5 file marking each region with values <0° or >90° |

## PLImigPipeline
```
PLImigPipeline --itra [input-ntransmittance] --iret [input-retardation] --output [output-folder] [[parameters]]
```
### Required Arguments
| Argument      | Function                                                                    |
| ------------------- | --------------------------------------------------------------------------- |
| `--itra`  | One or more normalized transmittance files with the file format `.h5`, `.nii`, `.nii.gz` or `.tiff` |
| `--iret` | One or more retardation files with the file format `.h5`, `.nii`, `.nii.gz` or `.tiff` |
| `-o, --output` | Output folder for mask, inclination and median10NTransmittance, if generated. |

### Optional Arguments
| Argument      | Function                                                                    |
| -------------- | --------------------------------------------------------------------------- |
| `--dataset` | Read and write from/to the given dataset instead of `/Image` |
| `--tthres`  | Transmittance threshold. This threshold is near `T_ref` and will be set to the point of maximum curvature between `T_ref` and `T_back` |
| `--rthres` | Set the point of maximum curvature in the retardation histogram |
| `--tref` | Set the mean value of the transmittance in a connected region of the largest retardation values |
| `--tback` | Set the point of maximum curvature near the absolute maximum in the transmittance histogram |
| `--detailed` | Using this parameter will add two more parameter maps to the output file. This will include a full mask of both the HM- and LM-regions as well as a mask showing an approximation of regions without any nerve fibers. The inclination file will also include a parameter map indicating saturated pixels. |

# Performance measurements
The determination of the histogram threshold parameters and the computation of the fiber inclinations are completely implemented on the GPU, so that no speedup is expected when increasing the number of CPU cores. The generation of the probability map uses CPUs, so that the computing time can be reduced with increasing numbers of CPU cores (running multiple iterations in parallel on the GPU). 

The size of the graphics memory determines the number of parallel iterations: each iteration requires twice the image size used for bootstrapping (here: 25\,\% of the original image size). As only histograms are calculated in each iteration, this part is memory and bandwidth bound on the GPU side, and CPU bound in the time it takes to create the random image. With increasing number of CPU cores, the program will be more GPU bound.
