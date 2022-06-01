set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Set source files
set(SOURCE
    inclination.cpp
    maskgeneration.cpp
    reader.cpp
    toolbox.cpp
    writer.cpp
    version.cpp
    cuda/cuda_toolbox.cu
    cuda/cuda_kernels.cu
    )

# Set header files
set(HEADER
    inclination.h
    maskgeneration.h
    reader.h
    toolbox.h
    writer.h
    version.h
    )

set(CUDA_HEADER
    cuda/exceptions.h
    cuda/cuda_toolbox.h
    cuda/cuda_kernels.h
    cuda/define.h
    )

# Generate executable file
if(WIN32)
    add_library(PLImig ${SOURCE})
else()
    add_library(PLImig SHARED ${SOURCE})
endif(WIN32)

target_link_libraries(PLImig ${OpenCV_LIBS} CLI11::CLI11 ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES} ${NIFTI_LIBRARIES}
        CUDA::cudart OpenMP::OpenMP_CXX OpenMP::OpenMP_C std::filesystem ${PLIM_LIBRARIES} ${PLIHDF5_LIBRARIES})

# install instructions for CMake
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    install(TARGETS PLImig LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ )
    install(DIRECTORY DESTINATION include/PLImig)
    install(FILES ${HEADER} DESTINATION include/PLImig PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)
    install(DIRECTORY DESTINATION include/PLImig/cuda)
    install(FILES ${CUDA_HEADER} DESTINATION include/PLImig/cuda PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)
endif()
