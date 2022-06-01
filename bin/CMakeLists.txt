# Set output folders to bin and lib
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(tools)

# normal executables
add_executable(PLIMaskGeneration MaskGeneration.cpp)
target_link_libraries(PLIMaskGeneration PLImig)

add_executable(PLIInclination CalcIncl.cpp)
target_link_libraries(PLIInclination PLImig)

add_executable(PLImigPipeline PLImig.cpp)
target_link_libraries(PLImigPipeline PLImig)

# add_executable(PLImigCheckResults CheckResults.cpp)
# target_link_libraries(PLImigCheckResults ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})


# install instructions for CMake
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    install(TARGETS PLImig LIBRARY PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ DESTINATION lib)
    install(TARGETS PLIMaskGeneration LIBRARY PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE DESTINATION bin)
    install(TARGETS PLIInclination LIBRARY PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE DESTINATION bin)
    install(TARGETS PLImigPipeline LIBRARY PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE DESTINATION bin)
    # install(TARGETS PLImigCheckResults LIBRARY PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE DESTINATION bin)
    install(DIRECTORY DESTINATION include/PLImig)
    install(FILES ${HEADER} DESTINATION include/PLImig PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)
endif()
