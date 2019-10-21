########################################################################
# Sparse Grid librareis and command line tool
########################################################################

macro(Tasmanian_macro_add_libsparsegrid Tasmanian_shared_or_static)

if (${Tasmanian_shared_or_static} STREQUAL "SHARED")
    set(Tasmanian_libtsg_target_name "Tasmanian_libsparsegrid_shared")
else()
    set(Tasmanian_libtsg_target_name "Tasmanian_libsparsegrid_static")
endif()

if (Tasmanian_ENABLE_CUDA)
    # CMake does not automatically add the CUDA include path, any source file that adds <cuda.h> directly or transitively
    # must be labeled as "CUDA" source files and compiled with the nvcc compiler.
    # Beware that the nvcc compiler seems to struggle with some STL algorithms, transitive inclusion of <cuda.h> must be minimized.
    set_source_files_properties(${Tasmanian_source_lisparsegrid_cuda} tsgAcceleratedDataStructures.cpp PROPERTIES LANGUAGE CUDA)
    add_library(${Tasmanian_libtsg_target_name} ${Tasmanian_shared_or_static} ${Tasmanian_source_libsparsegrid} ${Tasmanian_source_libsparsegrid_cuda})
    set_property(TARGET ${Tasmanian_libtsg_target_name} PROPERTY CUDA_STANDARD 11)
    target_link_libraries(${Tasmanian_libtsg_target_name} ${Tasmanian_cudamathlibs}) # add the math libraries
else()
    add_library(${Tasmanian_libtsg_target_name} ${Tasmanian_shared_or_static} ${Tasmanian_source_libsparsegrid})
endif()

# Tasmanian_EXTRA_LIBRARIES gives the user an option to force extra dependencies,
# for example, on some systems (e.g., OLCF) find_package(BLAS) fails to
# recognize that libacml_mp requires libgomp, so the build fails with either clang or ENABLE_OPENMP=OFF
# -D Tasmanian_EXTRA_LIBRARIES=/path/to/libgomp.so circumvents the issue
# NOTE: adding Tasmanian_EXTRA_LIBRARIES to SparseGrids will propagate to all other targets
# same holds for Tasmanian_EXTRA_INCLUDE_DIRS
target_link_libraries(${Tasmanian_libtsg_target_name} ${Tasmanian_EXTRA_LIBRARIES})
target_include_directories(${Tasmanian_libtsg_target_name} PUBLIC $<INSTALL_INTERFACE:${Tasmanian_EXTRA_INCLUDE_DIRS}>)
target_include_directories(${Tasmanian_libtsg_target_name} PUBLIC $<BUILD_INTERFACE:${Tasmanian_EXTRA_INCLUDE_DIRS}>)

target_include_directories(${Tasmanian_libtsg_target_name} PUBLIC $<INSTALL_INTERFACE:${Tasmanian_final_install_path}/include>)

target_include_directories(${Tasmanian_libtsg_target_name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>)
target_include_directories(${Tasmanian_libtsg_target_name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../Config/>)
target_include_directories(${Tasmanian_libtsg_target_name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../configured/>)

set_target_properties(${Tasmanian_libtsg_target_name} PROPERTIES OUTPUT_NAME "tasmaniansparsegrid"
                                                                 INSTALL_RPATH "${Tasmanian_final_install_path}/lib"
                                                                 CXX_EXTENSIONS OFF)

target_compile_features(${Tasmanian_libtsg_target_name} PUBLIC cxx_std_14)

# as MAGMA potentially depends on OpenMP, BLAS, and CuBLAS, it should be set first
if (Tasmanian_ENABLE_MAGMA)
    target_include_directories(${Tasmanian_libtsg_target_name} PUBLIC $<BUILD_INTERFACE:${Tasmanian_MAGMA_INCLUDE_DIRS}/>)

    if (${Tasmanian_shared_or_static} STREQUAL "SHARED")
        target_link_libraries(${Tasmanian_libtsg_target_name} ${Tasmanian_MAGMA_SHARED_LIBRARIES})
    else()
        target_link_libraries(${Tasmanian_libtsg_target_name} ${Tasmanian_MAGMA_LIBRARIES})
    endif()
endif()

if (Tasmanian_ENABLE_BLAS)
    target_link_libraries(${Tasmanian_libtsg_target_name} ${BLAS_LIBRARIES})
endif()

if (Tasmanian_ENABLE_OPENMP)
    target_link_libraries(${Tasmanian_libtsg_target_name} ${OpenMP_CXX_LIBRARIES})
    # the nvcc compiler does nor recognize OpenMP, add the flag only to non-CUDA source files
    target_compile_options(${Tasmanian_libtsg_target_name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
endif()

install(TARGETS "${Tasmanian_libtsg_target_name}"
        EXPORT  "${Tasmanian_export_name}"
        RUNTIME DESTINATION "bin"
        LIBRARY DESTINATION "lib"
        ARCHIVE DESTINATION "lib")

unset(Tasmanian_libtsg_target_name)
endmacro()

########################################################################
# Define source files
# in order to avoid GLOB, list all source files so they can be used for
# add_library and cuda_add_library in both shared and static cases
########################################################################
set(Tasmanian_source_libsparsegrid
        TasmanianSparseGrid.hpp
        TasmanianSparseGrid.cpp
        TasmanianSparseGridWrapC.cpp
        tsgAcceleratedDataStructures.hpp
        tsgAcceleratedDataStructures.cpp
        tsgCacheLagrange.hpp
        tsgCoreOneDimensional.hpp
        tsgCoreOneDimensional.cpp
        tsgDConstructGridGlobal.hpp
        tsgDConstructGridGlobal.cpp
        tsgCudaLoadStructures.hpp
        tsgEnumerates.hpp
        tsgGridCore.hpp
        tsgGridGlobal.hpp
        tsgGridGlobal.cpp
        tsgGridWavelet.hpp
        tsgGridWavelet.cpp
        tsgHardCodedTabulatedRules.hpp
        tsgHardCodedTabulatedRules.cpp
        tsgHiddenExternals.hpp
        tsgGridLocalPolynomial.hpp
        tsgGridLocalPolynomial.cpp
        tsgGridSequence.hpp
        tsgGridSequence.cpp
        tsgGridFourier.hpp
        tsgGridFourier.cpp
        tsgIndexManipulator.hpp
        tsgIndexManipulator.cpp
        tsgHierarchyManipulator.hpp
        tsgHierarchyManipulator.cpp
        tsgIOHelpers.hpp
        tsgIndexSets.hpp
        tsgIndexSets.cpp
        tsgLinearSolvers.hpp
        tsgLinearSolvers.cpp
        tsgOneDimensionalWrapper.hpp
        tsgOneDimensionalWrapper.cpp
        tsgRuleLocalPolynomial.hpp
        tsgRuleWavelet.hpp
        tsgRuleWavelet.cpp
        tsgSequenceOptimizer.hpp
        tsgSequenceOptimizer.cpp
        tsgThreadEngine.hpp
        tsgMathUtils.hpp
        tsgUtils.hpp)

# source files specific to cuda, used for both static and shared libs
set(Tasmanian_source_libsparsegrid_cuda
       tsgCudaKernels.cu
       tsgCudaBasisEvaluations.hpp
       tsgCudaLinearAlgebra.hpp)


########################################################################
# add the tasgrid and examples executables
########################################################################
add_executable(Tasmanian_tasgrid tasgrid_main.cpp
                                 TasmanianSparseGrid.hpp
                                 tasgridCLICommon.hpp
                                 tasgridExternalTests.hpp
                                 tasgridExternalTests.cpp
                                 tasgridTestHelpers.hpp
                                 tasgridTestFunctions.hpp
                                 tasgridTestFunctions.cpp
                                 tasgridWrapper.hpp
                                 tasgridWrapper.cpp)
add_executable(Tasmanian_gridtest gridtest_main.cpp
                                  TasmanianSparseGrid.hpp
                                  tasgridCLICommon.hpp
                                  tasgridExternalTests.hpp
                                  tasgridExternalTests.cpp
                                  tasgridTestHelpers.hpp
                                  tasgridTestFunctions.hpp
                                  tasgridTestFunctions.cpp
                                  tasgridUnitTests.hpp
                                  tasgridUnitTests.cpp
                                  tasgridTestInterfaceC.cpp)
add_executable(Tasmanian_gridbenchmarks  Benchmarks/benchCommon.hpp
                                         tasgridCLICommon.hpp
                                         Benchmarks/benchMakeGrid.hpp
                                         Benchmarks/benchLoadNeeded.hpp
                                         Benchmarks/benchEvaluate.hpp
                                         Benchmarks/bench_main.cpp)

set_target_properties(Tasmanian_tasgrid PROPERTIES OUTPUT_NAME "tasgrid"
                                                   INSTALL_RPATH "${Tasmanian_final_install_path}/lib"
                                                   CXX_EXTENSIONS OFF)
set_target_properties(Tasmanian_gridtest PROPERTIES OUTPUT_NAME "gridtest" CXX_EXTENSIONS OFF)
set_target_properties(Tasmanian_gridbenchmarks PROPERTIES OUTPUT_NAME "benchmark" CXX_EXTENSIONS OFF)

if (Tasmanian_ENABLE_OPENMP)
    # the OpenMP libraries are carried transitively from sparse grids library
    target_compile_options(Tasmanian_tasgrid PRIVATE ${OpenMP_CXX_FLAGS})
    target_compile_options(Tasmanian_gridtest PRIVATE ${OpenMP_CXX_FLAGS})
endif()


########################################################################
# add the libraries
########################################################################
foreach(_tsglibtype ${Tasmanian_libs_type})
    string(TOUPPER ${_tsglibtype} Tasmanian_ltype)
    Tasmanian_macro_add_libsparsegrid(${Tasmanian_ltype})
endforeach()
unset(_tsglibtype)
unset(Tasmanian_ltype)

add_library(Tasmanian_libsparsegrid ALIAS Tasmanian_libsparsegrid_${Tasmanian_lib_default})

foreach(_tsgtarget tasgrid gridtest gridbenchmarks)
    target_link_libraries(Tasmanian_${_tsgtarget} Tasmanian_libsparsegrid)
    if (Tasmanian_ENABLE_OPENMP)
        # the OpenMP libraries are carried transitively from sparse grids library
        target_compile_options(Tasmanian_${_tsgtarget} PRIVATE ${OpenMP_CXX_FLAGS})
    endif()
endforeach()
unset(_tsgtarget)

# data file, needed for testing and reference about custom rule definitions
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/GaussPattersonRule.table"  "${CMAKE_CURRENT_BINARY_DIR}/GaussPattersonRule.table" COPYONLY)


########################################################################
# Windows specific support (DLL export/import directives and names)
########################################################################
if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    # setup the name, in MacOSX and Linux static libraries use .a extension and shared libs use .dylib and .so
    # in Windows, shared .dll libs need another file for linking which shares .lib extension with the static libs
    # thus on Windows, we specifically add _static to the name of the lib
    # furthermore, additional defines are needed to suppress extraneous warnings and adjust some system calls
    macro(Tasmanian_macro_sparsegrids_windows_defines Tasmanian_sg_target)
        target_compile_definitions(${Tasmanian_sg_target} PRIVATE -D_SCL_SECURE_NO_WARNINGS) # suppresses warnings regarding pointers to the middle of an array
        target_compile_definitions(${Tasmanian_sg_target} PUBLIC  -D_HAS_ITERATOR_DEBUGGING=0) # needed to prevent crash on using STL vector iterators
    endmacro()

    if (TARGET Tasmanian_libsparsegrid_shared)
        set_target_properties(Tasmanian_libsparsegrid_shared PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
        Tasmanian_macro_sparsegrids_windows_defines(Tasmanian_libsparsegrid_shared)
    endif()

    if (TARGET Tasmanian_libsparsegrid_static)
        Tasmanian_macro_sparsegrids_windows_defines(Tasmanian_libsparsegrid_static)
        set_target_properties(Tasmanian_libsparsegrid_static PROPERTIES OUTPUT_NAME "tasmaniansparsegrid_static")
    endif()

    Tasmanian_macro_sparsegrids_windows_defines(Tasmanian_tasgrid)
    Tasmanian_macro_sparsegrids_windows_defines(Tasmanian_gridtest)
endif()


########################################################################
# Testing
########################################################################
add_test(SparseGridsAcceleration gridtest acceleration -gpuid ${Tasmanian_TESTS_GPU_ID})
add_test(SparseGridsDomain       gridtest domain)
add_test(SparseGridsRefinement   gridtest refinement)
add_test(SparseGridsGlobal       gridtest global)
add_test(SparseGridsLocal        gridtest local)
add_test(SparseGridsWavelet      gridtest wavelet)
add_test(SparseGridsFourier      gridtest fourier)
add_test(SparseGridsExceptions   gridtest errors)
add_test(SparseGridsAPI          gridtest api)
add_test(SparseGridsC            gridtest c)
if (Tasmanian_TESTS_OMP_NUM_THREADS GREATER 0)
    set_tests_properties(SparseGridsAcceleration SparseGridsDomain SparseGridsRefinement SparseGridsGlobal SparseGridsLocal SparseGridsWavelet
        PROPERTIES
        PROCESSORS "${Tasmanian_TESTS_OMP_NUM_THREADS}"
        ENVIRONMENT "OMP_NUM_THREADS=${Tasmanian_TESTS_OMP_NUM_THREADS}")
endif()


########################################################################
# Install headers and config files
########################################################################
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION include
        FILES_MATCHING PATTERN "*.hpp"
        PATTERN "*.windows.*" EXCLUDE
        PATTERN "Examples" EXCLUDE
        PATTERN "Benchmarks" EXCLUDE
        PATTERN "tsgHiddenExternals.hpp" EXCLUDE)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/TasmanianSparseGrid.h"
        DESTINATION include
        PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Examples/"
        DESTINATION "share/Tasmanian/examples/"
        FILES_MATCHING PATTERN "*.cpp"
        PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/GaussPattersonRule.table"
        DESTINATION "share/Tasmanian/"
        PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ)

install(TARGETS Tasmanian_tasgrid
        EXPORT "${Tasmanian_export_name}"
        RUNTIME DESTINATION "bin"
        LIBRARY DESTINATION "lib"
        ARCHIVE DESTINATION "lib")
