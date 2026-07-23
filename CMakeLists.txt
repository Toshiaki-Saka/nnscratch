# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)

project(nnscratch
    VERSION 0.1.0
    DESCRIPTION "A tiny, dependency-free neural network library written from scratch in modern C++"
    LANGUAGES CXX)

# ---- Project-is-top-level detection (CMake 3.21+ has PROJECT_IS_TOP_LEVEL) ----
if(NOT DEFINED PROJECT_IS_TOP_LEVEL)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        set(PROJECT_IS_TOP_LEVEL TRUE)
    else()
        set(PROJECT_IS_TOP_LEVEL FALSE)
    endif()
endif()

# ---- Options -----------------------------------------------------------------
option(NNSCRATCH_BUILD_APPS    "Build the example/demo applications" ${PROJECT_IS_TOP_LEVEL})
option(NNSCRATCH_BUILD_TESTS   "Build the unit tests"                ${PROJECT_IS_TOP_LEVEL})
option(NNSCRATCH_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

# ---- Standard / output -------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

include(GNUInstallDirs)

# ---- Warning interface -------------------------------------------------------
add_library(nnscratch_warnings INTERFACE)
if(MSVC)
    target_compile_options(nnscratch_warnings INTERFACE /W4 /utf-8)
    if(NNSCRATCH_WARNINGS_AS_ERRORS)
        target_compile_options(nnscratch_warnings INTERFACE /WX)
    endif()
else()
    target_compile_options(nnscratch_warnings INTERFACE
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
    if(NNSCRATCH_WARNINGS_AS_ERRORS)
        target_compile_options(nnscratch_warnings INTERFACE -Werror)
    endif()
endif()

# ---- Core library ------------------------------------------------------------
add_library(nnscratch
    src/tensor.cpp
    src/activations.cpp
    src/layers.cpp
    src/conv2d.cpp
    src/loss.cpp
    src/optimizer.cpp
    src/model.cpp
    src/training.cpp
    src/dataset.cpp)
add_library(nnscratch::nnscratch ALIAS nnscratch)

target_compile_features(nnscratch PUBLIC cxx_std_20)
target_link_libraries(nnscratch PRIVATE $<BUILD_INTERFACE:nnscratch_warnings>)
target_include_directories(nnscratch PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(nnscratch PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})

# ---- Demo applications -------------------------------------------------------
if(NNSCRATCH_BUILD_APPS)
    foreach(demo from_scratch compare)
        add_executable(${demo} apps/${demo}.cpp)
        target_link_libraries(${demo} PRIVATE nnscratch nnscratch_warnings)
        target_compile_definitions(${demo} PRIVATE
            NNSCRATCH_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
    endforeach()
endif()

# ---- Tests -------------------------------------------------------------------
if(NNSCRATCH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---- Install / packaging -----------------------------------------------------
include(CMakePackageConfigHelpers)

install(TARGETS nnscratch
    EXPORT nnscratchTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT nnscratchTargets
    FILE nnscratchTargets.cmake
    NAMESPACE nnscratch::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nnscratch)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/nnscratchConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/nnscratchConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nnscratch)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/nnscratchConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/nnscratchConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/nnscratchConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nnscratch)
