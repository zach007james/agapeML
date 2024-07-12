cmake_minimum_required(VERSION 3.10)
project(agapeML)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include the Eigen directory
include_directories(${PROJECT_SOURCE_DIR}/external/eigen)

# Include the gnuplot-iostream directory
include_directories(${PROJECT_SOURCE_DIR}/external/gnuplot-iostream)

# Set up the location of Boost depending on the operating system
if(WIN32)
    set(BOOST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/boost_multi/boost_1_85_0_win")
    set(BOOST_LIBRARYDIR "${BOOST_ROOT}/stage/lib")
else()
    set(BOOST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/boost_multi/boost_1_85_0_unix")
    set(BOOST_LIBRARYDIR "${BOOST_ROOT}/stage/lib")
endif()

set(Boost_INCLUDE_DIR "${BOOST_ROOT}")

# Ensure all necessary Boost libraries are built
if(WIN32)
    set(BOOST_SYSTEM_LIB "${BOOST_LIBRARYDIR}/libboost_system-vc141-mt-x64-1_85.lib")
    set(BOOST_FILESYSTEM_LIB "${BOOST_LIBRARYDIR}/libboost_filesystem-vc141-mt-x64-1_85.lib")
    set(BOOST_IOSTREAMS_LIB "${BOOST_LIBRARYDIR}/libboost_iostreams-vc141-mt-x64-1_85.lib")
    set(BOOST_REGEX_LIB "${BOOST_LIBRARYDIR}/libboost_regex-vc141-mt-x64-1_85.lib")
else()
    set(BOOST_SYSTEM_LIB "${BOOST_LIBRARYDIR}/libboost_system.a")
    set(BOOST_FILESYSTEM_LIB "${BOOST_LIBRARYDIR}/libboost_filesystem.a")
    set(BOOST_IOSTREAMS_LIB "${BOOST_LIBRARYDIR}/libboost_iostreams.a")
    set(BOOST_REGEX_LIB "${BOOST_LIBRARYDIR}/libboost_regex.a")
endif()

if(NOT EXISTS ${BOOST_SYSTEM_LIB} OR NOT EXISTS ${BOOST_FILESYSTEM_LIB} OR NOT EXISTS ${BOOST_IOSTREAMS_LIB} OR NOT EXISTS ${BOOST_REGEX_LIB})
    message(STATUS "Boost libraries not found. Attempting to build Boost now...")
    if(WIN32)
        execute_process(
            COMMAND bootstrap.bat
            WORKING_DIRECTORY ${BOOST_ROOT}
        )
        execute_process(
            COMMAND b2 link=static threading=multi address-model=64 --with-system --with-filesystem --with-iostreams --with-regex
            WORKING_DIRECTORY ${BOOST_ROOT}
        )
    else()
        execute_process(
            COMMAND ./bootstrap.sh --with-libraries=system,filesystem,iostreams,regex
            WORKING_DIRECTORY ${BOOST_ROOT}
        )
        execute_process(
            COMMAND ./b2 link=static threading=multi
            WORKING_DIRECTORY ${BOOST_ROOT}
        )
    endif()
endif()

find_package(Boost 1.85.0 COMPONENTS system filesystem iostreams regex REQUIRED)

include_directories(${Boost_INCLUDE_DIR})
link_directories(${BOOST_LIBRARYDIR})

# Add the executable
add_executable(agapeML src/main.cpp 
                       src/util/data_processing.cpp
                       includes/util/data_processing.hpp
                       src/models/linear_regression.cpp
                       includes/models/linear_regression.hpp
                       src/util/scaler.cpp
                       includes/util/scaler.hpp)

target_link_libraries(agapeML Boost::system Boost::filesystem Boost::iostreams Boost::regex)

# Check if Gnuplot is available
find_program(GNUPLOT_EXECUTABLE gnuplot)
if(NOT GNUPLOT_EXECUTABLE)
    message(FATAL_ERROR "Gnuplot not found")
endif()
