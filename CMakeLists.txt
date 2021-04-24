cmake_minimum_required(VERSION 3.15.0)
cmake_policy(SET CMP0048 NEW)
project(cupoch_contrib LANGUAGES C CXX VERSION 0.1)

find_package(catkin REQUIRED COMPONENTS
        ga_cmake
)

find_package(GFlags REQUIRED QUIET)
find_package(thrust REQUIRED QUIET)

GA_CHECK_CUDA()

set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "")

APPEND_TARGET_ARCH_FLAGS()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUPOCH_NVCC_FLAGS})
add_definitions(${CUPOCH_DEFINITIONS})
message(STATUS "zs: cupoch_contrib CUDA_NVCC_FLAGS = " ${CUDA_NVCC_FLAGS})


###################### ros configurations########################################

include_directories(
    .
    ${catkin_INCLUDE_DIRS}
    ${3RDPARTY_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

catkin_package(
    INCLUDE_DIRS ${3RDPARTY_INCLUDE_DIRS}
    INCLUDE_DIRS .
    LIBRARIES
        ${3RDPARTY_LIBRARIES}
    CATKIN_DEPENDS
        ga_cmake
)

add_subdirectory(modules)

if(GA_BUILD_TEST)
    # Don't enable testing until after building dependencies
    enable_testing()
    # in vscode using CTest as a test client
    include(CTest)
#     add_subdirectory(tests)
endif()

# Setup install
# install(DIRECTORY scripts/
#     DESTINATION ${CMAKE_INSTALL_SHAREDIR}/scripts
# )
# install(DIRECTORY cmake/
#     DESTINATION ${CMAKE_INSTALL_SHAREDIR}/cmake
# )
# install(DIRECTORY doc/pdf
#     DESTINATION ${CMAKE_INSTALL_SHAREDIR}/doc
# )
# install(DIRECTORY doc/md
#     DESTINATION ${CMAKE_INSTALL_SHAREDIR}/doc
# )
# install(DIRECTORY example/
#     DESTINATION ${CMAKE_INSTALL_SHAREDIR}/example
# )
# install(DIRECTORY config/
#     DESTINATION ${CMAKE_INSTALL_BINDIR}/config
# )
# install(DIRECTORY log/
#     DESTINATION ${CMAKE_INSTALL_BINDIR}/log
# )