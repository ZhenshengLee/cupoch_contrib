set(LIB_NAME "ga_cupoch_geometry")
# file(GLOB_RECURSE GEOMETRY_SOURCE_FILES "*.cu")
cuda_add_library(${LIB_NAME}
    # ${GEOMETRY_SOURCE_FILES}
    boundingvolume.cu
)

target_include_directories(${LIB_NAME} PRIVATE
        ${CUPOCH_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)
target_compile_definitions(${LIB_NAME} PRIVATE
        ${CUPOCH_DEFINITIONS}
)
target_link_directories(${LIB_NAME} PRIVATE
    ${CUPOCH_LIBRARY_DIRS}
)
target_link_libraries(${LIB_NAME}
    ${CUPOCH_LIBRARIES}
    ${jsoncpp_LIBRARIES}
)

install(TARGETS ${LIB_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
