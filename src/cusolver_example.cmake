function(add_cusolver_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCES})
set_property(TARGET ${EXAMPLE_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
target_include_directories(${EXAMPLE_NAME}
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(${EXAMPLE_NAME}
    PUBLIC
        cusolver
        cublas
        cublasLt
        cusparse
        cusolverMg
)
set_target_properties(${EXAMPLE_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Install example
install(
    TARGETS ${EXAMPLE_NAME}
    RUNTIME
    DESTINATION ${cusolver_examples_BINARY_INSTALL_DIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
)

add_dependencies(${GROUP_TARGET} ${EXAMPLE_NAME})
endfunction()

add_custom_target(cusolver_examples)