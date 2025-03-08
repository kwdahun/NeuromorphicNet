

if(WIN32)
    set(OpenBLAS_DIR "C:/OpenBLAS")
    set(OpenBLAS_LIB "libopenblas")
    include_directories(${OpenBLAS_DIR}/include)
    link_directories(${OpenBLAS_DIR}/lib)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(OPENBLAS REQUIRED openblas)
    include_directories(${OPENBLAS_INCLUDE_DIRS})
    link_directories(${OPENBLAS_LIBRARY_DIRS})
endif()

file(GLOB HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
)

set(NEUROMORPHICNET_MAIN "src/main.cpp")
set(CLASSIFICATION_GENERAL_SPIKEGEN "src/classification_general_spikegen.cpp")
set(CLASSIFICATION_DIET_SPIKEGEN "src/classification_diet_spikegen.cpp")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(main ${NEUROMORPHICNET_MAIN} ${HEADERS})
add_executable(classification_general_spikegen ${CLASSIFICATION_GENERAL_SPIKEGEN} ${HEADERS})
add_executable(classification_diet_spikegen ${CLASSIFICATION_DIET_SPIKEGEN} ${HEADERS})

target_compile_definitions(main PRIVATE
    MNIST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/MNIST"
)
target_compile_definitions(classification_general_spikegen PRIVATE
    MNIST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/MNIST"
)
target_compile_definitions(classification_diet_spikegen PRIVATE
    MNIST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/MNIST"
)

# DLL copy
if(WIN32)
    add_custom_command(TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenBLAS_DIR}/bin/libopenblas.dll"
        $<TARGET_FILE_DIR:main>
    )
endif()

if(WIN32)
    target_link_libraries(main ${OpenBLAS_LIB})
else()
    target_link_libraries(main ${OPENBLAS_LIBRARIES})
endif()
