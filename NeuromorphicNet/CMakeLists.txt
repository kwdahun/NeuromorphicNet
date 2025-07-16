

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
set(SNN_TRAINING_EXAMPLE "src/snn_training_example.cpp")
set(SNN_ADAM_TRAINING_EXAMPLE "src/snn_adam_training_example.cpp")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(main ${NEUROMORPHICNET_MAIN} ${HEADERS})
add_executable(classification_general_spikegen ${CLASSIFICATION_GENERAL_SPIKEGEN} ${HEADERS})
add_executable(classification_diet_spikegen ${CLASSIFICATION_DIET_SPIKEGEN} ${HEADERS})
add_executable(snn_training_example ${SNN_TRAINING_EXAMPLE} ${HEADERS})
add_executable(snn_adam_training_example ${SNN_ADAM_TRAINING_EXAMPLE} ${HEADERS})

target_compile_definitions(main PRIVATE
    MNIST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/MNIST"
)
target_compile_definitions(classification_general_spikegen PRIVATE
    MNIST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/MNIST"
)
target_compile_definitions(classification_diet_spikegen PRIVATE
    MNIST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/MNIST"
)
target_compile_definitions(snn_training_example PRIVATE
    MNIST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/MNIST"
)
target_compile_definitions(snn_adam_training_example PRIVATE
    MNIST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/MNIST"
    CHECKPOINT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/checkpoints"
    MODEL_CHECKPOINT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/checkpoints/MNIST/best_model.bin"
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
    target_link_libraries(classification_general_spikegen ${OpenBLAS_LIB})
    target_link_libraries(classification_diet_spikegen ${OpenBLAS_LIB})
    target_link_libraries(snn_training_example ${OpenBLAS_LIB})
    target_link_libraries(snn_adam_training_example ${OpenBLAS_LIB})
else()
    target_link_libraries(main ${OPENBLAS_LIBRARIES})
    target_link_libraries(classification_general_spikegen ${OPENBLAS_LIBRARIES})
    target_link_libraries(classification_diet_spikegen ${OPENBLAS_LIBRARIES})
    target_link_libraries(snn_training_example ${OPENBLAS_LIBRARIES})
    target_link_libraries(snn_adam_training_example ${OPENBLAS_LIBRARIES})
endif()
