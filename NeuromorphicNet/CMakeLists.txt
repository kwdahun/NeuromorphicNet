file(GLOB HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
)
file(GLOB SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

set(NEUROMORPHICNET_MAIN "src/main.cpp")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(Main ${NEUROMORPHICNET_MAIN} ${SOURCES} ${HEADERS})

target_compile_definitions(Main PRIVATE
    MNIST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/MNIST"
)
