# CMAKE generated file: DO NOT EDIT!
# Generated by CMake Version 3.27
cmake_policy(SET CMP0009 NEW)

# HEADERS at NeuromorphicNet/CMakeLists.txt:1 (file)
file(GLOB NEW_GLOB LIST_DIRECTORIES true "D:/Package/NeuromorphicNet/NeuromorphicNet/include/*.h")
set(OLD_GLOB
  "D:/Package/NeuromorphicNet/NeuromorphicNet/include/IFNeuron.h"
  "D:/Package/NeuromorphicNet/NeuromorphicNet/include/NeuralNetGenerator.h"
  "D:/Package/NeuromorphicNet/NeuromorphicNet/include/SpikeGenerator.h"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "D:/Package/NeuromorphicNet/out/build/x64-debug/CMakeFiles/cmake.verify_globs")
endif()

# HEADERS at NeuromorphicNet/CMakeLists.txt:1 (file)
file(GLOB NEW_GLOB LIST_DIRECTORIES true "D:/Package/NeuromorphicNet/NeuromorphicNet/include/*.hpp")
set(OLD_GLOB
  "D:/Package/NeuromorphicNet/NeuromorphicNet/include/json.hpp"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "D:/Package/NeuromorphicNet/out/build/x64-debug/CMakeFiles/cmake.verify_globs")
endif()

# SOURCES at NeuromorphicNet/CMakeLists.txt:5 (file)
file(GLOB NEW_GLOB LIST_DIRECTORIES true "D:/Package/NeuromorphicNet/NeuromorphicNet/src/*.cpp")
set(OLD_GLOB
  "D:/Package/NeuromorphicNet/NeuromorphicNet/src/main.cpp"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "D:/Package/NeuromorphicNet/out/build/x64-debug/CMakeFiles/cmake.verify_globs")
endif()
