cmake_minimum_required(VERSION 3.20)
project(superfamiconv LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_OSX_ARCHITECTURES arm64 x86_64)
endif()

if(MSVC)
  add_definitions(-DNOMINMAX)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif()

if(MSVC)
  set(SOURCES include/fmt/format.cpp include/LodePNG/lodepng.cpp include/getopt-win/getopt.c src/superfamiconv.cpp src/sfc_palette.cpp src/sfc_tiles.cpp src/sfc_map.cpp src/Image.cpp src/Map.cpp src/Palette.cpp src/Tiles.cpp src/Optimizer.cpp)
else()
  set(SOURCES include/fmt/format.cpp include/LodePNG/lodepng.cpp src/superfamiconv.cpp src/sfc_palette.cpp src/sfc_tiles.cpp src/sfc_map.cpp src/Image.cpp src/Map.cpp src/Palette.cpp src/Tiles.cpp src/Optimizer.cpp)
endif()

add_executable(superfamiconv ${SOURCES})

# Optimizer uses large stack arrays - increase stack size
# Static linking for standalone binary on MinGW
if(MSVC)
  target_link_options(superfamiconv PRIVATE /STACK:8388608)
else()
  target_link_options(superfamiconv PRIVATE -Wl,--stack,8388608 -static)
endif()
