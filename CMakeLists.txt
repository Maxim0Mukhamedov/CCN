cmake_minimum_required(VERSION 3.24)
project(CurseJob)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV CONFIG REQUIRED)

include_directories(${OpenCV_INCLUDE_DIR})
message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIR})


add_executable(CurseJob main.cpp)
target_link_libraries(CurseJob ${OpenCV_LIBS})
