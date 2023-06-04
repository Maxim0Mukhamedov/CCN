cmake_minimum_required(VERSION 3.21)
project(ccn_example)

find_package(OpenCV CONFIG REQUIRED)

include_directories(${OpenCV_INCLUDE_DIR})
message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIR})

add_executable(${PROJECT_NAME} example_cmd_app.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ccn)


set(CMAKE_INSTALL_PREFIX "./out")

install(TARGETS ${PROJECT_NAME})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test_input DESTINATION ./bin)
