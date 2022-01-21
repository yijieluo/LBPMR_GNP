cmake_minimum_required (VERSION 2.8)
project (LBPMR_GNP)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
include_directories("./Inc/")
file(GLOB source_files "./Src/*.cpp")

add_executable(a.out ${source_files})
target_link_libraries(a.out ${OpenCV_LIBS} -lpthread -lm)