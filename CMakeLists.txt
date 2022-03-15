

cmake_minimum_required(VERSION 3.12.0 FATAL_ERROR)
set(VCPKG_FEATURE_FLAGS versions)
project(tif2jxl)

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS} ${OpenCV_LIBS} ")

include_directories( ${OpenCV_INCLUDE_DIRS})

include_directories("D:/Checkout_2020/libjxl/" "D:/Checkout_2020/libjxl/lib/include/" "D:/Checkout_2020/libjxl/third_party/highway/" "D:/Checkout_2020/libjxl/build/lib/include/")
link_directories("D:/Checkout_2020/libjxl/build/lib/Debug/" "D:/Checkout_2020/libjxl/build/Debug/" "D:/Checkout_2020/libjxl/build/lib/Debug/")

#link_directories("$(WindowsSdkDir)\lib\x64")

find_package(Qt5Core CONFIG REQUIRED)
find_package(Qt5Concurrent CONFIG REQUIRED)

add_executable(tif2jxl main.cpp )

#target_link_directories(tif2jxl "D:\\Checkout_2020\\libjxl\\build\\lib\\Debug\\" "D:\\Checkout_2020\\libjxl\\build\\Debug\\") 
#target_link_directories(tif2jxl 
# "D:/Checkout_2020/libjxl/build/lib/Debug/"
# "D:/Checkout_2020/libjxl/build/Debug/"     
# )

target_link_libraries(tif2jxl
   jxl jxl_threads
    ${OpenCV_LIBS}
    Qt::Core Qt::Concurrent Qt5::Concurrent Qt::ConcurrentPrivate Qt5::ConcurrentPrivate Qt5::Core Qt::WinMain Qt5::WinMain
)
