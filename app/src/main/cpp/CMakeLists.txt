cmake_minimum_required(VERSION 3.18.1)

project("native-lib")

add_library(native-lib SHARED native-lib.cpp)

# Include headers
target_include_directories(native-lib PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Prebuilt libs (adjust paths if needed; Android Studio will use jniLibs)
add_library(ggml SHARED IMPORTED)
set_target_properties(ggml PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libggml.so)

add_library(llama SHARED IMPORTED)
set_target_properties(llama PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libllama.so)

add_library(mtmd SHARED IMPORTED)
set_target_properties(mtmd PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libmtmd.so)

# Link
target_link_libraries(native-lib ggml llama mtmd log)