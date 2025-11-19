rm -rf ./build-android
mkdir ./build-android
export ANDROID_NDK="/home/user/Android/Sdk/ndk/29.0.13599879" 
sleep 1

cmake \
  -S . \
  -B build-android \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-33 \
  -DGGML_LLAMAFILE=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_COMMON=ON \
  -DGGML_OPENMP=ON \
  -DGGML_CCACHE=OFF \
#  -DGGML_OPENMP=OFF \
#  -DGGML_CCACHE=OFF \
#  -DCMAKE_C_FLAGS="-march=armv8.7a" \
#  -DCMAKE_CXX_FLAGS="-march=armv8.7a" \
cmake --build build-android --config Release -j128

dir=./build-android-arm64-v8a
rm -rf $dir
mkdir $dir
cmake --install build-android --prefix $dir --config Release

LIBOMP_PATH="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/20/lib/linux/aarch64/libomp.so"

if [ -f "$LIBOMP_PATH" ]; then
  cp "$LIBOMP_PATH" "$dir/lib/"
  echo "libomp.so copied successfully."
else
  echo "Warning: libomp.so not found at $LIBOMP_PATH. Please verify the path."
fi
