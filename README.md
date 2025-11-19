# LlamaMtmdAndroid
an android app implementation of llama.cpp multimodel inference

U can contact me thr <img width="26" height="26" alt="image" src="https://github.com/user-attachments/assets/700220eb-9173-4a30-88bd-0b17a19bd4ae" />[twitter](https://twitter.com/kurtqian)
 OR <img width="32" height="32" alt="image" src="https://github.com/user-attachments/assets/d5ca4143-7faa-43ed-a061-15e3e1ea68e0" />Weixin：GalaticFolk

# Synopsis:
you should have an Android SDK and ndk on your local OS to build llama.cpp with OpenMP (Open Multi-Processing) support.For instance,i installed a ndk version on /home/user/Android/Sdk/ndk/29.0.13599879
at first,you should `git clone https://github.com/ggml-org/llama.cpp.git` to make a copy of llama.cpp project under your local disk, then mv build-android.sh under the root directory of llama.cpp, just edit the ANDROID_NDK path variable of this bash script which corresponding to your specific NDK pathname,the purpose is according to llama.cpp android Notes:While later versions of Android NDK ship with OpenMP, it must still be installed by CMake as a dependency.We should include the libomp.so in our own project later.

After running `bash build-android.sh` successfully, you should have a compiled build-android-arm64-v8a directory under the root of llama.cpp which we will in use later.In this project,you can find all the build-android-arm64-v8a/lib/*.so files are under app/src/main/jniLibs/arm64-v8a directory,and all the c++ API header files of build-android-arm64-v8a/include/*.h are under app/src/main/cpp/include directory.

Righ now i download the ggml-org/SmolVLM2-256M-Video-Instruct-GGUF models from https://huggingface.co/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF/tree/main,which located app/src/main/assets/models,you should download both the SmolVLM2-256M-Video-Instruct-Q8_0.gguf and mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf,if you don't know the model inference works well, we can give it a shot thr the android adb shell command line execution which i will talk about in the appendix.

you can run this project in Android Studio,it will be automatically build when you click the Run 'app' button,i've already tested this project both on low end phone UMIX V1 and Google Pixel7 as below,currently i've set the maximum response token length to only 128 in app/src/main/cpp/native-lib.cpp,you can tweak it with your own desire:

UMIX V1 snapshot:
![img_v3_02s6_b4b7e9de-0938-4600-8164-5caa6293f07g](https://github.com/user-attachments/assets/ad12c248-cf5c-4500-bef4-9a5dc6379eff)

pixel7 snapshot:
![img_v3_02s6_f85e40ba-d03e-43e2-ab27-935616c17b3g](https://github.com/user-attachments/assets/74458a36-efc6-4093-b3ca-4e6b972364cc)


the more advanced configuration of you hardware,the inference latency will be improved more significantly,likewise of my test,the Pixel7 output token will be in less than 5 seconds compare to UMIX V1 which will be extending to around 14 seconds.

# Appendix:
to make the cmd work out on your android device,you should at first `adb shell` connect to you android device,then `mkdir -p /data/local/tmp/llama.cpp` create a remote working directory, then you can run `adb push source /data/local/tmp/llama.cpp` upload all of the local files on android device(e.g. you can upload all the previously compiled files located build-android-arm64-v8a to `/data/local/tmp/llama.cpp`)when you connect your phone to your local OS through USB port,then `adb shell`,run the following cmd under /data/local/tmp/llama.cpp:`LD_LIBRARY_PATH=/data/local/tmp/llama.cpp/build-android-arm64-v8a/lib /data/local/tmp/llama.cpp/build-android-arm64-v8a/bin/llama-mtmd-cli -m /data/local/tmp/llama.cpp/SmolVLM2-256M-Video-Instruct-Q8_0.gguf --mmproj /data/local/tmp/llama.cpp/mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf --image /data/local/tmp/llama.cpp/test.jpg -c 8192 -p "describe this image"`
