#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include <llama.h>
#include <mtmd.h>
#include <mtmd-helper.h>

#define LOG_TAG "LlamaMtmd"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT jint JNICALL
Java_com_example_llamamtmdapp_MainActivity_runInference(
        JNIEnv *env,
        jobject thiz,
        jstring jModelPath,
        jstring jMmprojPath,
        jstring jImagePath,
        jstring jPrompt,
        jint ctxSize) {

    /* --------------------------------------------------------------
    1. Java → C strings
    -------------------------------------------------------------- */
    const char *model_path = env->GetStringUTFChars(jModelPath, nullptr);
    const char *mmproj_path = env->GetStringUTFChars(jMmprojPath, nullptr);
    const char *image_path = env->GetStringUTFChars(jImagePath, nullptr);
    const char *prompt_cstr = env->GetStringUTFChars(jPrompt, nullptr);

    /* --------------------------------------------------------------
    2. JNI helpers for updateOutput method
    -------------------------------------------------------------- */
    jclass cls = env->GetObjectClass(thiz);
    jmethodID updateMid = env->GetMethodID(cls, "updateOutput", "(Ljava/lang/String;)V");
    if (!updateMid) {
        LOGE("Failed to find updateOutput method");
        env->ReleaseStringUTFChars(jModelPath, model_path);
        env->ReleaseStringUTFChars(jMmprojPath, mmproj_path);
        env->ReleaseStringUTFChars(jImagePath, image_path);
        env->ReleaseStringUTFChars(jPrompt, prompt_cstr);
        return -1;
    }

    /* --------------------------------------------------------------
    3. ALL VARIABLES — DECLARED AT TOP TO ALLOW goto cleanup
    -------------------------------------------------------------- */
    const int MAX_GEN = 128;
    const int n_batch = 1;
    llama_pos n_past = 0;
    llama_seq_id seq_id = 0;

    // Resources (must be nullptr-initialized)
    llama_model *lmodel = nullptr;
    llama_context *lctx = nullptr;
    mtmd_context *mtmd = nullptr;
    mtmd_input_chunks *chunks = nullptr;
    mtmd_bitmap *bitmap = nullptr;
    const char *media_marker = nullptr;
    const struct llama_vocab *vocab = nullptr;

    llama_sampler_chain_params sparams = {};
    llama_sampler *sampler_chain = nullptr;

    std::string full_prompt;
    mtmd_context_params mtmd_params = {};
    mtmd_input_text input_text = {};
    const mtmd_bitmap *bitmaps[1] = { nullptr };
    int32_t tok_res = 0;

    std::vector<llama_token> tokens;
    int32_t n_tokens = 0, n_written = 0;

    std::string response;
    bool isFirstToken = true;
    std::string accumulatedPrefix;

    bool success = false;
    /* ==============================================================
    4. Load LLM
    ============================================================== */
    {
        llama_model_params lparams = llama_model_default_params();
        lparams.n_gpu_layers = 0;  // Force CPU to avoid GPU memory leak
        lmodel = llama_model_load_from_file(model_path, lparams);
        if (!lmodel) {
            LOGE("load model failed");
            goto cleanup;
        }
    }

    vocab = llama_model_get_vocab(lmodel);
    if (!vocab) {
        LOGE("failed to get vocab");
        goto cleanup;
    }

    /* ==============================================================
    5. Create context (NEW EVERY TIME!)
    ============================================================== */
    {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = static_cast<uint32_t>(ctxSize);
        cparams.n_batch = n_batch;
        lctx = llama_init_from_model(lmodel, cparams);
        if (!lctx) {
            LOGE("create context failed");
            goto cleanup;
        }
    }

    /* ============================================================== 6. Sampler chain ============================================================== */
    sparams = llama_sampler_chain_default_params();
    sampler_chain = llama_sampler_chain_init(sparams);
    if (!sampler_chain) {
        LOGE("Failed to init sampler chain");
        goto cleanup;
    }

    llama_sampler_chain_add(sampler_chain, llama_sampler_init_top_k(5));
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_penalties(64, 1.2f, 0.1f, 0.1f));
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_temp(0.0f));
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    /* ==============================================================
    7. MTMD (multimodal) path
    ============================================================== */
    media_marker = mtmd_default_marker();  // e.g., "<__media__>"
    full_prompt =
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" +
            std::string(media_marker) + "\n" +          // <image> or whatever marker the mmproj expects
            std::string(prompt_cstr) +
            "<|im_end|>\n"
            "<|im_start|>assistant\n";                  // this triggers generation
//    full_prompt = std::string(" ") + media_marker + "\n" + prompt_cstr;
//    full_prompt =
//            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
//            "<|im_start|>user\n<image>\n" + std::string(prompt_cstr) + "<|im_end|>\n"
//                                                                       "<|im_start|>assistant\n";
    mtmd_params = mtmd_context_params_default();
    mtmd_params.print_timings = false;

    mtmd = mtmd_init_from_file(mmproj_path, lmodel, mtmd_params);
    if (!mtmd) {
        LOGE("mtmd_init_from_file failed – fallback to text-only");
        goto text_only;
    }

    bitmap = mtmd_helper_bitmap_init_from_file(mtmd, image_path);
    if (!bitmap) {
        LOGE("mtmd_helper_bitmap_init_from_file failed – fallback");
        goto text_only;
    }

    chunks = mtmd_input_chunks_init();
    if (!chunks) {
        LOGE("mtmd_input_chunks_init failed");
        goto text_only;
    }

    input_text.text = full_prompt.c_str();
    input_text.add_special = true;
    input_text.parse_special = true;
    bitmaps[0] = bitmap;

    tok_res = mtmd_tokenize(mtmd, chunks, &input_text, bitmaps, 1);
    if (tok_res != 0) {
        LOGE("mtmd_tokenize failed: %d", tok_res);
        goto text_only;
    }

    LOGI("Tokenized into %zu chunks", mtmd_input_chunks_size(chunks));

    n_past = 0;
    seq_id = 0;

    if (mtmd_helper_eval_chunks(mtmd, lctx, chunks, n_past, seq_id, n_batch, true, &n_past)) {
        LOGE("mtmd_helper_eval_chunks failed");
        goto text_only;
    }
    LOGI("Multimodal eval done, n_past=%d", n_past);

    /* --------------------------------------------------------------
    8. Generation loop (NO DUMMY DECODE!)
    -------------------------------------------------------------- */
    for (int i = 0; i < MAX_GEN; ++i) {
        if (n_past >= llama_n_ctx(lctx)) {
            LOGE("Context full");
            break;
        }

        llama_token next = llama_sampler_sample(sampler_chain, lctx, -1);
        llama_sampler_accept(sampler_chain, next);

        if (next == llama_vocab_eos(vocab) || next == llama_vocab_eot(vocab)) {
            LOGI("EOS/EOT reached");
            break;
        }

        char piece[128] = {0};
        int len = llama_token_to_piece(vocab, next, piece, sizeof(piece), 0, true);

        if (len > 0 && len < (int)sizeof(piece)) {
            std::string token_str(piece, len);

            // Filter control tokens
            if (token_str.find("<fake_token_around_image>") != std::string::npos &&
                token_str.find(media_marker) != std::string::npos &&
                token_str.find("<start_of_image>") != std::string::npos &&
                token_str.find("<end_of_image>") != std::string::npos) {
                continue;
            }

            if (isFirstToken) {
                accumulatedPrefix += token_str;
                if (accumulatedPrefix.find("Assistant:") == 0) {
                    if (accumulatedPrefix == "Assistant:") {
                        accumulatedPrefix.clear();
                        isFirstToken = false;
                        continue;
                    }
                } else {
                    jstring jToken = env->NewStringUTF(accumulatedPrefix.c_str());
                    env->CallVoidMethod(thiz, updateMid, jToken);
                    env->DeleteLocalRef(jToken);
                    accumulatedPrefix.clear();
                    isFirstToken = false;
                }
            } else {
                jstring jToken = env->NewStringUTF(token_str.c_str());
                env->CallVoidMethod(thiz, updateMid, jToken);
                env->DeleteLocalRef(jToken);
                response += token_str;
            }
        }

        // Decode next token
        llama_batch batch = llama_batch_init(1, 0, 1);
        if (!batch.token) {
            LOGE("batch alloc failed in loop");
            break;
        }

        batch.n_tokens = 1;
        batch.token[0] = next;
        batch.pos[0] = n_past++;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = seq_id;
        batch.logits[0] = 1;

        int decode_res = llama_decode(lctx, batch);
        llama_batch_free(batch);

        if (decode_res != 0) {
            LOGE("llama_decode failed: %d", decode_res);
            break;
        }
    }

    success = true;
    goto cleanup;  // Changed: No extra append at 'success'

/* ==============================================================
    10. Text-only fallback
    ============================================================== */
    text_only:
    LOGI("Running text-only fallback");

    n_tokens = llama_tokenize(vocab, prompt_cstr, -1, nullptr, 0, true, false);
    if (n_tokens <= 0) goto cleanup;

    tokens.resize(n_tokens);
    n_written = llama_tokenize(vocab, prompt_cstr, -1, tokens.data(), n_tokens, true, false);
    if (n_written <= 0) goto cleanup;
    tokens.resize(n_written);

    {
        llama_batch batch = llama_batch_get_one(tokens.data(), n_written);
        for (int i = 0; i < n_written; ++i) {
            batch.logits[i] = (i == n_written - 1) ? 1 : 0;
        }
        if (llama_decode(lctx, batch) != 0) {
            LOGE("text-only eval failed");
            goto cleanup;
        }
        n_past = n_written;
    }

    isFirstToken = true;
    accumulatedPrefix.clear();

    for (int i = 0; i < MAX_GEN; ++i) {
        llama_token next = llama_sampler_sample(sampler_chain, lctx, -1);
        llama_sampler_accept(sampler_chain, next);

        if (next == llama_vocab_eos(vocab) || next == llama_vocab_eot(vocab)) break;

        char piece[128] = {0};
        int len = llama_token_to_piece(vocab, next, piece, sizeof(piece), 0, true);

        if (len > 0 && len < (int)sizeof(piece)) {
            std::string token_str(piece, len);

            if (token_str.find(media_marker) != std::string::npos) continue;

            if (isFirstToken && token_str.find("Assistant:") == 0) {
                if (token_str == "Assistant:") {
                    isFirstToken = false;
                    continue;
                }
            }

            jstring jToken = env->NewStringUTF(token_str.c_str());
            env->CallVoidMethod(thiz, updateMid, jToken);
            env->DeleteLocalRef(jToken);  // CRITICAL
            response += token_str;
        }

        llama_batch batch = llama_batch_init(1, 0, 1);
        if (!batch.token) break;

        batch.n_tokens = 1;
        batch.token[0] = next;
        batch.pos[0] = n_past++;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = seq_id;
        batch.logits[0] = 1;

        if (llama_decode(lctx, batch) != 0) {
            llama_batch_free(batch);
            break;
        }
        llama_batch_free(batch);
    }

    cleanup:
    if (sampler_chain) { llama_sampler_free(sampler_chain); sampler_chain = nullptr; }
    if (chunks) { mtmd_input_chunks_free(chunks); chunks = nullptr; }
    if (bitmap) { mtmd_bitmap_free(bitmap); bitmap = nullptr; }
    if (mtmd) { mtmd_free(mtmd); mtmd = nullptr; }
    if (lctx) { llama_free(lctx); lctx = nullptr;}
    if (lmodel) { llama_model_free(lmodel); lmodel = nullptr; }

    // Always release Java strings
    if (model_path) env->ReleaseStringUTFChars(jModelPath, model_path);
    if (mmproj_path) env->ReleaseStringUTFChars(jMmprojPath, mmproj_path);
    if (image_path) env->ReleaseStringUTFChars(jImagePath, image_path);
    if (prompt_cstr) env->ReleaseStringUTFChars(jPrompt, prompt_cstr);

    env->DeleteLocalRef(cls);

    return success ? 0 : -1;
}