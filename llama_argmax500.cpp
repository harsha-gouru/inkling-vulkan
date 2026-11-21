#include "llama.h"
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <model.gguf> <fa:0|1> <ngl:0|99> <tokens.txt>\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    int fa = atoi(argv[2]);
    int ngl = atoi(argv[3]);
    const char * tok_path = argv[4];

    std::vector<llama_token> toks;
    std::ifstream fin(tok_path);
    int t;
    while (fin >> t) toks.push_back(t);
    int n = (int)toks.size();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    ggml_backend_dev_t cpu_only[1] = { nullptr };
    if (ngl == 0) {
        mp.devices = cpu_only;
    }
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 2; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n + 64;
    cp.n_batch = n;
    cp.n_ubatch = n;
    cp.n_threads = 16;
    cp.n_threads_batch = 16;
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = fa ? GGML_TYPE_F16 : GGML_TYPE_F32;
    cp.type_v = fa ? GGML_TYPE_F16 : GGML_TYPE_F32;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 3; }

    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int i = 0; i < n; i++) {
        batch.token[i] = toks[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 4;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    for (int i = 0; i < n; i++) {
        const float * logits = llama_get_logits_ith(ctx, i);
        int best = 0;
        float best_v = logits[0], second_v = -1e30f;
        for (int v = 1; v < n_vocab; v++) {
            float x = logits[v];
            if (x > best_v) { second_v = best_v; best_v = x; best = v; }
            else if (x > second_v) { second_v = x; }
        }
        printf("%d %d %.5f %.5f\n", i, best, best_v, best_v - second_v);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
