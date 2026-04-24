// SPDX-License-Identifier: MIT
// Copyright (c) 2026-present K. S. Ernest (iFire) Lee
//
// Elixir NIF wrapping llama.cpp (turboquant fork).
// Mirrors the LLMModel / LLMContext / LLMChat API from the Godot module.
//
// Exposed functions (all scheduled on dirty CPU scheduler):
//   llm_load_model(path, n_gpu_layers)  -> {:ok, ref} | {:error, reason}
//   llm_free_model(ref)                 -> :ok
//   llm_create_context(model_ref, n_ctx, n_threads, flash_attn) -> {:ok, ref} | {:error, reason}
//   llm_free_context(ref)               -> :ok
//   llm_complete(ctx_ref, prompt, max_tokens, temperature, caller_pid)
//       -> streams {:llm_token, ref, token_binary} to caller_pid
//       -> returns {:ok, full_text} | {:error, reason}

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "erl_nif.h"
#include "llama.h"

// ── Resource types ─────────────────────────────────────────────────────────

typedef struct { llama_model *model; } ModelRes;
typedef struct { llama_context *ctx; } CtxRes;

static ErlNifResourceType *MODEL_RES_TYPE = NULL;
static ErlNifResourceType *CTX_RES_TYPE   = NULL;

static void model_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    ModelRes *r = (ModelRes *)obj;
    if (r->model) { llama_model_free(r->model); r->model = NULL; }
}

static void ctx_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    CtxRes *r = (CtxRes *)obj;
    if (r->ctx) { llama_free(r->ctx); r->ctx = NULL; }
}

// ── Helpers ────────────────────────────────────────────────────────────────

static ERL_NIF_TERM make_error(ErlNifEnv *env, const char *msg) {
    return enif_make_tuple2(env,
        enif_make_atom(env, "error"),
        enif_make_string(env, msg, ERL_NIF_UTF8));
}

static ERL_NIF_TERM make_ok(ErlNifEnv *env, ERL_NIF_TERM val) {
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), val);
}

// ── NIF: llm_load_model/2 ──────────────────────────────────────────────────

static ERL_NIF_TERM llm_load_model(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 2) return enif_make_badarg(env);

    // path
    unsigned path_len;
    if (!enif_get_list_length(env, argv[0], &path_len) && !enif_is_binary(env, argv[0]))
        return enif_make_badarg(env);

    ErlNifBinary path_bin;
    char *path = NULL;
    if (enif_inspect_binary(env, argv[0], &path_bin)) {
        path = (char *)malloc(path_bin.size + 1);
        memcpy(path, path_bin.data, path_bin.size);
        path[path_bin.size] = '\0';
    } else {
        return enif_make_badarg(env);
    }

    int n_gpu_layers;
    if (!enif_get_int(env, argv[1], &n_gpu_layers)) { free(path); return enif_make_badarg(env); }

    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;

    llama_model *model = llama_model_load_from_file(path, params);
    free(path);

    if (!model) return make_error(env, "failed to load model");

    ModelRes *r = (ModelRes *)enif_alloc_resource(MODEL_RES_TYPE, sizeof(ModelRes));
    r->model = model;
    ERL_NIF_TERM ref = enif_make_resource(env, r);
    enif_release_resource(r);
    return make_ok(env, ref);
}

// ── NIF: llm_free_model/1 ─────────────────────────────────────────────────

static ERL_NIF_TERM llm_free_model(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 1) return enif_make_badarg(env);
    ModelRes *r;
    if (!enif_get_resource(env, argv[0], MODEL_RES_TYPE, (void **)&r))
        return enif_make_badarg(env);
    if (r->model) { llama_model_free(r->model); r->model = NULL; }
    return enif_make_atom(env, "ok");
}

// ── NIF: llm_create_context/4 ─────────────────────────────────────────────

static ERL_NIF_TERM llm_create_context(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 4) return enif_make_badarg(env);

    ModelRes *mr;
    if (!enif_get_resource(env, argv[0], MODEL_RES_TYPE, (void **)&mr) || !mr->model)
        return make_error(env, "invalid model ref");

    int n_ctx, n_threads;
    if (!enif_get_int(env, argv[1], &n_ctx)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[2], &n_threads)) return enif_make_badarg(env);

    // argv[3] = flash_attn bool atom
    char flash_attn_atom[8];
    int flash_attn = 0;
    if (enif_get_atom(env, argv[3], flash_attn_atom, sizeof(flash_attn_atom), ERL_NIF_UTF8))
        flash_attn = (strcmp(flash_attn_atom, "true") == 0);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = (uint32_t)n_ctx;
    cparams.n_threads   = n_threads;
    cparams.flash_attn  = flash_attn;

    llama_context *ctx = llama_new_context_with_model(mr->model, cparams);
    if (!ctx) return make_error(env, "failed to create context");

    CtxRes *r = (CtxRes *)enif_alloc_resource(CTX_RES_TYPE, sizeof(CtxRes));
    r->ctx = ctx;
    ERL_NIF_TERM ref = enif_make_resource(env, r);
    enif_release_resource(r);
    return make_ok(env, ref);
}

// ── NIF: llm_free_context/1 ───────────────────────────────────────────────

static ERL_NIF_TERM llm_free_context(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 1) return enif_make_badarg(env);
    CtxRes *r;
    if (!enif_get_resource(env, argv[0], CTX_RES_TYPE, (void **)&r))
        return enif_make_badarg(env);
    if (r->ctx) { llama_free(r->ctx); r->ctx = NULL; }
    return enif_make_atom(env, "ok");
}

// ── NIF: llm_complete/5 (dirty CPU) ───────────────────────────────────────
// argv: ctx_ref, prompt_binary, max_tokens, temperature, caller_pid
// Streams {:llm_token, ctx_ref, token_binary} to caller_pid.
// Returns {:ok, full_text} | {:error, reason}.

static ERL_NIF_TERM llm_complete(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 5) return enif_make_badarg(env);

    CtxRes *cr;
    if (!enif_get_resource(env, argv[0], CTX_RES_TYPE, (void **)&cr) || !cr->ctx)
        return make_error(env, "invalid context ref");

    ErlNifBinary prompt_bin;
    if (!enif_inspect_binary(env, argv[1], &prompt_bin))
        return enif_make_badarg(env);

    int max_tokens;
    double temperature;
    if (!enif_get_int(env, argv[2], &max_tokens)) return enif_make_badarg(env);
    if (!enif_get_double(env, argv[3], &temperature)) {
        int tmp;
        if (!enif_get_int(env, argv[3], &tmp)) return enif_make_badarg(env);
        temperature = (double)tmp;
    }

    ErlNifPid caller;
    if (!enif_get_local_pid(env, argv[4], &caller)) return enif_make_badarg(env);

    llama_context *ctx = cr->ctx;
    llama_model   *model = llama_get_model(ctx);

    // Tokenize prompt
    char *prompt = (char *)malloc(prompt_bin.size + 1);
    memcpy(prompt, prompt_bin.data, prompt_bin.size);
    prompt[prompt_bin.size] = '\0';

    int n_prompt_tokens = -llama_tokenize(model, prompt, (int32_t)prompt_bin.size,
                                          NULL, 0, /*add_bos=*/1, /*special=*/1);
    llama_token *tokens = (llama_token *)malloc(n_prompt_tokens * sizeof(llama_token));
    llama_tokenize(model, prompt, (int32_t)prompt_bin.size,
                   tokens, n_prompt_tokens, /*add_bos=*/1, /*special=*/1);
    free(prompt);

    llama_batch batch = llama_batch_init(n_prompt_tokens, 0, 1);
    for (int i = 0; i < n_prompt_tokens; i++) {
        llama_batch_add(&batch, tokens[i], i, (llama_seq_id[]){0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;
    free(tokens);

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        return make_error(env, "llama_decode failed on prompt");
    }
    llama_batch_free(batch);

    // Sample loop
    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // Accumulate full response for return value
    size_t full_cap  = 4096;
    size_t full_len  = 0;
    char  *full_text = (char *)malloc(full_cap);
    full_text[0] = '\0';

    int n_cur = n_prompt_tokens;
    ErlNifEnv *msg_env = enif_alloc_env();

    for (int i = 0; i < max_tokens; i++) {
        llama_token tok = llama_sampler_sample(sampler, ctx, -1);
        if (llama_token_is_eog(model, tok)) break;

        char piece[128];
        int  piece_len = llama_token_to_piece(model, tok, piece, sizeof(piece) - 1, 0, false);
        if (piece_len < 0) break;
        piece[piece_len] = '\0';

        // Grow buffer if needed
        while (full_len + (size_t)piece_len + 1 > full_cap) {
            full_cap *= 2;
            full_text = (char *)realloc(full_text, full_cap);
        }
        memcpy(full_text + full_len, piece, piece_len);
        full_len += piece_len;
        full_text[full_len] = '\0';

        // Send token to caller
        enif_clear_env(msg_env);
        ERL_NIF_TERM tok_bin;
        unsigned char *tok_data = enif_make_new_binary(msg_env, piece_len, &tok_bin);
        memcpy(tok_data, piece, piece_len);
        ERL_NIF_TERM msg = enif_make_tuple3(msg_env,
            enif_make_atom(msg_env, "llm_token"),
            enif_make_copy(msg_env, argv[0]),  // ctx_ref
            tok_bin);
        enif_send(env, &caller, msg_env, msg);

        // Decode next token
        llama_batch next = llama_batch_init(1, 0, 1);
        llama_batch_add(&next, tok, n_cur++, (llama_seq_id[]){0}, true);
        if (llama_decode(ctx, next) != 0) {
            llama_batch_free(next);
            break;
        }
        llama_batch_free(next);
    }

    enif_free_env(msg_env);
    llama_sampler_free(sampler);

    ERL_NIF_TERM result_bin;
    unsigned char *out = enif_make_new_binary(env, full_len, &result_bin);
    memcpy(out, full_text, full_len);
    free(full_text);

    return make_ok(env, result_bin);
}

// ── NIF init ──────────────────────────────────────────────────────────────

static int load(ErlNifEnv *env, void **priv, ERL_NIF_TERM load_info) {
    (void)priv; (void)load_info;

    MODEL_RES_TYPE = enif_open_resource_type(env, NULL, "llm_model",
        model_dtor, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);
    CTX_RES_TYPE = enif_open_resource_type(env, NULL, "llm_context",
        ctx_dtor, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);

    if (!MODEL_RES_TYPE || !CTX_RES_TYPE) return -1;

    llama_backend_init();
    return 0;
}

static void unload(ErlNifEnv *env, void *priv) {
    (void)env; (void)priv;
    llama_backend_free();
}

static ErlNifFunc nif_funcs[] = {
    {"nif_load_model",    2, llm_load_model,    ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_free_model",    1, llm_free_model,    0},
    {"nif_create_context",4, llm_create_context,ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_free_context",  1, llm_free_context,  0},
    {"nif_complete",      5, llm_complete,      ERL_NIF_DIRTY_JOB_CPU_BOUND},
};

ERL_NIF_INIT(Elixir.MultiplayerFabricLLM.NIF, nif_funcs, load, NULL, NULL, unload)
