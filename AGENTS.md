# AGENTS.md — multiplayer-fabric-llm

Guidance for AI coding agents working in this submodule.

## What this is

Elixir NIF wrapping llama.cpp (turboquant fork). Provides in-process LLM
inference for zone servers and bots without a separate HTTP sidecar.
API mirrors the `LLMModel` / `LLMContext` / `LLMChat` interface from the
Godot `modules/llm` module.

## Build

```sh
mix compile          # compiles the C NIF + llama.cpp (takes several minutes)
mix test
```

GPU backends are auto-detected at compile time:
- **Metal** — macOS / iOS (no extra setup)
- **Vulkan** — set `VULKAN_SDK` before building; run `make shaders` first
- **CPU** — fallback, always available

The compiled NIF lands at `priv/llm_nif.so`.

## Key files

| Path | Purpose |
|------|---------|
| `mix.exs` | Uses `elixir_make` compiler to invoke `Makefile` |
| `Makefile` | Compiles ggml + llama.cpp + NIF; mirrors `SCsub` from turboquant-godot |
| `c_src/llm_nif.c` | Erlang NIF entry points |
| `lib/multiplayer_fabric_llm.ex` | Public API: `load_model`, `create_context`, `complete`, `stream` |
| `thirdparty/llama_cpp/` | Vendored turboquant fork of llama.cpp |

## Conventions

- Do not edit files under `thirdparty/` — patch via the turboquant upstream.
- GGUF model files are not stored in this repo; pass the path at runtime.
- Every new `.ex` / `.exs` file needs SPDX headers:
  ```elixir
  # SPDX-License-Identifier: MIT
  # Copyright (c) 2026 K. S. Ernest (iFire) Lee
  ```
- Commit message style: sentence case, no `type(scope):` prefix.
  Example: `Wire streaming token callback to BEAM process mailbox`
