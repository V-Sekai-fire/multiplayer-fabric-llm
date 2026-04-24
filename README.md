# multiplayer-fabric-llm

Elixir NIF binding for in-process LLM inference, powered by llama.cpp
([turboquant fork](https://github.com/TheTom/llama-cpp-turboquant)) which
adds `turbo2`/`turbo3`/`turbo4` KV cache quantization types.

Mirrors the `LLMModel` / `LLMContext` / `LLMChat` Godot API from
[turboquant-godot](https://github.com/V-Sekai-fire/turboquant-godot).

## Build

```sh
git submodule update --init --recursive
mix deps.get
mix compile          # runs make → priv/llm_nif.so
```

Requires a C++17 compiler. Metal is enabled automatically on macOS.

## Usage

```elixir
{:ok, model} = MultiplayerFabricLLM.load_model("model.gguf", n_gpu_layers: -1)
{:ok, ctx}   = MultiplayerFabricLLM.create_context(model, n_ctx: 8192)

{:ok, reply} = MultiplayerFabricLLM.complete(ctx, [
  %{"role" => "user", "content" => "Hello!"}
])

# Streaming
MultiplayerFabricLLM.stream(ctx, messages, &IO.write/1)
```
