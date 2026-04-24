# SPDX-License-Identifier: MIT
# Copyright (c) 2026-present K. S. Ernest (iFire) Lee

defmodule MultiplayerFabricLLM do
  @moduledoc """
  Elixir NIF binding for in-process LLM inference via llama.cpp
  (turboquant fork — adds turbo2/3/4 KV cache quantization and
  Metal/CUDA GPU offload).

  Mirrors the `LLMModel` / `LLMContext` / `LLMChat` API from the
  Godot `modules/llm` module in turboquant-godot.

  ## Quick start

      # 1. Load a GGUF model (Metal GPU acceleration on macOS)
      {:ok, model} = MultiplayerFabricLLM.load_model(
        "/path/to/Qwen3-0.6B-Q4_K_M.gguf"
      )

      # 2. Create an inference context
      {:ok, ctx} = MultiplayerFabricLLM.create_context(model, n_ctx: 8192)

      # 3. Chat (blocking)
      {:ok, reply} = MultiplayerFabricLLM.complete(ctx, [
        %{"role" => "user", "content" => "What is 2 + 2?"}
      ])

      # 4. Streaming
      MultiplayerFabricLLM.stream(ctx, messages, &IO.write/1)
  """

  alias MultiplayerFabricLLM.{Chat, Context, Model}

  defdelegate load_model(path, opts \\ []), to: Model, as: :load
  defdelegate free_model(ref), to: Model, as: :free
  defdelegate create_context(model_ref, opts \\ []), to: Context, as: :create
  defdelegate free_context(ref), to: Context, as: :free
  defdelegate complete(ctx_ref, messages, opts \\ []), to: Chat
  defdelegate stream(ctx_ref, messages, on_token, opts \\ []), to: Chat
end
