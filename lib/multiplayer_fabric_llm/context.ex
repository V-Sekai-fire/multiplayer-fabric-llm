# SPDX-License-Identifier: MIT
# Copyright (c) 2026-present K. S. Ernest (iFire) Lee

defmodule MultiplayerFabricLLM.Context do
  @moduledoc """
  Inference context wrapping `llama_context*`.

  KV cache types including turboquant types (turbo2/3/4) are configured
  via the Makefile/NIF build — see `thirdparty/llama_cpp`.

  ## Example

      {:ok, model} = MultiplayerFabricLLM.Model.load(path)
      {:ok, ctx}   = MultiplayerFabricLLM.Context.create(model, n_ctx: 32768)
  """

  alias MultiplayerFabricLLM.NIF

  @type t :: reference()
  @type create_opt ::
          {:n_ctx, pos_integer()}
          | {:n_threads, pos_integer()}
          | {:flash_attn, boolean()}

  @doc """
  Create an inference context from a loaded model.

  ## Options
  - `:n_ctx`       — context window size in tokens (default: 32_768)
  - `:n_threads`   — CPU thread count (default: `System.schedulers_online/0`)
  - `:flash_attn`  — enable flash attention (default: true)
  """
  @spec create(MultiplayerFabricLLM.Model.t(), [create_opt()]) ::
          {:ok, t()} | {:error, String.t()}
  def create(model_ref, opts \\ []) do
    n_ctx      = Keyword.get(opts, :n_ctx, 32_768)
    n_threads  = Keyword.get(opts, :n_threads, System.schedulers_online())
    flash_attn = Keyword.get(opts, :flash_attn, true)
    NIF.nif_create_context(model_ref, n_ctx, n_threads, flash_attn)
  end

  @doc "Explicitly free the context. The NIF destructor also frees it on GC."
  @spec free(t()) :: :ok
  def free(ctx_ref), do: NIF.nif_free_context(ctx_ref)
end
