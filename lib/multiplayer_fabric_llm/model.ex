# SPDX-License-Identifier: MIT
# Copyright (c) 2026-present K. S. Ernest (iFire) Lee

defmodule MultiplayerFabricLLM.Model do
  @moduledoc """
  Loads a GGUF model file via llama.cpp (turboquant fork).

  GPU acceleration is automatic: Metal on macOS, CUDA on Linux/Windows
  when the NIF was compiled with the respective backend.

  ## Example

      {:ok, model} = MultiplayerFabricLLM.Model.load("/path/to/Qwen3.5-0.8B-Q4_K_M.gguf")
      # or with explicit GPU layer count:
      {:ok, model} = MultiplayerFabricLLM.Model.load(path, n_gpu_layers: -1)
  """

  alias MultiplayerFabricLLM.NIF

  @type t :: reference()
  @type load_opt :: {:n_gpu_layers, integer()}

  @doc """
  Load a GGUF model from `path`.

  ## Options
  - `:n_gpu_layers` — number of layers to offload to GPU. `-1` = all (default).
  """
  @spec load(Path.t(), [load_opt()]) :: {:ok, t()} | {:error, String.t()}
  def load(path, opts \\ []) do
    n_gpu_layers = Keyword.get(opts, :n_gpu_layers, -1)
    NIF.nif_load_model(path_to_binary(path), n_gpu_layers)
  end

  @doc "Explicitly free the model. The NIF destructor also frees it on GC."
  @spec free(t()) :: :ok
  def free(model_ref), do: NIF.nif_free_model(model_ref)

  defp path_to_binary(path) when is_binary(path), do: path
  defp path_to_binary(path), do: Path.expand(path)
end
