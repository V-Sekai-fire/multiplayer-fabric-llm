# SPDX-License-Identifier: MIT
# Copyright (c) 2026-present K. S. Ernest (iFire) Lee

defmodule MultiplayerFabricLLM.NIF do
  @moduledoc false

  @on_load :load_nif

  def load_nif do
    nif_path = :filename.join(:code.priv_dir(:multiplayer_fabric_llm), ~c"llm_nif")
    :erlang.load_nif(nif_path, 0)
  end

  def nif_load_model(_path, _n_gpu_layers), do: :erlang.nif_error(:not_loaded)
  def nif_free_model(_ref), do: :erlang.nif_error(:not_loaded)
  def nif_create_context(_model_ref, _n_ctx, _n_threads, _flash_attn), do: :erlang.nif_error(:not_loaded)
  def nif_free_context(_ref), do: :erlang.nif_error(:not_loaded)
  def nif_complete(_ctx_ref, _prompt, _max_tokens, _temperature, _caller_pid), do: :erlang.nif_error(:not_loaded)
end
