# SPDX-License-Identifier: MIT
# Copyright (c) 2026-present K. S. Ernest (iFire) Lee

defmodule MultiplayerFabricLLM.Chat do
  @moduledoc """
  High-level chat completion using in-process llama.cpp inference.

  Mirrors the GDScript `LLMChat` API. Runs inference on a dirty CPU
  scheduler; streams tokens back via messages to the calling process.

  ## Synchronous example

      {:ok, text} = MultiplayerFabricLLM.Chat.complete(ctx, [
        %{"role" => "user", "content" => "Hello!"}
      ])

  ## Streaming example

      MultiplayerFabricLLM.Chat.stream(ctx, messages, fn token -> IO.write(token) end)
  """

  alias MultiplayerFabricLLM.NIF

  @type message :: %{String.t() => String.t()}
  @type complete_opt ::
          {:max_tokens, pos_integer()}
          | {:temperature, float()}

  @doc """
  Synchronous chat completion. Blocks until the full response is ready.

  Tokens are still streamed internally; this function collects them and
  returns `{:ok, full_text}`.
  """
  @spec complete(MultiplayerFabricLLM.Context.t(), [message()], [complete_opt()]) ::
          {:ok, String.t()} | {:error, String.t()}
  def complete(ctx_ref, messages, opts \\ []) do
    max_tokens  = Keyword.get(opts, :max_tokens, 1024)
    temperature = Keyword.get(opts, :temperature, 0.7)
    prompt      = apply_chat_template(messages)

    NIF.nif_complete(ctx_ref, prompt, max_tokens, temperature, self())
    |> collect_tokens()
  end

  @doc """
  Streaming chat completion. Calls `on_token.(token_string)` for each token,
  then returns `{:ok, full_text}`.
  """
  @spec stream(
          MultiplayerFabricLLM.Context.t(),
          [message()],
          (String.t() -> any()),
          [complete_opt()]
        ) :: {:ok, String.t()} | {:error, String.t()}
  def stream(ctx_ref, messages, on_token, opts \\ []) do
    max_tokens  = Keyword.get(opts, :max_tokens, 1024)
    temperature = Keyword.get(opts, :temperature, 0.7)
    prompt      = apply_chat_template(messages)

    task = Task.async(fn ->
      NIF.nif_complete(ctx_ref, prompt, max_tokens, temperature, self())
    end)

    collect_tokens_streaming(ctx_ref, task, on_token)
  end

  # ── Private ────────────────────────────────────────────────────────────────

  # Minimal ChatML template matching llama.cpp's default.
  defp apply_chat_template(messages) do
    Enum.map_join(messages, "", fn %{"role" => role, "content" => content} ->
      "<|im_start|>#{role}\n#{content}<|im_end|>\n"
    end) <> "<|im_start|>assistant\n"
  end

  # Collect {:llm_token, _ref, token} messages until the NIF call returns.
  defp collect_tokens(nif_result) do
    case nif_result do
      {:ok, full_text} -> {:ok, full_text}
      {:error, _} = err -> err
    end
  end

  defp collect_tokens_streaming(ctx_ref, task, on_token) do
    receive do
      {:llm_token, ^ctx_ref, token} ->
        on_token.(token)
        collect_tokens_streaming(ctx_ref, task, on_token)
    after
      0 ->
        case Task.yield(task, 0) do
          {:ok, result} -> collect_tokens(result)
          nil -> collect_tokens_streaming(ctx_ref, task, on_token)
          {:exit, reason} -> {:error, inspect(reason)}
        end
    end
  end
end
