PRIV_DIR  = priv
NIF_SO    = $(PRIV_DIR)/llm_nif.so
TP        = thirdparty/llama_cpp

ERL_INC   = $(shell erl -eval 'io:format("~s~n",[code:root_dir()])' -s init stop -noshell)/usr/include
ERTS_INC  = $(shell erl -eval 'io:format("~s~n",[code:lib_dir(erts)])' -s init stop -noshell)/include

CFLAGS   = -O2 -march=native -fPIC -I$(ERL_INC) -I$(ERTS_INC) \
           -I$(TP)/include -I$(TP)/ggml/include \
           -DGGML_USE_METAL=1
CXXFLAGS = $(CFLAGS) -std=c++17
LDFLAGS  = -shared

ifeq ($(shell uname), Darwin)
  NIF_SO   = $(PRIV_DIR)/llm_nif.so
  LDFLAGS += -undefined dynamic_lookup -framework Metal -framework Foundation \
             -framework Accelerate
  METAL_SHADER = $(PRIV_DIR)/ggml-metal.metal
endif

# ── llama.cpp source sets ─────────────────────────────────────────────────────
GGML_C_SRCS = \
  $(TP)/ggml/src/ggml.c \
  $(TP)/ggml/src/ggml-alloc.c \
  $(TP)/ggml/src/ggml-quants.c

GGML_CPP_SRCS = \
  $(TP)/ggml/src/ggml-backend.cpp \
  $(TP)/ggml/src/ggml-backend-reg.cpp \
  $(TP)/ggml/src/ggml-cpu/ggml-cpu.cpp \
  $(TP)/ggml/src/ggml-cpu/ggml-cpu-quants.cpp \
  $(TP)/ggml/src/gguf.cpp

LLAMA_SRCS = \
  $(TP)/src/llama.cpp \
  $(TP)/src/llama-arch.cpp \
  $(TP)/src/llama-batch.cpp \
  $(TP)/src/llama-context.cpp \
  $(TP)/src/llama-grammar.cpp \
  $(TP)/src/llama-kv-cache.cpp \
  $(TP)/src/llama-memory.cpp \
  $(TP)/src/llama-model.cpp \
  $(TP)/src/llama-model-loader.cpp \
  $(TP)/src/llama-mmap.cpp \
  $(TP)/src/llama-sampling.cpp \
  $(TP)/src/llama-vocab.cpp \
  $(TP)/src/unicode.cpp \
  $(TP)/src/unicode-data.cpp

NIF_SRC = c_src/llm_nif.c

OBJS = $(GGML_C_SRCS:.c=.o) \
       $(GGML_CPP_SRCS:.cpp=.o) \
       $(LLAMA_SRCS:.cpp=.o) \
       c_src/llm_nif.o

.PHONY: all clean

all: $(PRIV_DIR) $(NIF_SO)

$(PRIV_DIR):
	mkdir -p $(PRIV_DIR)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

c_src/llm_nif.o: c_src/llm_nif.c
	$(CC) $(CFLAGS) -c $< -o $@

$(NIF_SO): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

clean:
	find $(TP) -name "*.o" -delete
	rm -f c_src/*.o $(NIF_SO)
