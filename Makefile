CXX := g++
NVCC := nvcc
CXXFLAGS := -std=c++17 -Wall -Wextra -O3 -I./includes
NVCCFLAGS := -std=c++17 -O3 -I./includes
LDFLAGS := -lcudart -lcublas

# ICU path for Homebrew on Apple Silicon
ICU_PATH := /opt/homebrew/opt/icu4c

# Add ICU to CXXFLAGS and LDFLAGS
CXXFLAGS += -I$(ICU_PATH)/include
LDFLAGS += -L$(ICU_PATH)/lib

SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin
VOCAB_DIR := $(BIN_DIR)/vocab
TEST_DIR := tests

CPP_SRCS := $(wildcard $(SRC_DIR)/tokenizer/*.cpp) $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS := $(wildcard $(SRC_DIR)/model/*.cu)
CPP_OBJS := $(CPP_SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
CU_OBJS := $(CU_SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
OBJS := $(CPP_OBJS) $(CU_OBJS)
DEPS := $(OBJS:.o=.d)

TARGET := $(BIN_DIR)/cullm
TEST_TARGET := $(BIN_DIR)/run_tests
VOCAB_FILE := o200k_base.tiktoken

.PHONY: all clean run tests

all: $(TARGET) $(VOCAB_DIR)/$(VOCAB_FILE)

$(TARGET): $(OBJS) | $(BIN_DIR)
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -MMD -MP -c $< -o $@

$(VOCAB_DIR)/$(VOCAB_FILE): $(SRC_DIR)/tokenizer/vocab/$(VOCAB_FILE) | $(VOCAB_DIR)
	cp $< $@

$(BIN_DIR) $(OBJ_DIR) $(VOCAB_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

run: $(TARGET) $(VOCAB_DIR)/$(VOCAB_FILE)
	cd $(BIN_DIR) && ./cullm

tests: $(TEST_TARGET) $(VOCAB_DIR)/$(VOCAB_FILE)
	cd $(BIN_DIR) && ./run_tests

-include $(DEPS)