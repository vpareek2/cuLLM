CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O3 -I./includes
LDFLAGS := -licuuc -licui18n

# ICU path for Homebrew on Apple Silicon
ICU_PATH := /opt/homebrew/opt/icu4c

# Add ICU to CXXFLAGS and LDFLAGS
CXXFLAGS += -I$(ICU_PATH)/include
LDFLAGS += -L$(ICU_PATH)/lib

SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin
VOCAB_DIR := $(BIN_DIR)/vocab

SRCS := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/tokenizer/*.cpp)
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

TARGET := $(BIN_DIR)/tokenizer
VOCAB_FILE := o200k_base.tiktoken

.PHONY: all clean run

all: $(TARGET) $(VOCAB_DIR)/$(VOCAB_FILE)

$(TARGET): $(OBJS) | $(BIN_DIR)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(OBJ_DIR)/tokenizer/%.o: $(SRC_DIR)/tokenizer/%.cpp | $(OBJ_DIR)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(VOCAB_DIR)/$(VOCAB_FILE): $(SRC_DIR)/tokenizer/vocab/$(VOCAB_FILE) | $(VOCAB_DIR)
	cp $< $@

$(BIN_DIR) $(OBJ_DIR) $(VOCAB_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

run: $(TARGET) $(VOCAB_DIR)/$(VOCAB_FILE)
	cd $(BIN_DIR) && ./tokenizer

-include $(DEPS)