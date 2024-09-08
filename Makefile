CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O3 -I./includes/tokenizer -I./tests
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
TEST_DIR := tests

SRCS := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/tokenizer/*.cpp)
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

TARGET := $(BIN_DIR)/tokenizer
TEST_TARGET := $(BIN_DIR)/run_tests
VOCAB_FILE := o200k_base.tiktoken

.PHONY: all clean run tests

all: $(TARGET) $(VOCAB_DIR)/$(VOCAB_FILE)

$(TARGET): $(OBJS) | $(BIN_DIR)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(TEST_TARGET): $(filter-out $(OBJ_DIR)/main.o, $(OBJS)) $(OBJ_DIR)/tokenizer_test.o | $(BIN_DIR)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/tokenizer_test.o: $(TEST_DIR)/tokenizer/tokenizer_test.hpp | $(OBJ_DIR)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -DTESTING -x c++ -MMD -MP -c $< -o $@

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

tests: $(TEST_TARGET) $(VOCAB_DIR)/$(VOCAB_FILE)
	cd $(BIN_DIR) && ./run_tests

-include $(DEPS)