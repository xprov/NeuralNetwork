CC=g++
CFLAGS=-c -Wall -std=c++11
LDFLAGS=
BIN_DIR=./build/
OBJ_DIR=./build/objects/
SRC_DIR=./src/
MKDIR_P=mkdir -p

PROG_NAMES=evalNN trainBPN
PROG_LIST=$(addprefix $(BIN_DIR)/, $(PROG_NAMES))
SRC_LIST=$(wildcard $(SRC_DIR)/*.cpp)
OBJ_LIST=$(addprefix $(OBJ_DIR)/, $(notdir $(SRC_LIST:.cpp=.o)))
MAIN_LIST=$(addsuffix .o, $(addprefix $(OBJ_DIR)/, $(PROG_NAMES)))




.PHONY: all mkdirs clean

all: directories $(PROG_LIST)

directories: $(BIN_DIR) $(OBJ_DIR)

$(BIN_DIR):
	$(MKDIR_P) $(BIN_DIR)

$(OBJ_DIR):
	$(MKDIR_P) $(OBJ_DIR)

$(BIN_DIR)/evalNN: $(OBJ_LIST)
	$(CC) $(LDFLAGS) $(OBJ_DIR)/evalNN.o $(filter-out $(MAIN_LIST), $(OBJ_LIST)) -o $@

$(BIN_DIR)/trainBPN: $(OBJ_LIST)
	$(CC) $(LDFLAGS) $(OBJ_DIR)/trainBPN.o $(filter-out $(MAIN_LIST), $(OBJ_LIST)) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJ_DIR)/*.o $(addprefix $(BIN_DIR)/, $(PROG_NAMES))
