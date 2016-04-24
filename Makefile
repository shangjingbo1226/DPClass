export CXX = g++
export CFLAGS = -std=c++11 -Wall -O3 -msse2  -fopenmp  -I..

BIN = ./bin/produce_candidate_patterns ./bin/merge_patterns ./bin/rebuild_features ./bin/predict_by_rf ./bin/predict_by_svm ./bin/select_patterns_by_GLMNET ./bin/predict_by_glmnet
.PHONY: clean all

all: ./bin $(BIN)

./bin/produce_candidate_patterns: src/patterns/produce_candidate_patterns.cpp src/classification/*.h src/patterns/*.h src/utils/*.h
./bin/merge_patterns: src/patterns/merge_patterns.cpp src/classification/*.h src/patterns/*.h src/utils/*.h
./bin/select_patterns_by_GLMNET: src/patterns/select_patterns_by_GLMNET.cpp src/patterns/*.h src/utils/*.h
./bin/rebuild_features: src/classification/rebuild_features.cpp src/classification/*.h src/patterns/*.h src/utils/*.h
./bin/predict_by_rf: src/classification/predict_by_rf.cpp src/classification/*.h src/patterns/*.h src/utils/*.h
./bin/predict_by_svm: src/classification/predict_by_svm.cpp src/classification/*.h src/patterns/*.h src/utils/*.h
./bin/predict_by_glmnet: src/classification/predict_by_glmnet.cpp src/patterns/*.h src/utils/*.h

./bin:
	mkdir bin

export LDFLAGS= -pthread -lm -Wno-unused-result -Wno-sign-compare -Wno-unused-variable -Wno-parentheses -Wno-format

$(BIN) :
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)
$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

clean :
	rm -rf bin
