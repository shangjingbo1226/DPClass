#include "../utils/helper.h"
#include "../utils/union_find_set.h"
#include "../patterns/classification_rule.h"
#include "sparse_svm.h"

#include <omp.h>

int NTHREADS = 2;

int loadFeatureMatrixNoHeader(string filename, vector<vector<pair<int, double>>> &train, vector<int> &trainY)
{
	FILE* in = tryOpen(filename.c_str(), "r");
	getLine(in);
	int dimension = 0;
	fprintf(stderr, "# features = %d\n", dimension);
	for (;getLine(in);) {
		vector<string> tokens = splitBy(line, ',');
		if (dimension == 0) {
            dimension = (int)tokens.size() - 1;
		}
		vector<pair<int, double>> features;
		int ptr = 0;
		int label;
		fromString(tokens[0], label);
		for (size_t i = 1; i < tokens.size(); ++ i) {
            double coef;
			fromString(tokens[i], coef);
			if (coef > 0) {
                features.push_back(make_pair(ptr, coef));
            }
            ptr ++;
		}
		myAssert(ptr == dimension, "ptr exceeds the dimension");

        train.push_back(features);
        trainY.push_back(label == 1 ? 1 : -1);
	}
	fclose(in);
	fprintf(stderr, "# training instances = %d\n", train.size());
    return dimension;
}

vector<vector<pair<int, double>>> trainX, testX;
vector<int> trainY, testY;

int main(int argc, char* argv[])
{
    int LARGE_ROUNDS = 50;
    double LAMBDA = 1.0;
	if (argc != 6 || sscanf(argv[3], "%d", &NTHREADS) != 1 || sscanf(argv[4], "%d", &LARGE_ROUNDS) != 1 || sscanf(argv[5], "%lf", &LAMBDA) != 1) {
		fprintf(stderr, "[usage] <training data, csv format> <testing data, csv format> <nthread> <large_round>\n");
		return -1;
	}
	cerr << "# threads = " << NTHREADS << endl;
	omp_set_num_threads(NTHREADS);

    int dimension = loadFeatureMatrixNoHeader(argv[1], trainX, trainY);
	int dimension2 = loadFeatureMatrixNoHeader(argv[2], testX, testY);

    double avg = 0;
    vector<double> globalWeight(dimension, 0), cnt(dimension, 0);
    for (int i = 0; i < trainY.size(); ++ i) {
        avg += trainY[i];
        FOR (iter, trainX[i]) {
            globalWeight[iter->first] += iter->second * trainY[i];
            cnt[iter->first] += iter->second;
        }
    }
    avg /= trainY.size();
    vector<double> w;
    vector<vector<pair<int, double>>> x(trainX.size());
    for (int i = 0; i < dimension; ++ i) {
        for (int j = 0; j < trainX.size(); ++ j) {
            FOR (iter, trainX[j]) {
                if (iter->first == i) {
                    x[j].push_back(*iter);
                    break;
                }
            }
        }
        double init = globalWeight[i] / cnt[i];
        w.push_back(init);
        double acc = SparseSVM::train(x, trainY, w, avg, LARGE_ROUNDS, LAMBDA);
        //cerr << "Train Acc = " << acc << " " << w.size() << " " << cnt[i] << " " << init << endl;
    }
    double testAcc = SparseSVM::calcAccuracy(testX, testY, w, avg);
    cerr << "Test Acc = " << testAcc << endl;

    if (true) {
        vector<int> pred = SparseSVM::getPrediction(testX, testY, w, avg);
        string filename = (string)argv[2] + ".prediction_greedy";
        FILE* out = tryOpen(filename, "w");
        for (int i = 0; i < pred.size(); ++ i) {
            fprintf(out, "%d\n", pred[i]);
        }
        fclose(out);
    }

	return 0;
}

