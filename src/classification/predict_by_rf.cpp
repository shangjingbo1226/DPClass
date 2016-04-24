#include "random_forest.h"
#include "../utils/helper.h"
#include "../utils/training_matrix.h"

using namespace RandomForestRelated;

vector< vector<double> > trainX, testX;
vector<double> trainY, testY;
vector<string> featureNames;

int main(int argc, char* argv[])
{
    double threshold;
	if (argc != 6) {
		fprintf(stderr, "[usage] <train.csv> <test.csv> <output: prediction.csv> <TRAIN / LOAD> <model-name>\n");
		return -1;
	}

	int dimension2 = TrainingMatrix::loadFeatureMatrix(argv[2], testX, testY);
	featureNames = TrainingMatrix::featureNames;

	RandomForest *solver = new RandomForest();

	if (strcmp(argv[4], "TRAIN") == 0) {
	    int dimension = TrainingMatrix::loadFeatureMatrix(argv[1], trainX, trainY);
    	myAssert(dimension == dimension2, "Feature dimensions in Train & Test should be same");

        fprintf(stderr, "start to train...\n");
        rng.init();
        RANDOM_FEATURES = 4;
        RANDOM_POSITIONS = 8;
        solver->train(trainX, trainY, 200, 1, 100, featureNames);
        solver->dump(argv[5]);

        vector< pair<double, string> > ranks;
        for (size_t i = 0; i < featureNames.size(); ++ i) {
            ranks.push_back(make_pair(featureImportance[i], featureNames[i].c_str()));
        }
        sort(ranks.rbegin(), ranks.rend());
        fprintf(stderr, "=== selected feature importance ===\n");
        for (size_t i = 0; i < ranks.size() && i < 10; ++ i) {
            fprintf(stderr, "%.6f\t%s\n", ranks[i].first, ranks[i].second.c_str());
        }
    } else {
        solver->load(argv[5]);
    }

    FILE* out = tryOpen(argv[3], "w");
    int mat[2][2] = {{0, 0}, {0, 0}};
	for (size_t i = 0; i < testX.size(); ++ i) {
		double pred = solver->estimate(testX[i]);
		int pv = pred > 0.5 ? 1 : 0;
        fprintf(out, "%d,%.6f,%d\n", pv, pred, (int)testY[i]);

        ++ mat[pv][(int)testY[i]];
	}
	fclose(out);

    double accuracy = (double)(mat[0][0] + mat[1][1]) / testX.size();
    fprintf(stderr, "Accuracy = %.10f\n", accuracy);
    double recall = (double)(mat[1][1]) / (mat[0][1] + mat[1][1]);
    double precision = (double)(mat[1][1]) / (mat[1][0] + mat[1][1]);
    fprintf(stderr, "Recall = %.10f\n", recall);
    fprintf(stderr, "Precision = %.10f\n", precision);
    double F1 = recall * precision * 2 / (recall + precision);
    fprintf(stderr, "F1 score = %.10f\n", F1);

    fprintf(stderr, "Accuracy = %.10f\n", (double)(mat[0][0] + mat[1][1]) / testX.size());
    fprintf(stderr, "Baseline = %.10f\n", (double)max(mat[0][1] + mat[1][1], mat[1][0] + mat[0][0]) / testX.size());

    printf("%.10f,%.10f,%.10f,%.10f\n", accuracy, precision, recall, F1);

	return 0;
}

