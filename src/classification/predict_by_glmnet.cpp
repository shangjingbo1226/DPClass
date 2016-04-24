#include "../utils/helper.h"
#include "../utils/training_matrix.h"

vector< vector<double> > trainX, testX;
vector<double> trainY, testY;

int main(int argc, char* argv[])
{
	if (argc != 4) {
		fprintf(stderr, "[usage] <train.csv> <test.csv> <output: prediction>\n");
		return -1;
	}

    int dimension = TrainingMatrix::loadFeatureMatrix(argv[1], trainX, trainY);
	int dimension2 = TrainingMatrix::loadFeatureMatrix(argv[2], testX, testY);

    FILE* out = tryOpen("../tmp/glmnet_train.txt", "w");
    for (int i = 0; i < trainX.size(); ++ i) {
        for (int j = 0; j < trainX[i].size(); ++ j) {
            fprintf(out, "%.6f ", trainX[i][j]);
        }
        fprintf(out, "%d\n", (int)trainY[i]);
    }
    fclose(out);

    FILE* test_out = tryOpen("../tmp/glmnet_test.txt", "w");
    for (int i = 0; i < testX.size(); ++ i) {
        for (int j = 0; j < testX[i].size(); ++ j) {
            fprintf(test_out, "%.6f ", testX[i][j]);
        }
        fprintf(test_out, "%d\n", (int)testY[i]);
    }
    fclose(test_out);

    string command = "matlab -nosplash -nodesktop -logfile remoteAutocode.log -r \"predict('../tmp/glmnet_train.txt', '../tmp/glmnet_test.txt', '" + (string)argv[3] + "')\"";
    cerr << command << endl;
    system(command.c_str());

    if (false) {
        FILE* in = tryOpen(argv[3], "r");
        string filename = (string)argv[2] + ".prediction_glmnet";
        FILE* out = tryOpen(filename, "w");
        for (int i = 0; i < testY.size(); ++ i) {
            int x;
            myAssert(fscanf(in, "%d", &x) == 1 && 0 <= x && x <= 1, "[ERROR] prediction file error");
            fprintf(out, "%d\n", x == testY[i]);
        }
        fclose(in);
        fclose(out);
    }

	return 0;
}

