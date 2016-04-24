#include "../classification/random_forest.h"
#include "../utils/helper.h"

using namespace RandomForestRelated;

#include "../utils/training_matrix.h"
using namespace TrainingMatrix;

int main(int argc, char* argv[])
{
    RANDOM_FEATURES = 4;
    RANDOM_POSITIONS = 8;
    int MIN_SUP = 5;
    int MAX_DEPTH = 6;
    int TREE_NOS = 100;
	if (argc < 9 || sscanf(argv[5], "%d", &MIN_SUP) != 1 || sscanf(argv[6], "%d", &MAX_DEPTH) != 1 || sscanf(argv[7], "%d", &RANDOM_FEATURES) != 1 || sscanf(argv[8], "%d", &RANDOM_POSITIONS) != 1) {
		fprintf(stderr, "[usage] <training data, csv format> <input model> <output model> <output rules> <min_sup> <depth> <# random features> <# random positions> [optional, tree numbers]\n");
		return -1;
	}

	if (argc > 9) {
        sscanf(argv[9], "%d", &TREE_NOS);
	}

	fprintf(stderr, "MIN_SUP = %d, MAX_DEPTH = %d, TREE_NOS = %d\n", MIN_SUP, MAX_DEPTH, TREE_NOS);
	fprintf(stderr, "RANDOM_FEATURES = %d, RANDOM_POSIITONS = %d\n", RANDOM_FEATURES, RANDOM_POSITIONS);

	int dimension = loadFeatureMatrix(argv[1]);

	RandomForest *solver = new RandomForest();
	if (!strcmp(argv[2], "TRAIN")) {
	    fprintf(stderr, "start to train...\n");
	    rng.init();
	    solver->train(train, trainY, TREE_NOS, MIN_SUP, MAX_DEPTH, featureNames);
	} else {
	    fprintf(stderr, "start to load...\n");
	    solver->load(argv[2]);
	}

	if (strcmp(argv[3], "DONT_SAVE_TREE_MODEL")) {
	    solver->dump(argv[3]);
	}

	if (strcmp(argv[4], "DONT_SAVE_RULE_SET")) {
	    Rules rules = solver->getRules(train, trainY, 1);
	    int maximumConditions = 0, minimumConditions = dimension;
	    double maxLoss = 0, minLoss = 1e100;
	    for (int i = 0; i < rules.size(); ++ i) {
	        maximumConditions = max(maximumConditions, rules[i].size());
	        minimumConditions = min(minimumConditions, rules[i].size());
	        minLoss = min(minLoss, rules[i].loss);
	        maxLoss = max(maxLoss, rules[i].loss);
	    }

	    cout << "# Raw Rules = " << rules.size() << endl;
	    cout << "# maximum conditions = " << maximumConditions << endl;
	    cout << "# minimum conditions = " << minimumConditions << endl;
	    cout << "max loss = " << maxLoss << ", min loss = " << minLoss << endl;

	    rules.dump(argv[4]);
	}

	return 0;
}
