#include "../utils/helper.h"
#include "../utils/union_find_set.h"
#include "classification_rule.h"
#include "../classification/sparse_svm.h"

#include "../utils/training_matrix.h"
using namespace TrainingMatrix;

#include <omp.h>

int NTHREADS = 2;
int SMALL_ROUNDS = 10;
int LARGE_ROUNDS = 50;
double LAMBDA = 1.0;

double calc_similarity(const vector<int> &a, const vector<int> &b)
{
    int common = 0;
    for (int i = 0, j = 0; i < a.size() && j < b.size();) {
        if (a[i] < b[j]) {
            ++ i;
        } else if (a[i] == b[j]) {
            ++ i;
            ++ j;
            ++ common;
        } else {
            ++ j;
        }
    }
    return common / sqrt((double)a.size() * b.size());
}

Rules selectRulesBySVM(Rules &rules, int topK, vector<double> &labels)
{
    vector<double> globalWeight(rules.size(), 0);
    vector<vector<pair<int, double>>> x(labels.size(), vector<pair<int, double>>());
    vector<int> y(labels.size(), 1);
    double avg = 0;
    for (int i = 0; i < labels.size(); ++ i) {
        if (labels[i] == 0) {
            y[i] = -1;
        }
        avg += y[i];
    }
    avg /= labels.size();

    for (int i = 0; i < rules.size(); ++ i) {
        double sum = 0;
        FOR (id, rules[i].satisfiedTrainings) {
            sum += y[*id];
        }
        globalWeight[i] = sum / rules[i].satisfiedTrainings.size();
    }

    Rules selected;
    vector<bool> chosen(rules.size(), false);
    vector<double> weight;
    for (int iter = 0; iter < topK; ++ iter) {
        vector<pair<double, int>> performances(NTHREADS, make_pair(-1.0, 1));
        #pragma omp parallel for
        for (int i = 0; i < rules.size(); ++ i) {
            if (!chosen[i]) {
                vector<vector<pair<int, double>>> newX = x;
                vector<double> newWeight = weight;

                for (int j = 0; j < rules[i].satisfiedTrainings.size(); ++ j) {
                    newX[rules[i].satisfiedTrainings[j]].push_back(make_pair(iter, 1.0));
                }
                newWeight.push_back(globalWeight[i]);
                double acc = SparseSVM::train(newX, y, newWeight, avg, SMALL_ROUNDS + iter, LAMBDA);
                int threadID = omp_get_thread_num();
                performances[threadID] = max(performances[threadID], make_pair(acc, -i));
            }
        }

        pair<double, int> performance = make_pair(-1.0, 1);
        for (int i = 0; i < performances.size(); ++ i) {
            performance = max(performance, performances[i]);
        }
        int best = -performance.second;
        if (best == -1) {
            break;
        }
        chosen[best] = true;
        selected.push_back(rules[best]);
        for (int j = 0; j < rules[best].satisfiedTrainings.size(); ++ j) {
            x[rules[best].satisfiedTrainings[j]].push_back(make_pair(iter, 1.0));
        }
        weight.push_back(globalWeight[best]);
        double acc = SparseSVM::train(x, y, weight, avg, LARGE_ROUNDS, LAMBDA);
        //fprintf(stderr, "k = %d, best acc = %.6f, sat = %d, init = %.6f\n", iter, acc, rules[best].satisfiedTrainings.size(), globalWeight[best]);
        //fprintf(stderr, "\t%.6f\n", weight.back());
    }
    return selected;
}

int main(int argc, char* argv[])
{
    int topK = 0;
	if (argc != 9 || sscanf(argv[3], "%d", &topK) != 1 || sscanf(argv[5], "%d", &NTHREADS) != 1 || sscanf(argv[6], "%d", &SMALL_ROUNDS) != 1 || sscanf(argv[7], "%d", &LARGE_ROUNDS) != 1 || sscanf(argv[8], "%lf", &LAMBDA) != 1) {
		fprintf(stderr, "[usage] <training data, csv format> <candidate rules> <top-k rules> <output selected rules> <nthread> <small_svm_round> <large_svm_round>\n");
		return -1;
	}
	cerr << "# threads = " << NTHREADS << endl;

	omp_set_num_threads(NTHREADS);
	int dimension = loadFeatureMatrix(argv[1]);
	Rules rules;
	rules.load(argv[2]);

	if (topK > rules.size()) {
	    topK = rules.size();
	}

	cout << "start to select " << topK << " rules out of " << rules.size() << " rules..." << endl;

	rules.aggresively_unique();
	rules.sortByLoss();

	Rules selected = selectRulesBySVM(rules, topK, trainY);

	cerr << "[Selected Top " << topK << " Rules]" << endl;
	for (int i = 0; i < selected.size(); ++ i) {
	    cerr << selected[i].loss << "\t" << selected[i].showup(featureNames) << endl;
	}
    selected.dump(argv[4]);

	return 0;
}

