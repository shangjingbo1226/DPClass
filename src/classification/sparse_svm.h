#ifndef __SPARSE_SVM_H__
#define __SPARSE_SVM_H__

#include "../utils/random.h"

struct SparseSVM
{
    static vector<int> getPrediction(const vector<vector<pair<int, double>>> &x, const vector<int> &y, const vector<double> &w, double bias) {
        vector<int> ret(x.size(), 0);
        #pragma omp parallel for
        for (int i = 0; i < x.size(); ++ i) {
            double pred = bias;
            for (int j = 0; j < x[i].size(); ++ j) {
                pred += x[i][j].second * w[x[i][j].first];
            }
            if (pred > 0) {
                ret[i] = 1;
            }
        }
        return ret;
    }

    static double calcAccuracy(const vector<vector<pair<int, double>>> &x, const vector<int> &y, const vector<double> &w, double bias) {
        double accuracy = 0;
        #pragma omp parallel for reduction( + : accuracy )
        for (int i = 0; i < x.size(); ++ i) {
            double pred = bias;
            for (int j = 0; j < x[i].size(); ++ j) {
                pred += x[i][j].second * w[x[i][j].first];
            }
            if (pred * y[i] > 0) {
                accuracy += 1;
            }
        }
        accuracy /= x.size();
        return accuracy;
    }

    static double train(const vector<vector<pair<int, double>>> &x, const vector<int> &y, vector<double> &w, double bias = 0.0, int maxIter = 10, double LAMBDA = 0.01, double learningRate = 0.001) {
        RandomNumbers::RandomNumberGenerator local;
        local.init(19910724);
        for (int i = 0; i < w.size(); ++ i) {
            w[i] = 0;
        }
        vector<int> order;
        for (int i = 0; i < x.size(); ++ i) {
            order.push_back(i);
        }
        for (int iter = 0; iter < maxIter; ++ iter) {
            double accuracy = calcAccuracy(x, y, w, bias);
            //fprintf(stderr, "[SVM] iter = %d, accuracy = %.10f\n", iter, accuracy);
            vector<double> bakup = w;
            for (int __ = 0; __ < order.size(); ++ __) {
                int i = order[__];
            //for (int i = 0; i < x.size(); ++ i) {
                double pred = bias;
                for (int j = 0; j < x[i].size(); ++ j) {
                    pred += x[i][j].second * w[x[i][j].first];
                }
                if (1 - pred * y[i] > 0) {
                    for (int j = 0; j < x[i].size(); ++ j) {
                        double coef = x[i][j].second;
                        double delta = learningRate * (-y[i] * coef + LAMBDA * w[x[i][j].first]);
                        w[x[i][j].first] -= delta;
                    }
                }
            }
            double newAccuracy = calcAccuracy(x, y, w, bias);
            if (accuracy > newAccuracy + EPS) {
                learningRate /= 2;
                w = bakup;
            } else if (newAccuracy > accuracy + EPS){
                learningRate *= 1.3;
            } else {
                learningRate *= 1.3;
                learningRate = min(learningRate, 1.0);
            }
        }
        return calcAccuracy(x, y, w, bias);
    }

};

#endif
