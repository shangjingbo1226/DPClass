#ifndef __RANDOM_FOREST_H__
#define __RANDOM_FOREST_H__

#include "../utils/helper.h"
#include "../utils/random.h"
using namespace RandomNumbers;
#include "../patterns/classification_rule.h"

vector<double> featureImportance;

int RANDOM_FEATURES = 4;
int RANDOM_POSITIONS = 8;

namespace RandomForestRelated
{

double binaryEntropy(int p1, int total)
{
    if (p1 == 0 || p1 == total) {
        return 0;
    }
    double p = p1 / (double)total;
    return - p * log2(p) - (1 - p) * log2(1 - p);
}

struct TreeNode {
	bool leaf;
	int level, feature;
	double value, result;
	int left, right;

	TreeNode() {
		leaf = false;
		level = feature = left = right = -1;
		value = result = 0;
	}
};

class DecisionTree
{
public:
	vector<TreeNode> nodes;

	void dump(FILE* out) {
	    size_t size = nodes.size();
	    fwrite(&size, sizeof(size), 1, out);
	    if (size > 0) {
    	    fwrite(&nodes[0], sizeof(nodes[0]), size, out);
	    }
	}

	void load(FILE* in) {
	    size_t size;
	    fread(&size, sizeof(size), 1, in);
	    nodes.resize(size);
	    if (size > 0) {
    	    fread(&nodes[0], sizeof(nodes[0]), size, in);
	    }
    }

	DecisionTree() {}

	DecisionTree(vector< vector<double> > &features, vector<double> &results, int minNodeSize, int maxLevel = 18, vector<string> featureNames = vector<string>()) {
	    if (features.size() == 0) {
	        return;
	    }

	    vector< vector<int> > featureGroups;
	    if (featureNames.size() != 0) {
	        unordered_map<string, int> name2id;
	        for (int i = 0; i < featureNames.size(); ++ i) {
	            string name = featureNames[i];
	            if (name.find("=") != -1) {
	                name = name.substr(0, name.find("="));
	            }
	            if (!name2id.count(name)) {
	                name2id[name] = featureGroups.size();
	                featureGroups.push_back(vector<int>());
	            }
	            featureGroups[name2id[name]].push_back(i);
	        }
	    } else {
	        for (int i = 0; i < features[0].size(); ++ i) {
	            featureGroups.push_back(vector<int>(1, i));
	        }
	    }

		TreeNode root;
		root.level = 0;
		nodes.push_back(root);

        // Warning! Only work for binary labels!
		vector<int> index[2];
		for (int i = 0; i < (int)results.size(); ++ i) {
			index[(int)results[i]].push_back(i);
		}
		vector<int> rootBag;
		for (int i = 0; i < results.size(); ++ i) {
            rootBag.push_back(rng.next(results.size()));
		}
		/*for (int type = 0; type < 2; ++ type) {
			int selected = (int)(min(index[0].size(), index[1].size()));
			for (int i = 0; i < selected; ++ i) {
				int id = index[type][rng.next(index[type].size())];
				rootBag.push_back(id);
			}
		}*/

		vector< vector<int> > nodeBags;
		nodeBags.push_back(rootBag);

		for (int curNode = 0; curNode < (int)nodes.size(); ++ curNode) {
			TreeNode &node = nodes[curNode];
			vector<int> &bag = nodeBags[curNode];

			myAssert(bag.size() > 0, "[ERROR] empty node in decision tree!");
			myAssert(bag.size() >= minNodeSize, "[ERROR] bag is too small!");

			bool equal = true;

			double first = results[bag[0]];
			for (int i = 1; i < (int)bag.size(); ++ i) {
				if (sign(results[bag[i]] - first)) {
					equal = false;
					break;
				}
			}

			if (equal || (int)bag.size() < minNodeSize * 2 || node.level >= maxLevel) {
				// leaf
				node.leaf = true;
				for (int i = 0; i < (int)bag.size(); ++ i) {
					node.result += results[bag[i]];
				}
				node.result /= bag.size();
				continue;
			}

            double bag_sum = 0;
            for (int i = 0; i < (int)bag.size(); ++ i) {
                bag_sum += results[bag[i]];
            }
            double bag_mse = 0;
            bag_mse += binaryEntropy((int)bag_sum, bag.size());

			int bestFeature = -1;
			int bestLeft = 0, bestRight = 0;
			double bestValue = 0;
			double bestMSE = 1e100;
			for (int _ = 0; _ < RANDOM_FEATURES; ++ _) {
			    int groupID = rng.next(featureGroups.size());
			    int featureID = featureGroups[groupID][rng.next(featureGroups[groupID].size())];

			    bool continuous = false;
			    if (featureGroups[groupID].size() == 1) {
			        // continuous variable
			        continuous = true;
			    } else {
			        // categorical variable
			        continuous = false;
                }
				for (int __ = 0; __ < RANDOM_POSITIONS; ++ __) {
				    double splitValue = 0.5; // for categorical variable
				    if (continuous) {
				        // continuous
    					int instanceID = bag[rng.next(bag.size())];
					    splitValue = features[instanceID][featureID];
				    } else {
				        // categorical
				        if (__) {
				            // get a new value
				            featureID = featureGroups[groupID][rng.next(featureGroups[groupID].size())];
				        }
				    }

					double sum[2] = {0, 0};
					int total[2] = {0, 0};
					for (int i = 0; i < (int)bag.size(); ++ i) {
						int id = bag[i];
						int side = features[id][featureID] < splitValue;
						sum[side] += results[id];
						++ total[side];
					}

					if (total[0] < minNodeSize || total[1] < minNodeSize) {
						continue;
					}

					double entropy[2];
					for (int i = 0; i < 2; ++ i) {
					    entropy[i] = binaryEntropy((int)sum[i], total[i]);
					}
					double entropyAll = (entropy[0] * total[0] + entropy[1] * total[1]) / bag.size();
					double mse = entropyAll;

					/*
					double mean[2] = {sum[0] / total[0], sum[1] / total[1]};

					double mse = 0;
					for (int i = 0; i < (int)bag.size(); ++ i) {
						int id = bag[i];
						int side = features[id][featureID] < splitValue;

						mse += sqr(results[id] - mean[side]);
					}
					*/

					if (mse < bestMSE) {
						bestMSE = mse;
						bestValue = splitValue;
						bestFeature = featureID;
						bestLeft = total[1];
						bestRight = total[0];
					}
				}
			}
			if (bestLeft < minNodeSize || bestRight < minNodeSize) {
				// leaf
				node.leaf = true;
				for (int i = 0; i < (int)bag.size(); ++ i) {
					node.result += results[bag[i]];
				}
				node.result /= bag.size();
				continue;
			}

			featureImportance[bestFeature] += bag_mse - bestMSE;

			double nextValue = -1e100;
			for (int i = 0; i < (int)bag.size(); ++ i) {
				int id = bag[i];
				if (features[id][bestFeature] < bestValue) {
					nextValue = max(nextValue, features[id][bestFeature]);
				}
			}

			TreeNode left, right;

			left.level = right.level = node.level + 1;
			node.feature = bestFeature;
			node.value = (bestValue + nextValue) / 2;
			node.left = nodes.size();
			node.right = nodes.size() + 1;

			myAssert(bestLeft >= minNodeSize && bestRight >= minNodeSize, "[ERROR] bag is too small");

			vector<int> leftBag(bestLeft, 0), rightBag(bestRight, 0);
			for (int i = 0; i < (int)bag.size(); ++ i) {
				int id = bag[i];
				if (features[id][bestFeature] < bestValue) {
					leftBag[-- bestLeft] = id;
				} else {
					rightBag[-- bestRight] = id;
				}
			}

			myAssert(bestLeft == 0, "Left Space Remained!");
			myAssert(bestRight == 0, "Right Space Remained!");

			nodes.push_back(left);
			nodes.push_back(right);

			nodeBags.push_back(leftBag);
			nodeBags.push_back(rightBag);
		}
	}

	double estimate(vector<double> &features) {
		TreeNode *current = &nodes[0];
		while (!current->leaf) {
			if (features[current->feature] < current->value) {
				current = &nodes[current->left];
			} else {
				current = &nodes[current->right];
			}
		}
		return current->result;
	}

	double overallSum = 0, overallSum2 = 0;

	void traverse(int id, Rule &current, Rules &all, vector< vector<double> > &train, vector<double> &trainY, int MIN_SUP) {
        if (current.satisfiedTrainings.size() < MIN_SUP) {
            return;
        }
        double sum = 0, sum2 = 0;
        FOR (tid, current.satisfiedTrainings) {
            sum += trainY[*tid];
            sum2 += sqr(trainY[*tid]);
        }

        if (id == 0) {
            overallSum = sum;
            overallSum2 = sum2;
        } else {
            double localMSE = binaryEntropy(sum, current.satisfiedTrainings.size());
            int outsideCnt = train.size() - current.satisfiedTrainings.size();
            double outsideMSE = binaryEntropy(overallSum - sum, outsideCnt);
            /*double localMSE = sum2 / current.satisfiedTrainings.size() - sqr(sum / current.satisfiedTrainings.size());

            double outsideSum = overallSum - sum, outsideSum2 = overallSum2 - sum2;
            int outsideCnt = train.size() - current.satisfiedTrainings.size();
            double outsideMSE = outsideSum2 / outsideCnt - sqr(outsideSum / outsideCnt);
            */
//            cerr << localMSE << " " << outsideMSE << " " << overallSum << " " << sum << endl;

            current.loss = localMSE; // (localMSE * current.satisfiedTrainings.size() + outsideMSE * outsideCnt) / train.size();
            //current.loss = localMSE / current.satisfiedTrainings.size();

            all.push_back(current);
        }

	    if (nodes[id].leaf) {
	        //all.push_back(current);
	        return;
	    }

	    vector<int> bag = current.satisfiedTrainings;

        //split left & right
	    vector<int> leftBag, rightBag;
	    int index = nodes[id].feature;
	    double sep = nodes[id].value;
	    FOR (tid, bag) {
	        if (train[*tid][index] < sep) {
	            leftBag.push_back(*tid);
	        } else {
	            rightBag.push_back(*tid);
	        }
	    }

	    current.push_back(Condition(index, sep, true));
	    current.satisfiedTrainings = leftBag;
	    traverse(nodes[id].left, current, all, train, trainY, MIN_SUP);
	    current.pop_back();

	    current.push_back(Condition(index, sep, false));
	    current.satisfiedTrainings = rightBag;
	    traverse(nodes[id].right, current, all, train, trainY, MIN_SUP);
	    current.pop_back();

	    current.satisfiedTrainings = bag;
	}

	Rules getRules(vector< vector<double> > &train, vector<double> &trainY, int MIN_SUP) {
	    Rule current;
	    Rules all;
	    for (int i = 0; i < train.size(); ++ i) {
	        current.satisfiedTrainings.push_back(i);
	    }
	    traverse(0, current, all, train, trainY, MIN_SUP);
	    return all;
	}
};

class RandomForest
{
	vector<DecisionTree> trees;
	vector< vector<double> > features;
	vector<double> results;
public:
    void dump(string filename) {
        FILE* out = fopen(filename.c_str(), "wb");
        size_t size = trees.size();
        fwrite(&size, sizeof(size), 1, out);
        for (size_t i = 0; i < trees.size(); ++ i) {
            trees[i].dump(out);
        }
        fclose(out);
    }

    void load(string filename) {
        FILE* in = fopen(filename.c_str(), "rb");
        size_t size;
        fread(&size, sizeof(size), 1, in);
        trees.resize(size);
        for (size_t i = 0; i < trees.size(); ++ i) {
            trees[i].load(in);
        }
        fclose(in);
    }

	void clear() {
		features.clear();
		results.clear();
		trees.clear();
	}

	void train(vector< vector<double> > &_features, vector<double> _results, int treesNo = 100, int minNodeSize = 100, int maxLevel = 100, vector<string> featureNames = vector<string>()) {
		if (features.size() == 0) {
			features = _features;
			results = _results;
			if (features.size() > 0) {
    			featureImportance.resize(features[0].size(), 0);
			}
		}
		myAssert(features.size() == results.size(), "[ERROR] wrong training data!");

		for (int i = 0; i < treesNo; ++ i) {
			trees.push_back(DecisionTree(_features, _results, minNodeSize, maxLevel, featureNames));
		}
	}

	Rules getRules(vector< vector<double> > &train, vector<double> &trainY, int MIN_SUP) {
	    Rules ret;
		for (int i = 0; i < (int)trees.size(); ++ i) {
		    ret.extend(trees[i].getRules(train, trainY, MIN_SUP));
		}
		return ret;
	}

	double estimate(vector<double> &features) {
		if (trees.size() == 0) {
			return 0.0;
		}

		double sum = 0;
		for (int i = 0; i < (int)trees.size(); ++ i) {
			sum += trees[i].estimate(features);
		}
		return sum / trees.size();
	}
};
};

#endif
