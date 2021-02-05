using namespace std;
typedef float Value;
typedef Value (* Function) (Value);

#include <vector>
#include <string.h>
#include <functional>
#include <algorithm>
#include <math.h>
#include <sstream>
#include <iostream>

#include <random>
#include <ctime>

#include "matrix.h"
#include "layer.h"
#include "optimizer.h"
#include "neuralnetwork.h"


int main(){

	// neural network, with 3 layers
	// - input: 3 nodes
	// - hidden: 5 nodes
	// - output: 1 nodes
	
	// only a "line" of nodes (?x1) is supported by the underlying Matrix class
	
	// and an Adamax optmizer
	
	NeuralNetwork nn({3,5,1}, new Adamax());		
	
	// dataset (split in inputs and targets)
	vector<Matrix> inputs, targets;
	
	int input_size = 1000;
	
	// create examples
	for (int i = 0; i != input_size; i++){
		
		// arbitrary xs - inputs
		Value x1 = float(rand() % 1000 / 1000.0), x2 = float(rand() % 1000 / 1000.0), x3 = float(rand() % 1000 / 1000.0);
		
		// an arbitrary function to create ys - targets (limited in [0,1])
		Value y = sqrt(x1+x2+powf(x3, 2)) / sqrt(3);
		
		// add to dataset
		inputs.push_back(Matrix(new int[2]{3,1}, {x1,x2,x3}));
		targets.push_back(Matrix(new int[2]{1,1}, {y}));	
	}
	
	// split train and test
	int cut = input_size * 0.8;
	vector<Matrix> inputs_train(inputs.begin(), inputs.begin()+cut), inputs_test(inputs.begin()+cut, inputs.end());
	vector<Matrix> targets_train(targets.begin(), targets.begin()+cut), targets_test(targets.begin()+cut, targets.end());
	
	// train
	int epochs = 2000;
	int print_counter = 100;
	int batch_size = 10;
	nn.train(inputs_train, targets_train, epochs, batch_size, 0.1, print_counter);
	
	cout << "Learning completed. Now veryfing with test set\n"; system("pause");
	
	// print test dataset with outputs and errors
	nn.check(inputs_test, targets_test);
	
	return 0;
}