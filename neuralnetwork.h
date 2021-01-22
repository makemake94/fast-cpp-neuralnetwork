/* namespace std, typedfs and includes are in main.cpp */


// PRINTER OBJECT (see neuralnetwork.train)

struct {
	int col1_len = 10;
	int col_len = 14;
	int tab_len;
	string line;
	
	void adjust(const vector<Matrix> &inputs, const vector<Matrix> &outputs){
		tab_len = max(inputs[0].data_size, outputs[0].data_size);
		line = string(col1_len, '-') + "+" + string(col_len * tab_len, '-');
	}
	void print(const Matrix &inp, const Matrix &targ, const Matrix &out, const Matrix &err, string title = "", bool end_line = true){
		stringstream ss;
		if (title != "") ss << title << endl;
		ss << line << endl;
		ss.width(col1_len); ss << left << "input " << "|"; for (int i = 0; i != inp.data_size; i++) {ss.width(col_len); ss << right <<  inp.data[i];} ss << endl;
		ss.width(col1_len); ss << left << "target" << "|"; for (int i = 0; i != targ.data_size; i++){ss.width(col_len); ss << right <<  targ.data[i];} ss << endl;
		ss.width(col1_len); ss << left << "output" << "|"; for (int i = 0; i != out.data_size; i++) {ss.width(col_len); ss << right <<  out.data[i];} ss << endl;
		ss << line << endl;
		ss.width(col1_len); ss << left << "error " << "|"; for (int i = 0; i != err.data_size; i++){ss.width(col_len); ss << right <<  err.data[i];} ss << endl;
		if (end_line) ss << line << endl << endl;
		cout << ss.str();
	}
	// function to print many data points togheter (used in neuralnetwork.check)
	void print_many(const vector<Matrix> &inputs, const vector<Matrix> &targ, const vector<Matrix> &outputs, const vector<Matrix> &errors, string title = "RESULT"){ 
		for (int i = 0; i != inputs.size(); i++)
			print(inputs[i], targ[i], outputs[i], errors[i], (i == 0) ? title : line, (i == inputs.size()-1) ? true : false);
	}
}\
nn_printer;


// NEURAL NETWORK CLASS

namespace NeuralNetworkFuncs {
	// "d" stands for derivative
	Matrix mse(const Matrix& x, const Matrix& y){return x.minus(y).pow(2);}
	Matrix d_mse(const Matrix& x, const Matrix& y){return x.minus(y);}
}

class NeuralNetwork{
	public:
		vector<Layer> layers;
		Optimizer * optimizer;
		
		// TO-DO: now MSE is hard-coded, but other error functions (and derivatives) can be added in NeuralNetworkFuncs and given as parameter in nn constructors
		Matrix (* err_func)(const Matrix&, const Matrix&) = NeuralNetworkFuncs::mse;
		Matrix (* d_err_func)(const Matrix&, const Matrix&) = NeuralNetworkFuncs::d_mse;
		
		// CONSTRUCTORS
		
		NeuralNetwork() {};
		NeuralNetwork(vector<Layer> _layers) : layers(_layers) {}
		// used in main
		NeuralNetwork(vector<int> dimensions, Optimizer * _optimizer =  new Optimizer(), const Function init_gen = LayerFuncs::rand_init){
			for (int i = 1; i != dimensions.size(); i++)
				add_layer(Layer(dimensions[i]));
			
			// dimension[0] is input dimension
			init(dimensions[0], _optimizer);
		}
				
		// add layer
		void add_layer(const Layer layer){layers.push_back(layer);}
		
		// initialize (sets layers' weights and optimizer default state)
		void init(int input_size, Optimizer * _optimizer = new Optimizer(), const Function init_gen = LayerFuncs::rand_init) {		
			// set layers' input and get output sizes 
			vector<int> layer_output_sizes;
			for (Layer &layer : layers){
				layer.init(input_size, init_gen);
				input_size = layer.output_size;
				layer_output_sizes.push_back(input_size);
			}

			// assign optimizer to nn and initialize it with layers' output sizes	
			optimizer = _optimizer;
			optimizer->init(layer_output_sizes);
		}

		// forward propagation
		Matrix forward_propagation(Matrix forward_val, const int batch_i = 0){
			for (Layer &layer : layers) layer.fire(forward_val, batch_i);
			return forward_val;
		}
		                   
		// backward propagation
		void back_propagation(const Matrix &nn_output, const Matrix &target, const int batch_i = 0){
			
			// calcuate error derivative... first delta
			Matrix d = d_err_func(nn_buffer, target);
			
			// output layer's params used in back_fire
			const Matrix * next_layer_w = nullptr, * next_layer_in = &nn_output;
			
			// back-cycle through layers
			for(int layer_i = layers.size() - 1; layer_i != -1; layer_i--){
				Layer &layer = layers[layer_i];
				
				// back_fire and update next layer's params for back_fire
				layer.back_fire(d, next_layer_w, next_layer_in);
				next_layer_w = &layer.w, next_layer_in = &layer.ins[batch_i]; 

				// add gradient in layer's cache
				layer.gs[batch_i] = optimizer->compute_g(d, layer_i);
			}
		}
			
		// TRAIN FUNCTION
		
		void train(vector<Matrix>& inputs, vector<Matrix>& targets, const int epochs = 10000, const int batch_size = 1, const double eta = 0.1, const int print_count = 1000){
			
			// check data
			int input_size = inputs.size();
			if (input_size % batch_size != 0){
				input_size -= (input_size % batch_size);
				cout << "il batch size non e' un sottomuliplo del numero di item: verranno utilizzati i primi " << input_size << endl;
			}
			if (input_size > targets.size())
				{cout << "numero di output non sufficiente" << endl;}

			// prepare printer (layout's width is based on dimensionality)
			if (print_count) nn_printer.adjust(inputs, targets);
			
			// initialize inputs and gradients' containers for batch learning
			for (Layer &layer : layers) layer.prepare_batch_learning(batch_size);
			
			// initialize nn_bufferput container
			Matrix nn_output(targets[0].size);
			
			// cycle through epochs
			for (int epoch = 0; epoch != epochs; epoch++) {
				
				// select item to be printed in this epochoch
				const int print_i = ((epoch % print_count == 0) and (print_count)) ? rand() % input_size : -1;

				// cycle through examples
				for (int i = 0, batch_i = 0; i != input_size; i++){	
	
					// load input into buffer and start back_propagation (nn_buffer is given by reference and changed inside the function)
					nn_output = forward_propagation(inputs[i], batch_i);
				
					// backward propagation
					back_propagation(nn_output, targets[i], batch_i++);
					                       
					// if batch is completed, apply gradients
					if (batch_i == batch_size){
						batch_i = 0;
						for (Layer &layer : layers) layer.apply_g(eta);
					}
					
					// update optimizer's state after iteration (t++ in Adam and Adamax)
					optimizer->update();
					
					// if item has to be printed, do it
					if (i == print_i) {
						stringstream title; title << epoch << ") ITEM " << i << "/" << input_size;
						nn_printer.print(inputs[i], targets[i], nn_output, err_func(nn_output, targets[i]), title.str());
					}                                                
				}
				
				// reset optimizer's state after epoch (t=1 and flush m1 and m2 in Adam and Adamax)
				optimizer->reset(); 
			}
		}

		// CHECK PERFORMANCE ON TEST DATA	
				
		vector<vector<Matrix>> check(vector<Matrix>& inputs, vector<Matrix>& targets, bool print = true){
			
			// initialize containers
			vector<Matrix> outputs(inputs.size()), errors(inputs.size());
				
			// register final results
			for (int i = 0; i != inputs.size(); i++){
				outputs[i] = forward_propagation(inputs[i]);
				errors[i] = err_func(outputs[i], targets[i]);	
			}
			
			// print final results, all togheter
			if (print){nn_printer.adjust(inputs, targets); nn_printer.print_many(inputs, targets, outputs, errors);} 			

			// return final results			
			return {outputs, errors};
		}
		
		// DESTRUCTOR
		
		~NeuralNetwork(){delete optimizer;} //sure?
};