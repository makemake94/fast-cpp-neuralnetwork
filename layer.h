/* namespace std, typedfs and includes are in main.cpp */


// LAYER CLASS
// to-do suggestion: this could be made a virtual interface to support different kinds of layers, like optimizer class

namespace LayerFuncs{
	// "a" stands for activation, "d" for derivative (used in backprop)
	// I should couple them...
	Value a_sigmoid(Value a){return 1/(1+exp(-a));}
	Value d_sigmoid(Value a){return a*(1-a);}
	Value a_tanh(Value a){return tanh(a);}
	Value d_tanh(Value a){return (1 - powf(a, 2));}
	Value rand_init(Value a){return Value(-1.0 + rand() % 2001 / 1000.0);}
}

class Layer{
	public:		
		Function act, d_act;				
		Matrix b, w;
		vector<Matrix> gs; // "g" is gradient: plural because gradients are accumulated in batch learning
		vector<Matrix> ins; // inputs: have to be accumulated for batch learning to work (see apply_g)
			
		int input_size, output_size;

		Layer(int node_n, Function _act = LayerFuncs::a_sigmoid, Function _d_act = LayerFuncs::d_sigmoid)\
		: output_size(node_n), act(_act), d_act(_d_act) {}

		void init(int _input_size, Function init_gen = LayerFuncs::rand_init) {
			input_size = _input_size;
			srand(time(0)); 
			w = Matrix(new int[2]{output_size, input_size}).apply(init_gen);
			b = Matrix(new int[2]{output_size, 1}).apply(init_gen);
		}

		void prepare_batch_learning(int batch_size) 
			{ins = gs = vector<Matrix>(batch_size);}
		
		void fire(Matrix &forward_val, const int batch_i){
			ins[batch_i] = forward_val;
			forward_val = b.sum(w.mat_mul(forward_val)).apply(act); // BxA * Ax1 = Bx1 = bias dim
		}
		
		void back_fire(Matrix &d, const Matrix * next_layer_w, const Matrix * layer_out) const{
			if (next_layer_w) d = next_layer_w->T().mat_mul(d);	// BxC * Cx1 = Bx1 = out dims
			d.inp_dot_mul(layer_out->apply(d_act)); 				// back_val: Cx1 => Bx1							
		}
			
		void apply_g(const float eta){
			if (gs.size() == 1) {
				Matrix &g = gs[0], &in = ins[0];
				g.inp_scal_mul(eta);
				b.inp_minus(g);
		    		w.inp_minus(g.mat_mul(in.T()));
			} else {
				const Matrix g = Matrix::avg(gs).scal_mul(eta), in = Matrix::avg(ins);
				b.inp_minus(g);				// Bx1
		    		w.inp_minus(g.mat_mul(in.T()));		// BxA, BxA = Bx1 * 1xA (inp.T)	
			}
		}
};
