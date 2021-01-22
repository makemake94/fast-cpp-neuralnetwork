/* namespace std, typedfs and includes are in main.cpp */


// OPTMIZER (VIRTUAL) CLASS

// base class defines no optmizer at all
class Optimizer{
	public:
		Optimizer (){}
		// run when first coupled to an nn, and initializes parameters as correctly sized matrices (related to layers)
		virtual void init(vector<int> layer_output_sizes){}
		// run after every epoch (see neuralnetwork.train)
		virtual void reset(){}
		// run after every iteration
		virtual void update(){}
		// run in backprop
		virtual Matrix compute_g(const Matrix &d, int layer_i){return d;}
	};
	
	// those are real optmizers (<3 adamax)
	class Adam : public Optimizer{
		public:
			vector<Matrix> state; // m1 and m2, for each layer (see init)
			const float b1, b2, e;
			unsigned int t;
			
			Adam(const float _b1 = 0.9, const float _b2 = 0.999, const float _e = 1e-8)\
			: b1(_b1), b2(_b2), e(_e){}
			
			void init(const vector<int> layer_output_sizes){
				t = 1;
				state.clear();
				for (int layer_output_size : layer_output_sizes)
					for (auto _ = 2; _--;) //m1	and m2
						state.push_back(Matrix::Zeros(new int[2]{layer_output_size, 1}));
			}
			
			void update(){t++;} 
			
			void reset(){ 
				t = 1;
				for (Matrix& m : state) m.fill(0);
			}
			
			Matrix compute_g(const Matrix & d, const int layer_i){
				Matrix &m1 = state[layer_i*2], &m2 = state[layer_i*2+1];
				m1.inp_scal_mul(b1); m1.inp_sum(d.scal_mul(1-b1));
				m2.inp_scal_mul(b2); m2.inp_sum(d.pow(2).scal_mul(1-b2));
				return m1.div(1-powf(b1,t)).div(m2.div(1-powf(b2,t)).sqrt().sum(e)); //m1hat / sgrt(m2hat)+epochs
			}
	};
	class Adamax : public Optimizer{
		public:
			vector<Matrix> state; // m1 and m2, for each layer (see init)
			const float b1, b2, e;
			unsigned int t;
			
			Adamax(const float _b1 = 0.9, const float _b2 = 0.999, const float _e = 1e-8)\
			: b1(_b1), b2(_b2), e(_e) {}
	
			void init(const vector<int> layer_output_sizes){
				state.clear();
				for (int layer_output_size : layer_output_sizes)
					for (auto _ = 2; _--;)					 //m1	and m2
						state.push_back(Matrix::Zeros(new int[2]{layer_output_size, 1}));
				t = 1;
			}
	                                                            
			void reset(){
				for (Matrix & m : state) m.fill(0);
				t = 1;
			}
	
			void update(){t++;}
			
			Matrix compute_g(const Matrix & d, const int layer_i){
				Matrix &m1 = state[layer_i*2], &m2 = state[layer_i*2+1];
				const float b2l = b2, b1l = b1; // local for lambda capture
				transform(m1.data, m1.data+m1.data_size, d.data, m1.data,[b1l](Value i, Value j){return i*b1l + j*(1-b1l);});
				transform(m2.data, m2.data+m2.data_size, d.data, m2.data,[b2l](Value i, Value j){return max(i*b2l, abs(j));}); // max(m2*b2,|g|);
				return m1.div(1-powf(b1, t)).div(m2.sum(e)); // g
			}
	};