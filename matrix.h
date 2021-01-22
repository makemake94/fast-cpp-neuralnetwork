/* namespace std, typedfs and includes are in main.cpp */


// MATRIX CLASS

class Matrix{
	public:
		Value * data; 	// dynamic array (quite faster than vectors)
		int size[2]; 	// defines a 2-dimensional matrix 
		int data_size; 	// length of dynamic array = size[0] * size[1]
		
		// MEMORY AND SIZE
		
		static Value * create_zero_arr(const int data_arr_size)
			{return static_cast<Value *>(memset(new Value[data_arr_size], 0, data_arr_size*sizeof(Value)));}
				
		static Value * create_data_arr(const int data_arr_size)
			{return new Value[data_arr_size];}
			
		static Value * create_data_arr_by_copy(const int data_arr_size, const Value * data_pointer)
			{return static_cast<Value *>(memcpy(new Value[data_arr_size], data_pointer, data_arr_size*sizeof(Value)));}

		void set_size(const int _size[2])
			{data_size = (size[0] = _size[0]) * (size[1] = _size[1]);}

		void fill(const Value &number)
			{memset(data, number, data_size*sizeof(Value));}
					
		// CONSTRUCTORS
		
		Matrix()\
			: size({0,0}), data_size(0), data(nullptr) {};
		Matrix(const int _size[2])\
			: size{_size[0], _size[1]}, data_size(_size[0] * _size[1])
			{data = create_data_arr(data_size);}
		
		// zero-initialized matrix
		static Matrix Zeros(const int _size[2])
			{return Matrix(_size, create_zero_arr(_size[0] * _size[1]));}
			
		// many function use this: they build a data array and work on it before creating the return matrix
		Matrix(const int _size[2], Value * data_pointer)\	
			: size{_size[0], _size[1]}, data_size(_size[0] * _size[1]), data(data_pointer){} // only pointer's copy allowed!!!
					
		// used only in main to write matrice's content with brace-enclosers
		Matrix(const int _size[2], const vector<Value> data_vect)\
			 : size{_size[0], _size[1]}, data_size(_size[0] * _size[1])
			 {data = create_data_arr_by_copy(data_size, &data_vect.front());}
						
		// COPY CONSTRUCTOR AND ASSIGNMENT
		
		Matrix(const Matrix& other)\
			: size{other.size[0], other.size[1]}, data_size(other.size[0] * other.size[1])
			{data = create_data_arr_by_copy(data_size, other.data);}

		Matrix& operator=(const Matrix& other){
			// if receiver matrix is bigger: copy directly into it
			// else receiver matrix is smaller: reallocate data array
			if (data_size < other.data_size){ 
				delete[] data;				
				data = create_data_arr_by_copy(other.data_size, other.data);	
			}else	
				memcpy(data, other.data, other.data_size * sizeof(Value));				
			set_size(other.size);
			return *this;
		}
		Matrix& operator=(const Matrix&& other){
			if (data_size < other.data_size){ 
				delete[] data;				
				data = create_data_arr_by_copy(other.data_size, other.data);	
			}else
				memcpy(data, other.data, other.data_size * sizeof(Value));
			set_size(other.size);
			return *this;
		}

		// FUNCTIONS WITH IMPLICT DATA COPY
		// no "inp" version (inplace)

		// transpose
		Matrix T () const{
			if (size[0] == 1 or size[1] == 1){
			// vector matrix
				return Matrix(new int[2]{size[1], size[0]}, create_data_arr_by_copy(data_size, data));
			} else if (size[0] != size[1]){	
			// rectangular matrix
				Value * data2 = create_data_arr(data_size);				
				const int s = size[1], middle = data_size - s - 1;
				unsigned int j = 0;
				while(true){
					* data2++ = data[j];		
					if (j < middle) j += s;
					else if (j > middle) j -= middle;
					else break;
				}
				*data2 = data[j+=s];
				return Matrix(new int[2]{size[1], size[0]}, data2 - j);
			} else {
			// square matrix
				Value * data2 = create_data_arr_by_copy(data_size, data);
				for (unsigned int i = 0; i != size[0]-1; i++)
					for(unsigned int i1 = i*(size[0]+1)+1, i2 = i+(i+1)*size[0]; i2 < data_size-1; i1++, i2 += size[0])	
						swap(data2[i1], data2[i2]);							
				return Matrix(size, data2);			
			}
		}
		// matrix multiplication
		Matrix mat_mul (const Matrix &b) const{
			Value * data2 = create_zero_arr(size[0] * b.size[1]);
			for (unsigned int i = 0; i != size[0]; i++){
				for (unsigned int j = 0; j != size[1]; j++){
					const Value * i_a = data+i*size[1]+j;
					Value * i_c = data2+i*b.size[1];
					Value * i_b = b.data+j*b.size[1], * i_b_max = i_b+b.size[1];
					while(i_b < i_b_max)
						* (i_c++) += * (i_a) * * (i_b++);	
				}
			}
			return Matrix(new int[2]{size[0], b.size[1]}, data2);
		}
		
		// FUNCTIONS WITH NO IMPLICT DATA COPY
		// there is also a faster inplace version
		
		Matrix scal_mul(const Value b) const{
			Value * data2 = create_data_arr(data_size);
			transform(data, data+data_size, data2, bind(multiplies<Value>(), placeholders::_1, b));
			return Matrix(size, data2);
		}
		Matrix dot_mul(const Matrix& b) const{
			Value * data2 = create_data_arr(data_size);
			transform(data, data+data_size, b.data, data2, multiplies<Value>());
			return Matrix(size, data2);
		}
		Matrix div(const Matrix& b) const{
			Value * data2 = create_data_arr(data_size);
			transform(data, data+data_size, b.data, data2, divides<Value>());
			return Matrix(size, data2);
		}	
		Matrix div(const Value b) const{
			Value * data2 = create_data_arr(data_size);
			transform(data, data+data_size, data2, bind(divides<Value>(), placeholders::_1, b));
			return Matrix(size, data2);
		}	
		Matrix sum(const Matrix & b) const{
			Value * data2 = create_data_arr(data_size);
			transform(data, data+data_size, b.data, data2, plus<Value>());
			return Matrix(size, data2);
		}
		Matrix sum(const Value b) const{
			Value * data2 = create_data_arr(data_size);
			transform(data, data+data_size, data2, bind(plus<Value>(), placeholders::_1, b));
			return Matrix(size, data2);
		}
		Matrix minus(const Matrix & b) const{
			Value * data2 = create_data_arr(data_size);
			transform(data, data+data_size, b.data, data2, std::minus<Value>());
			return Matrix(size, data2);
		}
		Matrix minus(const Value b) const{
			Value * data2 = create_data_arr(data_size);
			transform(data, data+data_size, data2, bind(std::minus<Value>(), placeholders::_1, b));
			return Matrix(size, data2);
		}
		Matrix apply(const Function f) const{
			Value * data2 = create_data_arr(data_size);
			transform(data, data+data_size, data2, f);
			return Matrix(size, data2);
		}
		Matrix pow(const Value b) const{
			Value * data2 = create_data_arr(data_size);
			transform(data, data+data_size, data2, [b](Value d) {return powf(d, b);});
			return Matrix(size, data2);
		}
		Matrix sqrt() const{
			Value * data2 = create_data_arr(data_size);
			transform(data, data+data_size, data2, [](Value d) {return std::sqrt(d);});
			return Matrix(size, data2);
		}
		
		// INPLACE VERSIONS

		void inp_dot_mul(const Matrix &b){transform(data, data+data_size, b.data, data, multiplies<Value>());}
		void inp_div(const Matrix &b){transform(data, data+data_size, b.data, data, divides<Value>());}		
		void inp_scal_mul(const Value b){transform(data, data+data_size, data, bind(multiplies<Value>(), placeholders::_1, b));}
		void inp_div(const Value b){transform(data, data+data_size, data, bind(divides<Value>(), placeholders::_1, b));}
		void inp_sum(const Matrix &b){transform(data, data+data_size, b.data, data, plus<Value>());}
		void inp_minus(const Matrix &b){transform(data, data+data_size, b.data, data, std::minus<Value>());}
		void inp_pow(const Value b){transform(data, data+data_size, data, [b](Value d) {return powf(d, b);});}		
		void inp_apply(const Function f){transform(data, data+data_size, data, f);}		
		
		// OTHER FUNCTIONS
		
		// average matrices (used in batch learning)
		static Matrix avg(const vector<Matrix>&a){
			const int data_size = a[0].data_size;
			Value * data2 = Matrix::create_zero_arr(data_size); 
			for (const Matrix &ai : a) transform(ai.data, ai.data+data_size, data2, data2, plus<Value>());	
			transform(data2, data2+data_size, data2, bind(divides<Value>(), placeholders::_1, a.size()));
			return Matrix(a[0].size, data2);
		}
		
		// base printing
		void print(bool beautiful = true) const {
			stringstream ss;
			string line = string(4 * size[1] - 1, '-');
			for (int i = 0; i != data_size; i++){
				ss << data[i] << ((beautiful) ? " | " : " ");
				if ((i+1) % size[1] == 0){
					if (beautiful) ss << endl << line << endl;
					else ss << endl;
				}
			}
			cout << ss.str() << endl;
		}
		
		// DESTRUCTOR
		
		~Matrix(){delete [] data;}
};
	
static Matrix null_mat;