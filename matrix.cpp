class Matrix {
public:
	void init();
	Matrix() {
		init();
	}
	Matrix(int _numRows, int _numCols) {
	}
	Matrix(const Matrix& like) {
	}
	Matrix& operator=(const Matrix& like) {
	}
		
public:
	int numRows, numCols;
	int numElts;
public:
	float * h_fPtr;
	int * refCount;
};

void Matrix::init() {
	numCols = 0;
	numRows = 0;
	numElts = 0;
	h_fPtr = nullptr;
	refCount = nullptr;
}
