#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "gpumat.h"

using namespace gpumat;

///////// begin internal namespace ///////////////

namespace internal{

////////////////////////////////

struct Mtx{
	int rows;
	int cols;
	u_char* data;

	Mtx(){
		rows = cols = 0;
		data = 0;
	}
	Mtx(int rows, int cols, void* data){
		this->rows = rows;
		this->cols = cols;
		this->data = (u_char*)data;
	}
	Mtx(const GpuMat& mat){
		rows = mat.rows;
		cols = mat.cols;
		data = mat.data;
	}
};

#define BLOCKSIZE	32

/**
 * @brief memset
 * @param A = val
 * @param val

 */
template< class T >
__global__ void memset(Mtx A, T val)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] = val;
}

/**
 * @brief add
 * @param A
 * @param B
 * @param C - out C = A .+ B
 */
template< class T >
__global__ void add(Mtx A, Mtx B, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * A.cols + col] = dA[row * A.cols + col] + dB[row * B.cols + col];
}

/**
 * @brief add
 * @param A
 * @param B
 * @param C - out C = val1 * A .+ val2 * B
 */
template< class T >
__global__ void add(Mtx A, Mtx B, T valA, T valB, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * A.cols + col] = valA * dA[row * A.cols + col] + valB * dB[row * B.cols + col];
}

/**
 * @brief add
 * @param A
 * @param B
 * @param C - out C = A .+ B
 */
template< class T >
__global__ void add(Mtx A, Mtx B, T valA, T valB)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] = valA * dA[row * A.cols + col] + valB * dB[row * B.cols + col];
}

/**
 * @brief sub
 * @param A
 * @param B
 * @param C -> C = valA * A - valB * B
 * @param valA
 * @param valB
 */template< class T >
__global__ void sub(Mtx A, Mtx B, Mtx C, T valA, T valB)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = valA * dA[row * A.cols + col] - valB * dB[row * B.cols + col];
}

template< class T >
__global__ void sub(Mtx A, Mtx B, T valA, T valB)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] = valA * dA[row * A.cols + col] - valB * dB[row * B.cols + col];
}


/**
 * @brief matmul
 * @param A
 * @param B
 * @param C - out C = A * B
 */
template< class T >
__global__ void matmul(Mtx A, Mtx B, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* DA = (T*)A.data;
	T* DB = (T*)B.data;
	T* DC = (T*)C.data;

	float sC = 0;

	if(row < A.rows && col < B.cols){
		for(int i = 0; i < B.rows; i++){
			sC += DA[row * A.cols + i] * DB[i * B.cols + col];
		}
		DC[row * B.cols + col] = sC;
	}
}

/**
 * @brief matmulT1
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C = A' * B
 */
template< class T >
__global__ void matmulT1(Mtx At, Mtx B, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* DA = (T*)At.data;
	T* DB = (T*)B.data;
	T* DC = (T*)C.data;

	float sC = 0;

//	s += val1[j * At.cols + i]/*at(i, j)*/ * val2[j * B.cols + k];
	if(row < At.cols && col < B.cols){
		for(int i = 0; i < B.rows; i++){
			sC += DA[i * At.cols + row] * DB[i * B.cols + col];
		}
		DC[row * C.cols + col] = sC;
	}

}

/**
 * @brief matmulT2
 * @param A
 * @param Bt - used as transposed matrix
 * @param C - out C = A * B'
 */
template< class T >
__global__ void matmulT2(Mtx A, Mtx Bt, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* DA = (T*)A.data;
	T* DB = (T*)Bt.data;
	T* DC = (T*)C.data;

	float sC = 0;

//	s += val1[i * A.cols + j]/*at(i, j)*/ * val2[k * Bt.cols + j]/*at(j, k)*/;
	if(row < A.rows && col < Bt.rows){
		for(int i = 0; i < Bt.cols; i++){
//			sC += DA[row * B.rows + i] * DB[i * B.cols + col];
			sC += DA[row * A.cols + i] * DB[col * Bt.cols + i];
		}
		DC[row * C.cols + col] = sC;
	}
}

/**
 * @brief mulval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A * value
 */
template< class T >
__global__ void mulval(Mtx A, T value, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA =(T*) A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = dA[row * A.cols + col] * value;
}

/**
 * @brief mulval
 * @param A -> A = A * value
 * @param value - mat 1x1
 */
template< class T >
__global__ void mulval(Mtx A, double value)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA =(T*) A.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] *= value;
}

/**
 * @brief addval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A + value
 */
template< class T >
__global__ void addval(Mtx A, T value, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = dA[row * A.cols + col] + value;
}

/**
 * @brief addval
 * @param A -> A += value
 * @param value - mat 1x1
 */
template< class T >
__global__ void addval(Mtx A, T value)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] += value;
}

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A - value
 */
template< class T >
__global__ void subval(Mtx A, T value, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = dA[row * A.cols + col] - value;
}

/**
 * @brief subval
 * @param A -> A - val
 * @param value - mat 1x1
 */
template< class T >
__global__ void subval(Mtx A, T value)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] -= value;
}

/**
 * @brief subval
 * @param A -> A - val
 * @param value - mat 1x1
 */
template< class T >
__global__ void subval_inv(Mtx A, T value)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] = value - dA[row * A.cols + col];
}

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = value - C
 */
template< class T >
__global__ void subval(T value, Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA =(T*) A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = value - dA[row * A.cols + col];

}

/**
 * @brief biasPlus
 * @param A - out A[i] = A[i] + bias
 * @param bias
 */
template< class T >
__global__ void biasPlus(Mtx A, const Mtx bias)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dBias = (T*)bias.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] += dBias[col];

}

/**
 * @brief elemwiseMul
 * @param A
 * @param B
 * @param C - out C = A .* B
 */
template< class T >
__global__ void elemwiseMul(Mtx A, Mtx B, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = dA[row * A.cols + col] * dB[row * B.cols + col];
}

/**
 * @brief elemwiseMul
 * @param A
 * @param B
 */
template< class T >
__global__ void elemwiseMul(Mtx A, Mtx B)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] *= dB[row * A.cols + col];
}

/**
 * @brief elemwiseDiv
 * @param A
 * @param B
 * @param C - out C = A ./ B
 */
template< class T >
__global__ void elemwiseDiv(Mtx A, Mtx B, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = dA[row * A.cols + col] / dB[row * B.cols + col];
}

/**
 * @brief transpose
 * @param A
 * @param C = A'
 */
template< class T >
__global__ void transpose(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[col * C.cols + row] = dA[row * A.cols + col];
}

/**
 * @brief sqrt
 * @param A
 * @param C = sqrt(A)
 */
template< class T >
__global__ void sqrt(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = std::sqrt(dA[row * A.cols + col]);
}

/**
 * @brief sqr
 * @param A
 * @param C = A .* a
 */
template< class T >
__global__ void sqr(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols){
		T val = dA[row * A.cols + col];
		dC[row * C.cols + col] = val * val;
	}
}

/**
 * @brief reLu
 * @param A
 * @param C = reLu(A)
 */
template< class T >
__global__ void reLu(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = max(dA[row * A.cols + col], 0.);
}

/**
 * @brief deriv_reLu
 * @param A
 * @param C = reLu(A)
 */
template< class T >
__global__ void deriv_reLu(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = (dA[row * A.cols + col] > 0)? 1 : 0;
}

/**
 * @brief _exp
 * @param A
 * @param C = exp(A)
 */
template< class T >
__global__ void _exp(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols){
		T val = exp(dA[row * A.cols + col]);
		dC[row * C.cols + col] = val;
	}
}

/**
 * @brief sum_col
 * @param A
 * @param C = exp(A)
 * @param rows = sum(C)
 */
template< class T >
__global__ void sum_col(Mtx C, Mtx cols, T val = (T)1.)
{
	//int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dC = (T*)C.data;
	T* dZ = (T*)cols.data;

	if(col < C.cols){
		dZ[col] = 0;
		for(int i = 0; i < C.rows; i++){
			dZ[col] += dC[i * C.cols + col];
		}
		dZ[col] *= val;
	}
}
/**
 * @brief sum_row
 * @param A
 * @param C = exp(A)
 * @param rows = sum(C)
 */
template< class T >
__global__ void sum_row(Mtx C, Mtx rows, T val = (T)1.)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	//int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dC = (T*)C.data;
	T* dZ = (T*)rows.data;

	if(row < C.rows){
		dZ[row] = 0;
		for(int i = 0; i < C.cols; i++){
			dZ[row] += dC[row * C.cols + i];
		}
		dZ[row] *= val;
	}
}

/**
 * @brief div_col
 * @param C -> in/out C[i, j] /=  cols[j]
 * @param cols -> in
 */
template< class T >
__global__ void div_col(Mtx C, Mtx cols)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dC = (T*)C.data;
	T* dZ = (T*)cols.data;

	if(row < C.rows && col < C.cols){
		dC[row * C.cols + col] = dC[row * C.cols + col] / dZ[col];
	}
}

/**
 * @brief div_row
 * @param C -> in/out C[i, j] /=  rows[i]
 * @param rows -> in
 */
template< class T >
__global__ void div_row(Mtx C, Mtx rows)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dC = (T*)C.data;
	T* dZ = (T*)rows.data;

	if(row < C.rows && col < C.cols){
		if(dZ[row] != 0)
			dC[row * C.cols + col] = dC[row * C.cols + col] / dZ[row];
		else
			dC[row * C.cols + col] = 0;
	}
}

/**
 * @brief cuda_adamgrad
 * @param A = -alpha * (sb1 * mA / (sqrt(sb2 * vA) + eps)
 * @param mA
 * @param vA
 * @param alpha
 * @param sb1
 * @param sb2
 */
template< class T >
__global__ void adamgrad(Mtx A, const Mtx mA, const Mtx vA, T alpha, T sb1, T sb2)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	const T eps = 10e-8;

	T* dA = (T*)A.data;
	T* dmA = (T*)mA.data;
	T* dvA = (T*)vA.data;
	if(row < A.rows && col < A.cols){
		T m = sb1 * dmA[row * A.cols + col];
		T v = sb2 * dvA[row * A.cols + col];
		T val = alpha * m / (::sqrt(v) + eps);
		dA[row * A.cols + col] -= val;
	}
}

}

//////// end namespace /////////////////

/**
 * @brief cuda_memset
 * @param A = val
 * @param val
 */
extern "C"
void cuda_memset(GpuMat& A, double val)
{
	if(A.empty())
		return;

	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::memset<double> <<<dimGrid, dimBlock>>>(A, (double)val);
		break;
	case GPU_FLOAT:
		internal::memset<float> <<<dimGrid, dimBlock>>>(A, (float)val);
		break;
	}
}

/**
 * @brief add
 * @param A
 * @param B
 * @param C - out C = A .+ B
 */
extern "C"
void cuda_add(const GpuMat& A, const GpuMat& B, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::add<double> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	case GPU_FLOAT:
		internal::add<float> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	}
}

/**
 * @brief cuda_add_params
 * @param A
 * @param val1
 * @param B
 * @param val2
 * @param C = val1 * A + val2 * B
 */
extern "C"
void cuda_add_params(const GpuMat& A, const GpuMat& B, double val1, double val2, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::add<double> <<<dimGrid, dimBlock>>>(A, B, (double)val1, (double)val2, C);
		break;
	case GPU_FLOAT:
		internal::add<float> <<<dimGrid, dimBlock>>>(A, B, (float)val1, (float)val2, C);
		break;
	}
}

/**
 * @brief cuda_add_paramsA
 * @param A -> A = val1 * A + val2 * B
 * @param val
 * @param B
 */
extern "C"
void cuda_add_paramsA(GpuMat& A, const GpuMat& B, double valA, double valB)
{
	int x1 = B.cols / BLOCKSIZE + 1;
	int x2 = B.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (B.type) {
	case GPU_DOUBLE:
		internal::add<double> <<<dimGrid, dimBlock>>>(A, B, (double)valA, (double)valB);
		break;
	case GPU_FLOAT:
		internal::add<float> <<<dimGrid, dimBlock>>>(A, B, (float)valA, (float)valB);
		break;
	}
}

/**
 * @brief sub
 * @param A
 * @param B
 * @param C - out C = A .- B
 */
extern "C"
void cuda_sub(const GpuMat& A, const GpuMat& B, GpuMat& C, double valA, double valB)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::sub<double> <<<dimGrid, dimBlock>>>(A, B, C, (double)valA, (double)valB);
		break;
	case GPU_FLOAT:
		internal::sub<float> <<<dimGrid, dimBlock>>>(A, B, C, (float)valA, (float)valB);
		break;
	}
}

/**
 * @brief cuda_subA
 * @param A = A * valA - B * valB
 * @param B
 * @param valA
 * @param valB
 */
extern "C"
void cuda_subA(GpuMat& A, const GpuMat& B, double valA, double valB)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::sub<double> <<<dimGrid, dimBlock>>>(A, B, (double)valA, (double)valB);
		break;
	case GPU_FLOAT:
		internal::sub<float> <<<dimGrid, dimBlock>>>(A, B, (float)valA, (float)valB);
		break;
	}
}

/**
 * @brief matmul
 * @param A
 * @param B
 * @param C - out C = A * B
 */
extern "C"
void cuda_matmul(const GpuMat& A, const GpuMat& B, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::matmul<double> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	case GPU_FLOAT:
		internal::matmul<float> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	}
}

/**
 * @brief matmulT1
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C = A' * B
 */
extern "C"
void cuda_matmulT1(const GpuMat& At, const GpuMat& B, GpuMat& C)
{
	//	int r = At.cols;
	//	int c = B.cols;

	int x1 = B.cols / BLOCKSIZE + 1;
	int x2 = At.cols / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (At.type) {
	case GPU_DOUBLE:
		internal::matmulT1<double> <<<dimGrid, dimBlock>>>(At, B, C);
		break;
	case GPU_FLOAT:
		internal::matmulT1<float> <<<dimGrid, dimBlock>>>(At, B, C);
		break;
	}
}

/**
 * @brief matmulT2
 * @param A
 * @param Bt - used as transposed matrix
 * @param C - out C = A * B'
 */
extern "C"
void cuda_matmulT2(const GpuMat& A, const GpuMat& Bt, GpuMat& C)
{
	int x1 = Bt.rows / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::matmulT2<double> <<<dimGrid, dimBlock>>>(A, Bt, C);
		break;
	case GPU_FLOAT:
		internal::matmulT2<float> <<<dimGrid, dimBlock>>>(A, Bt, C);
		break;
	}
}

/**
 * @brief mulval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A * value
 */
extern "C"
void cuda_mulval(const GpuMat& A, double value, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::mulval<double> <<<dimGrid, dimBlock>>>(A, (double)value, C);
		break;
	case GPU_FLOAT:
		internal::mulval<float> <<<dimGrid, dimBlock>>>(A, (float)value, C);
	}
}

/**
 * @brief mulval
 * @param A -> A *= value
 * @param value - mat 1x1
 */
extern "C"
void cuda_mulvalA(const GpuMat& A, double value)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::mulval<double> <<<dimGrid, dimBlock>>>(A, (double)value);
		break;
	case GPU_FLOAT:
		internal::mulval<float> <<<dimGrid, dimBlock>>>(A, (float)value);
	}
}

/**
 * @brief mulval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A * value
 */
extern "C"
void cuda_mulval_in(GpuMat& A, const double value)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::mulval<double> <<<dimGrid, dimBlock>>>(A, (double)value);
		break;
	case GPU_FLOAT:
		internal::mulval<float> <<<dimGrid, dimBlock>>>(A, (float)value);
		break;
	}
}

/**
 * @brief addval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A + value
 */
extern "C"
void cuda_addval(const GpuMat& A, double value, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::addval<double> <<<dimGrid, dimBlock>>>(A, (double)value, C);
		break;
	case GPU_FLOAT:
		internal::addval<float> <<<dimGrid, dimBlock>>>(A, (float)value, C);
		break;
	}
}

/**
 * @brief addval
 * @param A -> A += val
 * @param value - mat 1x1
 */
extern "C"
void cuda_addvalA(GpuMat& A, double value)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::addval<double> <<<dimGrid, dimBlock>>>(A, (double)value);
		break;
	case GPU_FLOAT:
		internal::addval<float> <<<dimGrid, dimBlock>>>(A, (float)value);
		break;
	}
}

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A - value
 */
extern "C"
void cuda_subval_AvaltoC(const GpuMat& A, double value, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::subval<double> <<<dimGrid, dimBlock>>>(A, (double)value, C);
		break;
	case GPU_FLOAT:
		internal::subval<float> <<<dimGrid, dimBlock>>>(A, (float)value, C);
		break;
	}
}

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = value - C
 */
extern "C"
void cuda_subval_valAtoC(double value, const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::subval<double> <<<dimGrid, dimBlock>>>((double)value, A, C);
		break;
	case GPU_FLOAT:
		internal::subval<float> <<<dimGrid, dimBlock>>>((float)value, A, C);
		break;
	}
}

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 */
extern "C"
void cuda_subval_Aval(GpuMat& A, double value)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::subval<double> <<<dimGrid, dimBlock>>>(A, (double)value);
		break;
	case GPU_FLOAT:
		internal::subval<float> <<<dimGrid, dimBlock>>>(A, (float)value);
		break;
	}
}

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 */
extern "C"
void cuda_subval_valA(GpuMat& A, double value)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::subval_inv<double> <<<dimGrid, dimBlock>>>(A, (double)value);
		break;
	case GPU_FLOAT:
		internal::subval_inv<float> <<<dimGrid, dimBlock>>>(A, (float)value);
		break;
	}
}

/**
 * @brief biasPlus
 * @param A - out A[i] = A[i] + bias
 * @param bias
 */
extern "C"
void cuda_biasPlus(GpuMat& A, const GpuMat& bias)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::biasPlus<double> <<<dimGrid, dimBlock>>>(A, bias);
		break;
	case GPU_FLOAT:
		internal::biasPlus<float> <<<dimGrid, dimBlock>>>( A, bias);
		break;
	}
}

/**
 * @brief elemwiseMul
 * @param A
 * @param B
 * @param C - out C = A .* B
 */
extern "C"
void cuda_elemwiseMul(const GpuMat& A, const GpuMat& B, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::elemwiseMul<double> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	case GPU_FLOAT:
		internal::elemwiseMul<float> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	}
}

/**
 * @brief elemwiseMul
 * @param A = A .* B
 * @param B
 */
extern "C"
void cuda_elemwiseMulA(GpuMat& A, const GpuMat& B)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::elemwiseMul<double> <<<dimGrid, dimBlock>>>(A, B);
		break;
	case GPU_FLOAT:
		internal::elemwiseMul<float> <<<dimGrid, dimBlock>>>(A, B);
		break;
	}
}


/**
 * @brief elemwiseDiv
 * @param A
 * @param B
 * @param C - out C = A ./ B
 */
extern "C"
void cuda_elemwiseDiv(const GpuMat& A, const GpuMat& B, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::elemwiseDiv<double> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	case GPU_FLOAT:
		internal::elemwiseDiv<float> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	}
}

/**
 * @brief cuda_transpose
 * @param A
 * @param C = A'
 */
extern "C"
void cuda_transpose(const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::transpose<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	case GPU_FLOAT:
		internal::transpose<float> <<<dimGrid, dimBlock>>>(A, C);
		break;
	}
}

/**
 * @brief cuda_elemwiseSqrt
 * @param A
 * @param C = sqrt(A)
 */
extern "C"
void cuda_elemwiseSqrt(const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::sqrt<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	case GPU_FLOAT:
		internal::sqrt<float> <<<dimGrid, dimBlock>>>(A, C);
		break;
	}
}

/**
 * @brief cuda_elemwiseSqr
 * @param A
 * @param C =  A .* a
 */
extern "C"
void cuda_elemwiseSqr(const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::sqr<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	case GPU_FLOAT:
		internal::sqr<float> <<<dimGrid, dimBlock>>>(A, C);
		break;
	}
}

/**
 * @brief cuda_sumrows
 * @param A
 * @param C - out C[i] = sum(A[i, j])(j = [1..cols])
 */
extern "C"
void cuda_sumrows(const GpuMat& A, GpuMat& sums, double val)
{
	int x1 = A.cols / BLOCKSIZE + 1;
//	int x2 = A.rows / BLOCKSIZE + 1;

	switch (A.type) {
	case GPU_DOUBLE:
			internal::sum_col<double> <<<dim3(x1, 1), dim3(BLOCKSIZE, 1)>>>(A, sums, (double)val);
		break;
	case GPU_FLOAT:
			internal::sum_col<float> <<<dim3(x1, 1), dim3(BLOCKSIZE, 1)>>>(A, sums, (float)val);
		break;
	}
}

/**
 * @brief cuda_reLu
 * @param A
 * @param C = reLu(A)
 */
extern "C"
void cuda_reLu(const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::reLu<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	case GPU_FLOAT:
		internal::reLu<float> <<<dimGrid, dimBlock>>>(A, C);
		break;
	}
}

/**
 * @brief cuda_derivReLu
 * @param A
 * @param C = reLu(A)
 */
extern "C"
void cuda_derivReLu(const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::deriv_reLu<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	case GPU_FLOAT:
		internal::deriv_reLu<float> <<<dimGrid, dimBlock>>>(A, C);
		break;
	}
}

/**
 * @brief cuda_softmax
 * @param A
 * @param axis -> 0 - in row, 1 - in col
 * @param C = softmax(A)
 */
extern "C"
void cuda_softmax(const GpuMat& A, int axis, GpuMat& C, GpuMat& partZ)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::_exp<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	case GPU_FLOAT:
		internal::_exp<float> <<<dimGrid, dimBlock>>>(A, C);
		break;
	}

	switch (axis) {
		case 0:
			{
				switch (A.type) {
				case GPU_DOUBLE:
						internal::sum_col<double> <<<dim3(x1, 1), dim3(BLOCKSIZE, 1)>>>(C, partZ);
						internal::div_col<double> <<<dimGrid, dimBlock>>>(C, partZ);
					break;
				case GPU_FLOAT:
						internal::sum_col<float> <<<dim3(x1, 1), dim3(BLOCKSIZE, 1)>>>(C, partZ);
						internal::div_col<float> <<<dimGrid, dimBlock>>>(C, partZ);
					break;
				}
			}
			break;
		case 1:
			{
				switch (A.type) {
				case GPU_DOUBLE:
						internal::sum_row<double> <<<dim3(1, x2), dim3(1, BLOCKSIZE)>>>(C, partZ);
						internal::div_row<double> <<<dimGrid, dimBlock>>>(C, partZ);
					break;
				case GPU_FLOAT:
						internal::sum_row<float> <<<dim3(1, x2), dim3(1, BLOCKSIZE)>>>(C, partZ);
						internal::div_row<float> <<<dimGrid, dimBlock>>>(C, partZ);
					break;
				}
			}
			break;
	}
}

/**
 * @brief cuda_adamgrad
 * @param A = -alpha * (sb1 * mA / (sqrt(sb2 * vA) + eps)
 * @param mA
 * @param vA
 * @param alpha
 * @param sb1
 * @param sb2
 */
extern "C"
void cuda_adamgrad(GpuMat& A, const GpuMat& mA, const GpuMat& vA, double alpha, double sb1, double sb2)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::adamgrad<double> <<<dimGrid, dimBlock>>>(A, mA, vA, (double)alpha, (double)sb1, (double)sb2);
		break;
	case GPU_FLOAT:
		internal::adamgrad<float> <<<dimGrid, dimBlock>>>(A, mA, vA, (float)alpha, (float)sb1, (float)sb2);
		break;
	}
}
