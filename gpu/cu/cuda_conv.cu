#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "gpumat.h"
#include "cuda_common.h"
#include "common_types.h"

using namespace gpumat;

///////// begin internal namespace ///////////////

namespace gpumat{

namespace internal{


template< typename T >
inline __device__ T empty(T val)
{
	return val;
}

template< typename T >
inline __device__ T reLu(T val)
{
	return max(val, T(0));
}

template< typename T >
inline __device__ T deriv_reLu(T val)
{
	return val > 0? T(1) : T(0);
}

////////////

template< typename T >
__global__ void conv2d(Mtx A0, SmallMtxArray W, SmallMtxArray A1,
					   ct::Size szI, ct::Size szO, int stride,
					   SmallSingleArray<T> B, etypefunction func)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	typedef T (*func_t)(T val);

	func_t _func = empty;
	switch (func) {
		case RELU:
			_func = reLu;
			break;
	}

	if(row < A1.mtx[0].rows && col < A1.mtx[0].cols){
		int yr = col / szO.width;
		int xr = col - yr * szO.width;

		int x = xr * stride;
		int y = yr * stride;

		T *dA0 = (T*)A0.data;
		T *dA0i = &dA0[row * A0.cols];

		for(int w = 0; w < W.count; ++w){
			Mtx& Wi = W.mtx[w];
			Mtx A1I = A1.mtx[w];
			T *dA1 = (T*)A1I.data;
			T *dA1i = &dA1[row * A1I.cols];

			T *dW = (T*)Wi.data;
			T sum = 0;
			for(int a = 0; a < Wi.rows; ++a){
				if(y + a < szI.height){
					for(int b = 0; b < Wi.cols; ++b){
						if(x + b < szI.width){
							sum += dA0i[(y + a) * szI.width + (x + b)] * dW[a * Wi.cols + b];
						}
					}
				}
			}

			sum += B.values[w];
			sum = _func(sum);
			dA1i[col] = sum;
		}
	}
}

template< typename T >
__global__ void subsample(Mtx A0, Mtx A1, Mtx Mask, ct::Size szA0, ct::Size szA1)
{
	const int kLen = 2;

	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(row < A1.rows && col < A1.cols){
		int yr = col / szA1.width;
		int xr = col - yr * szA1.width;

		int x = xr * kLen;
		int y = yr * kLen;

		T *dA0 = (T*)A0.data;
		T *dM = (T*)Mask.data;
		T *dA1 = (T*)A1.data;

		T *dA0i = &dA0[row * A0.cols];
		T *dMi = &dM[row * Mask.cols];
		T *dA1i = &dA1[row * A1.cols];

		T maximum = -99999999;
		int xm = -1, ym = -1;
		for(int a = 0; a < kLen; ++a){
			if(y + a < szA0.height){
				for(int b = 0; b < kLen; ++b){
					if(x + b < szA0.width){
						dMi[(y + a) * szA0.width + (x + b)] = 0;
						T val = dA0i[(y + a) * szA0.width + (x + b)];
						if(val > maximum){
							xm = x + b; ym = y + b;
							maximum = val;
						}
					}
				}
			}
		}
		if(xm >= 0 && ym >= 0){
			dMi[ym * szA0.width + xm] = 1;
			dA1i[yr * szA1.width + xr] = maximum;
		}
	}
}

template< typename T >
__global__ void upsample(Mtx A1, Mtx Mask, Mtx A0, ct::Size szA1, ct::Size szA0)
{
	const int kLen = 2;

	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(row < A1.rows && col < A1.cols){
		int yr = col / szA1.width;
		int xr = col - yr * szA1.width;

		int x = xr * kLen;
		int y = yr * kLen;

		T *dA0 = (T*)A0.data;
		T *dM = (T*)Mask.data;
		T *dA1 = (T*)A1.data;

		T *dA0i = &dA0[row * A0.cols];
		T *dMi = &dM[row * Mask.cols];
		T *dA1i = &dA1[row * A1.cols];

		T val = dA1i[yr * szA1.width + xr];

		for(int a = 0; a < kLen; ++a){
			if(y + a < szA0.height){
				for(int b = 0; b < kLen; ++b){
					if(x + b < szA0.width){
						T mask = dMi[(y + a) * szA0.width + (x + b)];
						dA0i[(y + a) * szA0.width + (x + b)] = val * mask;
					}
				}
			}
		}
	}
}

template< typename T >
__global__ void deriv_conv2d(Mtx A0, Mtx gA1, ct::Size szA0, ct::Size szA1, Mtx gW, int stride, Mtx Blocks)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	extern __shared__ int iW[];
	T *sW = (T*)iW;

	if(row < gA1.rows && col < gA1.cols){
		int y = col / szA1.width;
		int x = col - y * szA1.width;

		int x0 = x * stride;
		int y0 = y * stride;

		T *dA0 = (T*)A0.data;
//		T *dgW = (T*)gW.data;
		T *dgA1 = (T*)gA1.data;

		T *dB = (T*)Blocks.data;

		T *dA0i = &dA0[row * A0.cols];
		T *dgA1i = &dgA1[row * gA1.cols];

		T d = dgA1i[y * szA1.width + x];

		int brow = row * gW.rows;
		int bcol = col * gW.cols;

		T *dBi = &dB[brow * Blocks.cols + bcol];

		for(int a = 0; a < gW.rows; ++a){
			int y1 = y0 + a;
			if(y1 < szA0.height){
				for(int b = 0; b < gW.cols; ++b){
					int x1 = x0 + b;
					if(x1 < szA0.width){
						T a0 = dA0i[y1 * szA0.width + x1];
						dBi[(a) * Blocks.cols + (b)] = d * a0;
					}
				}
			}
		}

	}
}

template< typename T >
__global__ void reduce_blocks(Mtx Blocks, Mtx W, T val = 1.)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(row < W.rows && col < W.cols){
		T *dW = (T*)W.data;
		T *dB = (T*)Blocks.data;

		int ca = Blocks.rows / W.rows;
		int cb = Blocks.cols / W.cols;
		for(int a = 0; a < ca; ++a){
			for(int b = 0; b < cb; ++b){
				int ra = a * W.rows;
				int rb = b * W.cols;

				T *dBi = &dB[ra * Blocks.cols + rb];
				dW[row * W.cols + col] += dBi[row * Blocks.cols + col];
			}
		}
		dW[row * W.cols + col] *= val;
	}

}

template< typename T >
__global__ void deriv_prev_conv2d(SmallMtxArray deriv, SmallMtxArray W,
								  ct::Size sL, ct::Size sLsub1, int stride, Mtx D)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(row < D.rows && col < D.cols){
		int y = col / sLsub1.width;
		int x = col - y * sLsub1.width;

		int x0 = x / stride;
		int y0 = y / stride;

		T *dD = (T*)D.data;

		T *dDi = &dD[row * D.cols];

		T sum = 0;
		for(int w = 0; w < W.count; ++w){
			T *dDrv = (T*)deriv.mtx[w].data;
			T *dDrvi = &dDrv[row * deriv.mtx[w].cols];

			Mtx& Wi = W.mtx[w];
			T *dW = (T*)Wi.data;

			for(int a = 0; a < Wi.rows; ++a){
				int yi = y0 - a;
				if(yi >=0 && yi < sL.height){
					for(int b = 0; b < Wi.cols; ++b){
						int xi = x0 - b;
						if(xi >=0 && xi < sL.width){
							T d = dDrvi[yi * sL.width + xi];
							T w = dW[a * Wi.cols + b];
							sum += d * w;
						}
					}/* W.cols */
				}
			}/* W.rows */
		}/* W */
		dDi[y * sLsub1.width + x] = sum;
	}
}

template< typename T >
__global__ void hsplit(Mtx Res, SmallMtxArray List)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(row < Res.rows && col < Res.cols){
		T *dR =(T*)Res.data;

		int lid = col / List.mtx[0].cols;
		Mtx& mtx = List.mtx[lid];
		int lcol = col - lid * mtx.cols;
		T* dM = (T*)mtx.data;
		dM[row * mtx.cols + lcol] = dR[row * Res.cols + col];
	}
}

template< typename T >
__global__ void hconcat(SmallMtxArray List, Mtx Res)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(row < Res.rows && col < Res.cols){
		T *dR =(T*)Res.data;

		int lid = col / List.mtx[0].cols;
		Mtx& mtx = List.mtx[lid];
		int lcol = col - lid * mtx.cols;
		T* dM = (T*)mtx.data;
		dR[row * Res.cols + col] = dM[row * mtx.cols + lcol];
	}
}

}/*@internal end*/

}/*@gpumat end*/

///////////

extern "C"
void cuda_conv2d(const GpuMat &A0,
				 const ct::Size &szI, const ct::Size &szO,
				 int stride,
				 const std::vector<GpuMat> &W,
				 const std::vector<float> B,
				 std::vector<GpuMat> &A1,
				 etypefunction func)
{
	int x1 = A1[0].cols / BLOCKSIZE + 1;
	int x2 = A1[0].rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::SmallMtxArray sW(W), sA1(A1);

	switch (A0.type) {
		case GPU_DOUBLE:{
			internal::SmallSingleArray<double> sB(B);
			internal::conv2d<double> <<<dimGrid, dimBlock>>>(A0, sW, sA1, szI, szO, stride, sB, func);
			break;
		}
		case GPU_FLOAT:{
			internal::SmallSingleArray<float> sB(B);
			internal::conv2d<float> <<<dimGrid, dimBlock>>>(A0, sW, sA1, szI, szO, stride, sB, func);
			break;
		}
	}

}

extern "C"
void cuda_subsample(const GpuMat &A0,
					const ct::Size &szA0,
					const ct::Size &szA1,
					GpuMat &A1,
					GpuMat &Mask)
{
	int x1 = A1.cols / BLOCKSIZE + 1;
	int x2 = A1.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A0.type) {
	case GPU_DOUBLE:
		internal::subsample<double> <<<dimGrid, dimBlock>>>(A0, A1, Mask, szA0, szA1);
		break;
	case GPU_FLOAT:
		internal::subsample<float> <<<dimGrid, dimBlock>>>(A0, A1, Mask, szA0, szA1);
		break;
	}
}

extern "C"
void cuda_upsample(const GpuMat &A1,
				   const ct::Size &szA1,
				   const ct::Size &szA0,
				   const GpuMat &Mask,
				   GpuMat &A0)
{
	int x1 = A1.cols / BLOCKSIZE + 1;
	int x2 = A1.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A0.type) {
		case GPU_DOUBLE:
		internal::upsample<double> <<<dimGrid, dimBlock>>>(A1, Mask, A0, szA1, szA0);
		break;
	case GPU_FLOAT:
		internal::upsample<float> <<<dimGrid, dimBlock>>>(A1, Mask, A0, szA1, szA0);
		break;
	}
}

template< typename T >
void cuda_reduce_blocks(const GpuMat& Blocks, GpuMat& W, T val)
{
	int x1 = W.cols / BLOCKSIZE + 1;
	int x2 = W.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::reduce_blocks<T> <<< dimGrid, dimBlock >>>(Blocks, W, val);
}

extern "C"
void cuda_deriv_conv2d(const GpuMat &A0, const GpuMat &gradA1,
				  const ct::Size &szA0, const ct::Size &szA1,
				  int stride,
				  GpuMat &gradW, float &gradB)
{
	int x1 = gradA1.cols / BLOCKSIZE + 1;
	int x2 = gradA1.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	gpumat::GpuMat blocks(gradA1.rows * gradW.rows, gradA1.cols * gradW.cols, gradW.type);
	gpumat::memset(blocks, 0);

	switch (A0.type) {
		case GPU_DOUBLE:{
			internal::deriv_conv2d<double> <<<dimGrid, dimBlock, gradW.size() >>>(A0, gradA1, szA0, szA1,
																   gradW, stride, blocks);
			cuda_reduce_blocks<double>(blocks, gradW, 1./gradA1.rows);
			double val = thrust::reduce(thrust::device, (double*)gradA1.data, (double*)gradA1.data + gradA1.total());
			val /= gradA1.total();
			gradB = val;
			break;
		}
		case GPU_FLOAT:{
			internal::deriv_conv2d<float> <<<dimGrid, dimBlock, gradW.size() >>>(A0, gradA1, szA0, szA1,
																  gradW, stride, blocks);
//			std::cout << blocks.print() << std::endl << A0.print() << std::endl << gradA1.print() << std::endl;
			cuda_reduce_blocks<float>(blocks, gradW, 1./gradA1.rows);
			float val = thrust::reduce(thrust::device, (float*)gradA1.data, (float*)gradA1.data + gradA1.total());
			val /= gradA1.total();
			gradB = val;
			break;
		}
	}
//	gpumat::mulval(gradW, (double)1./gradA1.rows);

}

extern "C"
void cuda_deriv_prev_conv2d(const std::vector<GpuMat> &deriv,
							const std::vector<GpuMat> &W,
							const ct::Size &sL, const ct::Size &sLsub1, int stride,
							GpuMat &D)
{
	int x1 = D.cols / BLOCKSIZE + 1;
	int x2 = D.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::SmallMtxArray sderiv(deriv);
	internal::SmallMtxArray sW(W);

	switch (D.type) {
	case GPU_DOUBLE:
		internal::deriv_prev_conv2d<double> <<<dimGrid, dimBlock>>>(sderiv, sW, sL, sLsub1, stride, D);
		break;
	case GPU_FLOAT:
		internal::deriv_prev_conv2d<float> <<<dimGrid, dimBlock>>>(sderiv, sW, sL, sLsub1, stride, D);
		break;
	}
}

extern "C"
void cuda_hsplit(const GpuMat &res, std::vector<GpuMat> &list)
{
	int x1 = res.cols / BLOCKSIZE + 1;
	int x2 = res.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::SmallMtxArray slist(list);

	switch (res.type) {
	case GPU_DOUBLE:
		internal::hsplit<double> <<<dimGrid, dimBlock>>>(res, slist);
		break;
	case GPU_FLOAT:
		internal::hsplit<float> <<<dimGrid, dimBlock>>>(res, slist);
		break;
	}

}

extern "C"
void cuda_hconcat(const std::vector<GpuMat> &list, GpuMat &res)
{
	int x1 = res.cols / BLOCKSIZE + 1;
	int x2 = res.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::SmallMtxArray slist(list);

	switch (res.type) {
	case GPU_DOUBLE:
		internal::hconcat<double> <<<dimGrid, dimBlock>>>(slist, res);
		break;
	case GPU_FLOAT:
		internal::hconcat<float> <<<dimGrid, dimBlock>>>(slist, res);
		break;
	}
}
