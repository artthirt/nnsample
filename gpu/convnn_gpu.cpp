#include "convnn_gpu.h"
#include "cuda_common.h"

#include "exception"

using namespace gpumat;

convnn::convnn()
{

}

/////////////////////////////////////////
///////**********************////////////
/////////////////////////////////////////

extern "C"
void cuda_conv2d(const GpuMat &A0,
				 const ct::Size &szI,
				 const ct::Size &szO,
				 int stride,
				 const std::vector<GpuMat> &W,
				 const std::vector<float> B,
				 std::vector<GpuMat> &A1,
				 etypefunction func);


extern "C"
void cuda_subsample(const GpuMat &A0,
					const ct::Size &szA0,
					const ct::Size &szA1,
					GpuMat &A1,
					GpuMat &Mask);

extern "C"
void cuda_upsample(const GpuMat &A1, const ct::Size &szA1,
			  const ct::Size &szA0, const GpuMat &Mask, GpuMat &A0);

extern "C"
void cuda_deriv_conv2d(const GpuMat &A0, const GpuMat &gradA1,
				  const ct::Size &szA0, const ct::Size &szA1,
				  int stride, GpuMat &gradW, float &gradB);

extern "C"
void cuda_deriv_prev_conv2d(std::vector<GpuMat> &deriv,
							const std::vector<GpuMat> &W,
							const ct::Size &sL, const ct::Size &sLsub1, int stride,
							GpuMat &D);


/////////////////////////////

namespace gpumat{

ct::Size conv2D(const GpuMat &A0,
			const ct::Size &szI,
			int stride,
			const std::vector<GpuMat> &W,
			const std::vector<float> B,
			std::vector<GpuMat> &A1,
			etypefunction func)
{
	if(A0.empty() || W.empty() || B.empty())
		throw new std::invalid_argument("gpumat::conv2D: check parameters");

	if(A1.size() != W.size())
		A1.resize(W.size());

	int w_rows = W[0].rows;
	int w_cols = W[0].cols;

	ct::Size szO;
	szO.width	= (szI.width - w_cols + 1) / stride;
	szO.height	= (szI.height - w_rows + 1) / stride;

	int sz = szO.area();

	for(size_t i = 0; i < A1.size(); ++i)
		A1[i].resize(A0.rows, sz, A0.type);

	cuda_conv2d(A0, szI, szO, stride, W, B, A1, func);

	return szO;
}

void subsample(const GpuMat &A0, const ct::Size &szA0, GpuMat &A1, GpuMat &Mask, ct::Size &szA1)
{
	if(A0.empty())
		throw new std::invalid_argument("gpumat::subsample: invalid parameters");

	int rows = A0.rows;
	int cols = A0.cols;

	if(!rows || !cols)
		throw new std::invalid_argument("gpumat::subsample: invalid parameters");

	szA1 = ct::Size(szA0.width/2, szA0.height/2);

	A1.resize(rows, szA1.area(), A0.type);
	Mask.resize(rows, szA0.area(), A0.type);

	cuda_subsample(A0, szA0, szA1, A1, Mask);
}

void subsample(const std::vector<GpuMat> &A0, const ct::Size &szA0,
			   std::vector<GpuMat> &A1, std::vector<GpuMat> &Masks,
			   ct::Size &szA1)
{
	if(A0.empty())
		throw new std::invalid_argument("gpumat::subsample: invalid parameters");
	A1.resize(A0.size());
	Masks.resize(A0.size());

	for(size_t i = 0; i < A0.size(); i++){
		subsample(A0[i], szA0, A1[i], Masks[i], szA1);
	}
}

void upsample(const GpuMat &A1, const ct::Size &szA1,
			  const ct::Size &szA0, const GpuMat &Mask, GpuMat &A0)
{
	if(A1.empty() || Mask.empty())
		throw new std::invalid_argument("gpumat::upsample: invalid parameters");

	int m = A1.rows;

	A0.resize(m, szA0.area(), A1.type);

	cuda_upsample(A1, szA1, szA0, Mask, A0);
}

void upsample(const std::vector<GpuMat> &A1, ct::Size &szA1, const ct::Size &szA0, const std::vector<GpuMat> &Masks, std::vector<GpuMat> &A0)
{
	if(A1.empty() || Masks.empty())
		throw new std::invalid_argument("gpumat::upsample: invalid parameters");
	A0.resize(A1.size());

	for(size_t i = 0; i < A1.size(); i++){
		upsample(A1[i], szA1, szA0, Masks[i], A0[i]);
	}
}

void deriv_conv2D(const GpuMat &A0, const GpuMat &gradA1,
				  const ct::Size &szA0, const ct::Size &szA1,
				  const ct::Size &szW, int stride,
				  GpuMat &gradW, float &gradB)
{
	if(A0.empty() || gradA1.empty() || !stride){
		std::cout << "gpumat::deriv_conv2D wrong parameters\n";
	}

	gradW.resize(szW.height, szW.width, A0.type);
	gradB = 0;

	memset(gradW, 0);

	cuda_deriv_conv2d(A0, gradA1, szA0, szA1, stride, gradW, gradB);

	// need reduce for B
}

void deriv_prev_cnv(std::vector<GpuMat> &deriv, const std::vector<GpuMat> &W,
					const ct::Size &sL, const ct::Size &sLsub1, int stride, GpuMat &D)
{
	if(deriv.empty() || W.empty())
		std::cout << "gpumat::deriv_prev_cnv wrong parameters\n";

	D.resize(deriv[0].rows, sLsub1.area(), deriv[0].type);

	cuda_deriv_prev_conv2d(deriv, W, sL, sLsub1, stride, D);
}

}
