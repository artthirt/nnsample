#ifndef CONVNN_GPU_H
#define CONVNN_GPU_H

#include "common_types.h"
#include "cuda_common.h"
#include "gpumat.h"
#include "helper_gpu.h"
#include "nn.h"

namespace gpumat{

class convnn
{
public:
	typedef std::vector< GpuMat > tvmat;

	convnn();

	gpumat::GpuMat A0;
	gpumat::GpuMat DltA0;
	tvmat A1;
	tvmat A2;
	tvmat W;
	tvmat prevW;
	std::vector< float > B;
	tvmat Masks;
	ct::Size szA0;
	ct::Size szA1;
	ct::Size szA2;
	int stride;
	int weight_size;
	gpumat::AdamOptimizer m_optim;

	void setWeightSize(int ws);

	void init(int count_weight, const ct::Size& _szA0);

	void update_random();

	void setAlpha(double alpha);

	void clear();

	void forward(const gpumat::GpuMat & mat, gpumat::etypefunction func);

	void apply_func(const GpuMat& A, GpuMat& B, etypefunction func);

	void back2conv(const tvmat& A1, const tvmat& dA2, tvmat& dA1, etypefunction func);

	void backward(const std::vector< gpumat::GpuMat >& Delta, gpumat::etypefunction func);

	void hconcat(const std::vector< convnn > &cnv, gpumat::GpuMat & _out);


private:
	bool m_init;

	std::vector< gpumat::GpuMat  > dA2, dA1;
	std::vector< gpumat::GpuMat  > slice;
};

ct::Size conv2D(const GpuMat& A0,
			const ct::Size& szI,
			int stride,
			const std::vector< GpuMat >& W,
			const std::vector<float> B,
			std::vector< GpuMat > &A1,
			etypefunction func = RELU);

/**
 * @brief subsample
 * @param A0
 * @param szA0
 * @param A1
 * @param Mask
 * @param szA1
 * @return
 */
void subsample(const GpuMat &A0,
			   const ct::Size& szA0,
			   GpuMat& A1,
			   GpuMat& Mask,
			   ct::Size& szA1);

/**
 * @brief subsample
 * @param A0
 * @param szA0
 * @param A1
 * @param Masks
 * @param szA1
 * @return
 */
void subsample(const std::vector< GpuMat > &A0,
			   const ct::Size& szA0, std::vector< GpuMat > &A1,
			   std::vector< GpuMat > &Masks,
			   ct::Size& szA1);

/**
 * @brief upsample
 * @param A1
 * @param szA1
 * @param szA0
 * @param Mask
 * @param A0
 * @return
 */
void upsample(const GpuMat &A1,
			  const ct::Size& szA1,
			  const ct::Size& szA0,
			  const GpuMat &Mask,
			  GpuMat& A0);

/**
 * @brief upsample
 * @param A1
 * @param szA1
 * @param szA0
 * @param Masks
 * @param A0
 * @return
 */
void upsample(const std::vector< GpuMat > &A1,
			  ct::Size& szA1,
			  const ct::Size& szA0,
			  const std::vector< GpuMat > &Masks,
			  std::vector< GpuMat >& A0);

/**
 * @brief deriv_conv2D
 * @param A0
 * @param gradA1
 * @param szA0
 * @param szA1
 * @param szW
 * @param stride
 * @param gradW
 * @param gradB
 */
void deriv_conv2D(const GpuMat& A0,
				  const GpuMat& gradA1,
				  const ct::Size& szA0,
				  const ct::Size& szA1,
				  const ct::Size &szW,
				  int stride,
				  GpuMat &gradW,
				  float &gradB,
				  GpuMat *pblock = nullptr);

/**
 * @brief deriv_conv2D
 * @param A0
 * @param gradA1
 * @param szA0
 * @param szA1
 * @param szW
 * @param stride
 * @param gradW
 * @param gradB
 */
void deriv_conv2D(const GpuMat & A0,
				  const std::vector< GpuMat>& gradA1,
				  const ct::Size& szA0,
				  const ct::Size& szA1,
				  const ct::Size &szW,
				  int stride,
				  std::vector< GpuMat > &gradW,
				  std::vector< float > &gradB, std::vector<GpuMat> *pblocks = nullptr);

/**
 * @brief deriv_prev_cnv
 * @param deriv
 * @param W
 * @param sL
 * @param sLsub1
 * @param D
 */
void deriv_prev_cnv(const std::vector<GpuMat> &deriv,
					const std::vector< GpuMat >& W,
					const ct::Size& sL, const ct::Size& sLsub1, int stride,
					GpuMat& D);


/**
 * @brief hsplit
 * @param res
 * @param cols
 * @param list
 */
void hsplit(const GpuMat& res, int cols, std::vector< GpuMat >& list);

/**
 * @brief hconcat
 * @param list
 * @param res
 */
void hconcat(const std::vector< GpuMat >& list, GpuMat& res);

}/* @end gpumat */

#endif // CONVNN_GPU_H
