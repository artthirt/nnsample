#ifndef CONVNN_GPU_H
#define CONVNN_GPU_H

#include "common_types.h"
#include "cuda_common.h"
#include "gpumat.h"

namespace gpumat{

class convnn
{
public:
	convnn();
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
				  float &gradB);

}

#endif // CONVNN_GPU_H
