#ifndef HELPER_GPU_H
#define HELPER_GPU_H

#include "custom_types.h"
#include "gpumat.h"

namespace gpumat{

/**
 * @brief convert_to_gpu
 * @param mat
 * @param gmat
 */
void convert_to_gpu(const ct::Matf& mat, gpumat::GpuMat& gmat);
/**
 * @brief convert_to_gpu
 * @param mat
 * @param gmat
 */
void convert_to_gpu(const ct::Matd& mat, gpumat::GpuMat& gmat);
/**
 * @brief convert_to_mat
 * @param gmat
 * @param mat
 */
void convert_to_mat(const gpumat::GpuMat& gmat, ct::Matf& mat);
/**
 * @brief convert_to_mat
 * @param gmat
 * @param mat
 */
void convert_to_mat(const gpumat::GpuMat& gmat, ct::Matd& mat);

/////////////////////////////////////////

class AdamOptimizer{
public:
	AdamOptimizer();

	double alpha()const;

	void setAlpha(double v);

	double betha1() const;

	void setBetha1(double v);

	double betha2() const;

	void setBetha2(double v);

	uint32_t iteration() const;

	bool empty() const;

	bool init(const std::vector< int >& layers, int samples, int type);

	bool pass(const std::vector< gpumat::GpuMat >& gradW, const std::vector< gpumat::GpuMat >& gradB,
			  std::vector< gpumat::GpuMat >& W, std::vector< gpumat::GpuMat >& b);
private:
	uint32_t m_iteration;
	double m_betha1;
	double m_betha2;
	double m_alpha;

	gpumat::GpuMat sB, sW;

	std::vector< gpumat::GpuMat > m_mW;
	std::vector< gpumat::GpuMat > m_mb;
	std::vector< gpumat::GpuMat > m_vW;
	std::vector< gpumat::GpuMat > m_vb;
};

}

#endif // HELPER_GPU_H
