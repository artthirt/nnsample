#ifndef GPU_MODEL_H
#define GPU_MODEL_H

#include "custom_types.h"
#include "gpumat.h"
#include "convnn2_gpu.h"
#include "gpu_mlp.h"

class gpu_model
{
public:
	gpu_model();

	bool isInit() const;
	/**
	 * @brief setConvLength
	 * @param count_cnvW
	 * @param weight_sizes
	 */
	void setConvLength();
	/**
	 * @brief forward_gpu
	 * @param X
	 * @return
	 */
	ct::Matf forward_gpu(const std::vector<gpumat::GpuMat> &X, bool use_dropout = false, bool converToMatf = true);
	/**
	 * @brief init_gpu
	 * @param seed
	 */
	void init_gpu(const std::vector<int> &layers);
	/**
	 * @brief pass_batch_gpu
	 * @param X
	 * @param y
	 */
	void pass_batch_gpu(const std::vector<gpumat::GpuMat> &X, const gpumat::GpuMat& y);

	uint32_t iteration() const;
	void setAlpha(double val);

	void setLayers(const std::vector<int> &layers);

	std::vector< gpumat::convnn_gpu > &cnv();

private:
	std::vector< int > m_layers;
	std::vector< gpumat::convnn_gpu > m_cnv;
	std::vector< gpumat::mlp > m_gpu_mlp;
	gpumat::convnn_gpu m_adds;
	bool m_init;

	uint32_t m_iteration;

	ct::Size m_cnv_out_size;
	int m_cnv_out_len;

	int m_dropout_count;
	gpumat::GpuMat m_gX;
	gpumat::GpuMat m_gy;
//	gpumat::GpuMat partZ;
	gpumat::GpuMat g_d;
	gpumat::GpuMat g_Xout;
	std::vector< gpumat::GpuMat > ds;

	gpumat::MlpOptim m_gpu_adam;

	void conv(const std::vector<gpumat::GpuMat> &X, gpumat::GpuMat &X_out);

	void setGpuDropout(size_t count, float prob);
	void clearGpuDropout();
};

#endif // GPU_MODEL_H
