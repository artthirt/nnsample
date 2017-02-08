#ifndef GPU_MODEL_H
#define GPU_MODEL_H

#include "custom_types.h"
#include "gpumat.h"
#include "convnn_gpu.h"

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
	void setConvLength(const std::vector< int > &count_cnvW, std::vector< int >* weight_sizes = 0);
	/**
	 * @brief forward_gpu
	 * @param X
	 * @return
	 */
	ct::Matf forward_gpu(const ct::Matf& X);
	/**
	 * @brief init_gpu
	 * @param seed
	 */
	void init_gpu(const std::vector<int> &layers, int seed);
	/**
	 * @brief pass_batch_gpu
	 * @param X
	 * @param y
	 */
	void pass_batch_gpu(const gpumat::GpuMat& X, const gpumat::GpuMat& y);

	uint iteration() const;
	void setAlpha(double val);

	void setLayers(const std::vector<int> &layers);

private:
	std::vector< int > m_layers;
	std::vector< std::vector< gpumat::convnn > > m_cnv;
	gpumat::convnn m_adds;
	std::vector< int > m_count_cnvW;
	bool m_init;

	int m_conv_length;
	uint m_iteration;

	ct::Size m_cnv_out_size;
	int m_cnv_out_len;

	int m_dropout_count;
	gpumat::GpuMat m_cnvA;
	gpumat::GpuMat m_gX;
	gpumat::GpuMat m_gy;
	gpumat::GpuMat partZ;
	std::vector< gpumat::GpuMat > g_d;
	std::vector< gpumat::GpuMat > g_sz, g_tmp;
	std::vector< gpumat::GpuMat > m_gW;
	std::vector< gpumat::GpuMat > m_Dropout;
	std::vector< gpumat::GpuMat > m_DropoutT;
	std::vector< gpumat::GpuMat > m_gb;
	std::vector< gpumat::GpuMat > g_z, g_a;
	std::vector< gpumat::GpuMat > g_dW, g_dB;
	std::vector< std::vector< gpumat::GpuMat > > ds;

	std::vector< gpumat::SimpleAutoencoder > enc_gpu;

	gpumat::AdamOptimizer m_gpu_adam;

	void conv(const gpumat::GpuMat &X, gpumat::GpuMat &X_out);

	void init_arrays();
};

#endif // GPU_MODEL_H
