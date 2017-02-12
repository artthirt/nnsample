#ifndef GPU_MODEL_H
#define GPU_MODEL_H

#include "custom_types.h"
#include "gpumat.h"
#include "convnn_gpu.h"
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
	void setConvLength(const std::vector< int > &count_cnvW, std::vector< int >* weight_sizes = 0);
	/**
	 * @brief forward_gpu
	 * @param X
	 * @return
	 */
	ct::Matf forward_gpu(const gpumat::GpuMat &X, bool use_dropout = false, bool converToMatf = true);
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
	void pass_batch_gpu(const gpumat::GpuMat& X, const gpumat::GpuMat& y);

	uint iteration() const;
	void setAlpha(double val);

	void setLayers(const std::vector<int> &layers);

	std::vector<std::vector<gpumat::convnn> > &cnv();

private:
	std::vector< int > m_layers;
	std::vector< std::vector< gpumat::convnn > > m_cnv;
	std::vector< gpumat::mlp > m_gpu_mlp;
	gpumat::convnn m_adds;
	std::vector< int > m_count_cnvW;
	bool m_init;

	int m_conv_length;
	uint m_iteration;

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

	void conv(const gpumat::GpuMat &X, gpumat::GpuMat &X_out);

	void setGpuDropout(size_t count, float prob);
	void clearGpuDropout();
};

#endif // GPU_MODEL_H
