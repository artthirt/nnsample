#ifndef MIST_CONV_H
#define MIST_CONV_H

#include "custom_types.h"
#include "mnist_reader.h"
#include "nn.h"

#include "convnn2.h"
#include "mlp.h"

#include <random>

#ifdef _USE_GPU

#include "gpu_model.h"
#include "gpu_mlp.h"

#endif

class mnist_conv
{
public:
	mnist_conv();
	/**
	 * @brief forward
	 * @param X
	 * @return
	 */
	ct::Matf forward(const std::vector<ct::Matf> &X,
					 bool use_dropout = false,
					 bool use_gpu = false, bool train = true);
	/**
	 * @brief forward
	 * @param index
	 * @param count
	 * @param use_gpu
	 * @return
	 */
	ct::Matf forward(int index, int count, bool use_gpu = false, bool train = true);
	/**
	 * @brief forward_test
	 * @param index
	 * @param count
	 * @param use_gpu
	 * @return
	 */
	ct::Matf forward_test(int index, int count, bool use_gpu = false);
	/**
	 * @brief setAlpha
	 * @param alpha
	 */
	void setAlpha(double alpha);
	/**
	 * @brief setLayers
	 * @param layers
	 */
	void setLayers(const std::vector<int>& layers);
	/**
	 * @brief iteration
	 * @return
	 */
	uint iteration() const;
	/**
	 * @brief getEstimate
	 * @param batch
	 * @param l2
	 * @param accuracy
	 * @param use_gpu
	 */
	void getEstimate(int batch, double &l2, double &accuracy, bool use_gpu = false);
	/**
	 * @brief getEstimateTest
	 * @param batch
	 * @param l2
	 * @param accuracy
	 * @param use_gpu
	 */
	void getEstimateTest(double &l2, double &accuracy, bool use_gpu = false);
	/**
	 * @brief init
	 * @param seed
	 */
	void init(int seed);
	/**
	 * @brief pass_batch
	 * @param batch
	 */
	void pass_batch(int batch, bool use_gpu = false);
	/**
	 * @brief setMnist
	 * @param mnist
	 */
	void setMnist(mnist_reader* mnist);
	void setConvLength();

	std::vector< conv2::convnn<float> > &cnv();

	std::vector<ct::Matf> cnvW();

	void save_model(bool gpu = false);
	void load_model(bool gpu = false);

private:
	std::vector< int > m_layers;
	std::vector< conv2::convnn<float> > m_cnv;
	std::vector< ct::mlp<float> > m_mlp;
	mnist_reader *m_mnist;
	std::mt19937 m_generator;
	uint m_iteration;
	bool m_use_gpu;
	int m_seed;

	///*********
	ct::Matf m_d;
	ct::Matf m_Xout;
	std::vector< ct::Matf > m_ds;
	///*********

#ifdef _USE_GPU
	gpu_model m_gpu_model;
	std::vector< gpumat::GpuMat > gX;
	gpumat::GpuMat gY;
#endif

	ct::Size m_cnv_out_size;
	size_t m_cnv_out_len;

	void setDropout(size_t count, float prob);
	void clearDropout();

	ct::MlpOptimMoment<float> m_optim;
	conv2::CnvMomentumOptimizer<float> m_cnv_optim;

	void pass_batch(const std::vector<ct::Matf> &X, const ct::Matf& y);

	void getX(ct::Matf& X, int batch);
	void getXyTest(std::vector<ct::Matf> &X, ct::Matf &yp, int batch, bool use_rand = true, int beg = -1);
	void getXy(std::vector<ct::Matf> &X, ct::Matf& y, int batch);
	void randX(ct::Matf& X);
	void getBatchIds(std::vector< int >& indexes, int batch = -1);

	void conv(const std::vector<ct::Matf> &X, ct::Matf &X_out);
};

#endif // MIST_CONV_H
