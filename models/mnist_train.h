#ifndef MNIST_TRAIN_H
#define MNIST_TRAIN_H

#include <random>

#include "nn.h"
#include "custom_types.h"
#include "mnist_reader.h"

#ifdef _USE_GPU
#include "gpumat.h"
#include "helper_gpu.h"
#endif

class mnist_train
{
public:
	mnist_train();
	~mnist_train();
	/**
	 * @brief setMnist
	 * set ref to reader of mnist data
	 * @param mnist
	 */
	void setMnist(mnist_reader* mnist);
	/**
	 * @brief forward
	 * @param X
	 * @return
	 */
	ct::Matf forward(const ct::Matf& X) const;
	/**
	 * @brief forward
	 * @param index
	 * @param count
	 * @param use_gpu
	 * @return
	 */
	ct::Matf forward(int index, int count, bool use_gpu = false);
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
	void getEstimateTest(int batch, double &l2, double &accuracy, bool use_gpu = false);
	/**
	 * @brief init
	 * @param seed
	 */
	void init(int seed);
	/**
	 * @brief pass_batch
	 * @param batch
	 */
	void pass_batch(int batch);
	/**
	 * @brief load
	 * load weigths and biases
	 * @param fn
	 */
	void load(const QString& fn);
	/**
	 * @brief save
	 * load weigths and biases
	 * @param fn
	 */
	void save(const QString& fn);
	/**
	 * @brief pass_batch_autoencoder
	 * @param batch
	 */
	void pass_batch_autoencoder(int batch, bool use_gpu = false);
	/**
	 * @brief copyWbMat2GpuMat
	 */
	void copyWbMat2GpuMat();
	/**
	 * @brief copyWbGpuMat2Mat
	 */
	void copyWbGpuMat2Mat();
	/**
	 * @brief init_weights
	 */
	void init_weights(int seed = 1);

#ifdef _USE_GPU
	/**
	 * @brief forward_gpu
	 * @param X
	 * @return
	 */
	ct::Matf forward_gpu(const ct::Matf& X);
	/**
	 * @brief forward_test_gpu
	 * @param index
	 * @param count
	 * @return
	 */
	ct::Matf forward_test_gpu(int index, int count);
	/**
	 * @brief init_gpu
	 * @param seed
	 */
	void init_gpu(int seed);
	/**
	 * @brief pass_batch_gpu
	 * @param batch
	 */
	void pass_batch_gpu(int batch);
	/**
	 * @brief pass_batch_gpu
	 * @param X
	 * @param y
	 */
	void pass_batch_gpu(const gpumat::GpuMat& X, const gpumat::GpuMat& y);
#endif

private:
	std::vector< int > m_layers;
	std::vector< ct::Matf > m_W;
	std::vector< ct::Matf > m_b;
	mnist_reader* m_mnist;
	float m_lambda;
	uint m_iteration;

	ct::Matf m_X;
	ct::Matf m_y;

	nn::AdamOptimizer<float> m_AdamOptimizer;

	std::vector< nn::SimpleAutoencoder<float> > enc;

	std::mt19937 m_generator;

	void pass_batch(const ct::Matf& X, const ct::Matf& y);

	void getX(ct::Matf& X, int batch);

#ifdef _USE_GPU
	int m_dropout_count;
	gpumat::GpuMat m_gX;
	gpumat::GpuMat m_gy;
	gpumat::GpuMat partZ;
	gpumat::GpuMat g_d;
	gpumat::GpuMat g_di, g_sz, g_tmp;
	std::vector< gpumat::GpuMat > m_gW;
	std::vector< gpumat::GpuMat > m_Dropout;
	std::vector< gpumat::GpuMat > m_DropoutT;
	std::vector< gpumat::GpuMat > m_gb;
	std::vector< gpumat::GpuMat > g_z, g_a;
	std::vector< gpumat::GpuMat > g_dW, g_dB;

	std::vector< gpumat::SimpleAutoencoder > enc_gpu;

	gpumat::AdamOptimizer m_gpu_adam;
#endif
};

#endif // MNIST_TRAIN_H
