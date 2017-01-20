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

	void setMnist(mnist_reader* mnist);

	ct::Matf forward(const ct::Matf& X) const;

	ct::Matf forward(int index, int count, bool use_gpu = false);
	ct::Matf forward_test(int index, int count, bool use_gpu = false);

	void setAlpha(double alpha);

	void setLayers(const std::vector<int>& layers);

	uint iteration() const;

	double L2(int batch = 1000);
	double L2test(int batch = 1000);
	double cross_entropy(int batch = 1000);

	void getEstimate(int batch, double &l2, double &accuracy, bool use_gpu = false);
	void getEstimateTest(int batch, double &l2, double &accuracy, bool use_gpu = false);

	void init(int seed);
	void pass_batch(int batch);

	void load(const QString& fn);
	void save(const QString& fn);

	void pass_batch_autoencoder(int batch);

#ifdef _USE_GPU
	ct::Matf forward_gpu(const ct::Matf& X);
	ct::Matf forward_test_gpu(int index, int count);
	void init_gpu(int seed);
	void pass_batch_gpu(int batch);
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

	gpumat::AdamOptimizer m_gpu_adam;
#endif
};

#endif // MNIST_TRAIN_H
