#ifndef MNIST_TRAIN_H
#define MNIST_TRAIN_H

#include <random>

#include "nn.h"
#include "custom_types.h"
#include "mnist_reader.h"

class mnist_train
{
public:
	mnist_train();

	void setMnist(mnist_reader* mnist);

	ct::Matf forward(const ct::Matf& X) const;

	ct::Matf forward(int index, int count) const;
	ct::Matf forward_test(int index, int count) const;

	void setAlpha(double alpha);

	void setLayers(const std::vector<int>& layers);

	uint iteration() const;

	double L2(int batch = 1000);
	double L2test(int batch = 1000);
	double cross_entropy(int batch = 1000);

	void getEstimate(int batch, double &l2, double &accuracy);
	void getEstimateTest(int batch, double &l2, double &accuracy);

	void init(int seed);
	void pass_batch(int batch);

	void load(const QString& fn);
	void save(const QString& fn);

	void pass_batch_autoencoder(int batch);
private:
	std::vector< int > m_layers;
	std::vector< ct::Matf > m_W;
	std::vector< ct::Matf > m_b;
	mnist_reader* m_mnist;
	float m_lambda;

	ct::Matf m_X;
	ct::Matf m_y;

	nn::AdamOptimizer<float> m_AdamOptimizer;

	std::vector< nn::SimpleAutoencoder<float> > enc;


	std::mt19937 m_generator;

	void pass_batch(const ct::Matf& X, const ct::Matf& y);

	void getX(ct::Matf& X, int batch);
};

#endif // MNIST_TRAIN_H
