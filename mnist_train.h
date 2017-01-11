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

	ct::Matd forward(const ct::Matd& X) const;

	ct::Matd forward(int index, int count) const;
	ct::Matd forward_test(int index, int count) const;

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
private:
	std::vector< int > m_layers;
	std::vector< ct::Matd > m_W;
	std::vector< ct::Matd > m_b;
	mnist_reader* m_mnist;
	double m_lambda;

	ct::Matd m_X;
	ct::Matd m_y;

	nn::AdamOptimizer<double> m_AdamOptimizer;

	std::mt19937 m_generator;

	void pass_batch(const ct::Matd& X, const ct::Matd& y);
};

#endif // MNIST_TRAIN_H
