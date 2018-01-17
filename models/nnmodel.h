#ifndef NNMODEL_H
#define NNMODEL_H

#include <custom_types.h>
#include <random>
#include "nn.h"
#include "mlp.h"

class nnmodel
{
public:
	nnmodel();

	enum EResultModel{
		ESigmoid,
		ESquarError
	};

	void setData(const ct::Matf& X, const ct::Matf& y);
	void setAlpha(double alpha);
	double alpha() const;

	uint32_t iteration() const;

	void setBetha1(double v);
	void setBetha2(double v);

//	ct::Matf forward(const ct::Matf& X) const;

	double L2();

//	ct::Matf& w1();
//	ct::Matf& w2();
//	ct::Matf& w3();
//	ct::Matf& w4();

//	ct::Matf& b1();
//	ct::Matf& b2();
//	ct::Matf& b3();
//	ct::Matf& b4();

	ct::Matf w(int index);
	ct::Matf b(int index);
	int count() const;

//	void init_weights(int seed = 0);
	void pass();

//	void pass_batch(int batch = 100);
//	void pass_batch(const ct::Matf& X, const ct::Matf y);

	ct::Matf resultModel(ct::Matf &m);
	ct::Matf diffModel(ct::Matf& m);

	void setLayers(const std::vector< int >& layers);
	void init_model(int seed = 0);

	ct::Matf forward_model(const ct::Matf& X);
	void pass_batch_model(const ct::Matf &X, const ct::Matf y);
	void pass_batch_model(int batch = 100);

private:
	ct::Matf m_X;
	ct::Matf m_y;

	std::vector< ct::mlpf > m_mlp;
//	std::vector< ct::Matf > m_W;
//	std::vector< ct::Matf > m_b;
	std::vector< int > m_layers;

	ct::MlpAdamOptimizer<float> m_AdamOptimizer;

	EResultModel m_resultModel;

	std::mt19937 m_generator;
#ifdef _MSC_VER
#else
#endif
};

#endif // NNMODEL_H
