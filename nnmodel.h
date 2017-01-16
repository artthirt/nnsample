#ifndef NNMODEL_H
#define NNMODEL_H

#include <custom_types.h>
#include <random>
#include "nn.h"

class nnmodel
{
public:
	nnmodel();

	enum EResultModel{
		ESigmoid,
		ESquarError
	};

	void setData(const ct::Matd& X, const ct::Matd& y);
	void setAlpha(double alpha);
	double alpha() const;

	uint32_t iteration() const;

	void setBetha1(double v);
	void setBetha2(double v);

//	ct::Matd forward(const ct::Matd& X) const;

	double L2() const;

//	ct::Matd& w1();
//	ct::Matd& w2();
//	ct::Matd& w3();
//	ct::Matd& w4();

//	ct::Matd& b1();
//	ct::Matd& b2();
//	ct::Matd& b3();
//	ct::Matd& b4();

	ct::Matd w(int index);
	ct::Matd b(int index);
	int count() const;

//	void init_weights(int seed = 0);
	void pass();

//	void pass_batch(int batch = 100);
//	void pass_batch(const ct::Matd& X, const ct::Matd y);

	ct::Matd resultModel(ct::Matd& m);
	ct::Matd diffModel(ct::Matd& m);

	void setLayers(const std::vector< int >& layers);
	void init_model(int seed = 0);

	ct::Matd forward_model(const ct::Matd& X) const;
	void pass_batch_model(const ct::Matd &X, const ct::Matd y);
	void pass_batch_model(int batch = 100);

private:
	ct::Matd m_X;
	ct::Matd m_y;

	std::vector< ct::Matd > m_W;
	std::vector< ct::Matd > m_b;
	std::vector< int > m_layers;

	nn::AdamOptimizer<double> m_AdamOptimizer;

	EResultModel m_resultModel;

	std::mt19937 m_generator;
#ifdef _MSC_VER
#else
#endif
};

#endif // NNMODEL_H
