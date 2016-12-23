#ifndef NNMODEL_H
#define NNMODEL_H

#include <custom_types.h>
#include <random>

class nnmodel
{
public:
	nnmodel();

	enum EResultModel{
		ESigmoid,
		ESquarError
	};

	void setData(const ct::Matd& X, const ct::Matd& y);
	void setNeurons(int layer, int neurons);

	void setAlpha(double alpha);
	double alpha() const;

	ct::Matd forward(const ct::Matd& X);

	double L2() const;

	ct::Matd& w1();
	ct::Matd& w2();
	ct::Matd& w3();

	ct::Matd& b1();
	ct::Matd& b2();
	ct::Matd& b3();

	void init_weights(int seed = 0);
	void pass();

	ct::Matd resultModel(ct::Matd& m);
	ct::Matd diffModel(ct::Matd& m);

private:
	ct::Matd m_X;
	ct::Matd m_y;

	ct::Matd m_W1;
	ct::Matd m_b1;

	ct::Matd m_W2;
	ct::Matd m_b2;

	ct::Matd m_W3;
	ct::Matd m_b3;

	ct::Matd m_dW1, m_dW2, m_dW3, m_dB1, m_dB2, m_dB3;

	int m_layer1;
	int m_layer2;

	int m_inputs;
	int m_outputs;
	double m_alpha;
	double m_betha;

	double m_L2;

	EResultModel m_resultModel;

	std::normal_distribution< double > m_normal;
	std::mt19937 m_generator;
#ifdef _MSC_VER
#else
#endif
};

#endif // NNMODEL_H
