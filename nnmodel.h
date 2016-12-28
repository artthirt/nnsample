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

	void setBetha1(double v);
	void setBetha2(double v);

	ct::Matd forward(const ct::Matd& X) const;

	double L2() const;

	ct::Matd& w1();
	ct::Matd& w2();
	ct::Matd& w3();
	ct::Matd& w4();

	ct::Matd& b1();
	ct::Matd& b2();
	ct::Matd& b3();
	ct::Matd& b4();

	ct::Matd w(int index);
	ct::Matd b(int index);
	int count() const;

	void init_weights(int seed = 0);
	void pass();

	void pass_batch(int batch = 100);
	void pass_batch(const ct::Matd& X, const ct::Matd y);

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
	double m_iteration;

	std::vector< ct::Matd > m_W;
	std::vector< ct::Matd > m_b;
	std::vector< int > m_layers;

	std::vector< ct::Matd > m_mW;
	std::vector< ct::Matd > m_mb;
	std::vector< ct::Matd > m_vW;
	std::vector< ct::Matd > m_vb;

	ct::Matd m_W1;
	ct::Matd m_b1;

	ct::Matd m_W2;
	ct::Matd m_b2;

	ct::Matd m_W3;
	ct::Matd m_b3;

	ct::Matd m_W4;
	ct::Matd m_b4;

	ct::Matd m_dW1, m_dW2, m_dW3, m_dW4, m_dB1, m_dB2, m_dB3, m_dB4;

	int m_layer1;
	int m_layer2;
	int m_layer3;

	int m_inputs;
	int m_outputs;
	double m_alpha;
	double m_betha1;
	double m_betha2;

	double m_L2;

	EResultModel m_resultModel;

	std::normal_distribution< double > m_normal;
	std::mt19937 m_generator;
#ifdef _MSC_VER
#else
#endif
};

#endif // NNMODEL_H
