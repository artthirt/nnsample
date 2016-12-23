#include "nnmodel.h"

using namespace ct;

nnmodel::nnmodel()
	: m_alpha(0.1)
	, m_betha(0.9)
	, m_inputs(0)
	, m_outputs(0)
	, m_layer1(50)
	, m_layer2(30)
	, m_L2(99999999)
	, m_resultModel(ESquarError)
{
	m_normal = std::normal_distribution< double >(0, 0.1);
}

void nnmodel::setData(const ct::Matd &X, const ct::Matd &y)
{
	if(X.rows != y.rows)
		return;
	m_X = X;
	m_y = y;
}

void nnmodel::setNeurons(int layer, int neurons)
{
	switch (layer) {
		case 0:
			m_layer1 = layer;
			break;
		default:
			m_layer2 = layer;
			break;
	}
}

void nnmodel::setAlpha(double alpha)
{
	m_alpha = alpha;
}

double nnmodel::alpha() const
{
	return m_alpha;
}

ct::Matd nnmodel::forward(const ct::Matd &X)
{
	Matd z2 = X * m_W1;			/// [1000 2] * [2 5] = [1000 5]
	z2.biasPlus(m_b1);				///
	Matd a2 = tanh(z2);			/// [1000 5]
	Matd z3 = a2 * m_W2;			/// [1000 5] * [5 4] = [1000 4]
	z3.biasPlus(m_b2);				///
	Matd a3 = tanh(z3);			/// [1000 4]
	Matd z4 = a3 * m_W3;			/// [1000 4] * [4 1] = [1000 1]
	z4.biasPlus(m_b3);			///
//	Matd a4 = (z4);					/// [1000 1]
	return z4;
}

double nnmodel::L2() const
{
	return m_L2;
}

Matd &nnmodel::w1()
{
	return m_W1;
}

Matd &nnmodel::w2()
{
	return m_W2;
}

Matd &nnmodel::w3()
{
	return m_W3;
}

Matd &nnmodel::b1()
{
	return m_b1;
}

Matd &nnmodel::b2()
{
	return m_b2;
}

Matd &nnmodel::b3()
{
	return m_b3;
}

void nnmodel::pass()
{
	if(!m_W1.total() || !m_W2.total()
			|| !m_b1.total() || !m_b2.total()
			|| !m_X.total() || !m_y.total()
			|| m_W1.rows != m_X.cols)
		return;

	/// forward
	Matd z2 = m_X * m_W1;			/// [1000 2] * [2 5] = [1000 5]
	z2.biasPlus(m_b1);				///
	Matd a2 = tanh(z2);				/// [1000 5]
	Matd z3 = a2 * m_W2;			/// [1000 5] * [5 4] = [1000 4]
	z3.biasPlus(m_b2);				///
	Matd a3 = tanh(z3);				/// [1000 4]
	Matd z4 = a3 * m_W3;			/// [1000 4] * [4 1] = [1000 1]
	z4.biasPlus(m_b3);
	Matd a4 = (z4);					/// [1000 1]

	/// backward
	Matd d4 = a4 - m_y;

	Matd dl2 = elemwiseMult(d4, d4);
	m_L2 = dl2.sum() / dl2.total();

	Matd sz3 = elemwiseMult(a3, a3);
	sz3 = 1. - sz3;
	Matd sz2 = elemwiseMult(a2, a2);
	sz2 = 1. - sz2;

	int m = m_X.rows;

	Matd dW3;
	Matd dW2;
	Matd dW1;
	Matd dB3, dB2, dB1;
//	for(int i = 0; i < m; i++){

//	}

	Matd d3 = d4 * m_W3.t();		/// [1000 1] * [1 4] = [1000 4]
	d3 = elemwiseMult(d3, sz3);		/// [1000 4]
	dW3 = a3.t() * d4;				/// [4 1000] * [1000 1] = [4 1]
	dW3 *= 1./m;
	dB3 = (sumRows(d4) * (1./m)).t();

	Matd d2 = d3 * m_W2.t();		///
	d2 = elemwiseMult(d2, sz2);		///
	dW2 = a2.t() * d3;				///
	dW2 *= 1./m;
	dB2 = (sumRows(d3) * (1./m)).t();

	dW1 = m_X.t() * d2;				///
	dW1 *= 1./m;
	dB1 = (sumRows(d2) * (1./m)).t();

	m_dB1 = (m_betha * m_dB1) + (1 - m_betha) * (dB1);
	m_dB2 = (m_betha * m_dB2) + (1 - m_betha) * (dB2);
	m_dB3 = (m_betha * m_dB3) + (1 - m_betha) * (dB3);
	m_dW1 = (m_betha * m_dW1) + (1 - m_betha) * (dW1);
	m_dW2 = (m_betha * m_dW2) + (1 - m_betha) * (dW2);
	m_dW3 = (m_betha * m_dW3) + (1 - m_betha) * (dW3);

	m_W1 -= m_alpha * (m_dW1);
	m_b1 -= m_alpha * (m_dB1);
	m_W2 -= m_alpha * (m_dW2);
	m_b2 -= m_alpha * (m_dB2);
	m_W3 -= m_alpha * (m_dW3);
	m_b3 -= m_alpha * (m_dB3);

}

Matd nnmodel::resultModel(Matd &m)
{
	switch (m_resultModel) {
		case ESigmoid:
			return sigmoid(m);
		default:
			return m;
	}
}

Matd nnmodel::diffModel(Matd &m)
{
	switch (m_resultModel) {
		case ESigmoid:
		{
			Matd s = m * (1. - m);
			return s;

		}
		default:
			return m;
	}
}

void nnmodel::init_weights(int seed)
{
	m_W1 = Matd(m_X.cols, m_layer1);
	m_W1.randn(0., 0.1, seed);
	m_b1 = Matd::ones(m_layer1, 1);
	m_b1.randn(0, 0.1, seed);

	m_W2 = Matd(m_layer1, m_layer2);
	m_W2.randn(0., 0.1, seed);
	m_b2 = Matd::ones(m_layer2, 1);
	m_b2.randn(0, 0.1, seed);

	m_W3 = Matd(m_layer2, m_y.cols);
	m_W3.randn(0., 0.1, seed);
	m_b3 = Matd::ones(m_y.cols, 1);
	m_b3.randn(0, 0.1, seed);

	m_dB1 = Matd::zeros(m_b1.size());
	m_dB2 = Matd::zeros(m_b2.size());
	m_dB3 = Matd::zeros(m_b3.size());
	m_dW1 = Matd::zeros(m_W1.size());
	m_dW2 = Matd::zeros(m_W2.size());
	m_dW3 = Matd::zeros(m_W3.size());
}
