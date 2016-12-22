#include "nnmodel.h"

using namespace ct;

nnmodel::nnmodel()
	: m_alpha(0.01)
	, m_inputs(0)
	, m_outputs(0)
	, m_layer1(5)
	, m_layer2(4)
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
	Matd z1 = X * m_W1;
	z1.biasPlus(m_b1);
	Matd a1 = sigmoid(z1);
	Matd z2 = a1 * m_W2;
	z2.biasPlus(m_b2);
	return resultModel(z2);
}

double nnmodel::L2() const
{
	return m_L2;
}

void nnmodel::pass()
{
	if(!m_W1.total() || !m_W2.total()
			|| !m_b1.total() || !m_b2.total()
			|| !m_X.total() || !m_y.total()
			|| m_W1.rows != m_X.cols)
		return;

	/// forward
	Matd z1 = m_X * m_W1;
	z1.biasPlus(m_b1);
	Matd a1 = sigmoid(z1);
	Matd z2 = a1 * m_W2;
	z2.biasPlus(m_b2);
	Matd a2 = sigmoid(z2);
	Matd z3 = a2 * m_Wout;
	z3.biasPlus(m_bout);
	Matd a3 = (z3);

	/// backward
	Matd d = a3 - m_y;

	Matd d2 = elemwiseMult(d, d);
	m_L2 = d2.sum() / d2.total();

	Matd sz2 = elemwiseMult(a2, 1. - a2);
	Matd sz1 = elemwiseMult(a1, 1. - a1);

//	m_W1 -= (dEdW1 * m_alpha);
//	m_b1 -= (dEdb1 * m_alpha);
//	m_W2 -= (dEdW2 * m_alpha);
//	m_b2 -= (dEdb2 * m_alpha);
//	m_Wout -= (dEdWout * m_alpha);
//	m_bout -= (dEdbout * m_alpha);
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

void nnmodel::init_weights()
{
	m_W1 = Matd(m_X.cols, m_layer1);
	m_W1.randn(0., 0.1);
	m_b1 = Matd(m_layer1, 1);
	m_b1.randn(0, 0.1);

	m_W2 = Matd(m_layer1, m_layer2);
	m_W2.randn(0., 0.1);
	m_b2 = Matd(m_layer2, 1);
	m_b2.randn(0, 0.1);

	m_Wout = Matd(m_layer2, m_y.cols);
	m_Wout.randn(0., 0.1);
	m_bout = Matd(m_y.cols, 1);
	m_bout.randn(0, 0.1);
}
