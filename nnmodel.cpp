#include "nnmodel.h"
#include  <map>

using namespace ct;

nnmodel::nnmodel()
	: m_alpha(0.1)
	, m_betha(0.9)
	, m_inputs(0)
	, m_outputs(0)
	, m_layer1(8)
	, m_layer2(40)
	, m_layer3(20)
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

ct::Matd nnmodel::forward(const ct::Matd &X) const
{
	Matd z2 = X * m_W1;			/// [1000 2] * [2 5] = [1000 5]
	z2.biasPlus(m_b1);				///
	Matd a2 = tanh(z2);			/// [1000 5]
	Matd z3 = a2 * m_W2;			/// [1000 5] * [5 4] = [1000 4]
	z3.biasPlus(m_b2);				///
	Matd a3 = tanh(z3);			/// [1000 4]
	Matd z4 = a3 * m_W3;			/// [1000 4] * [4 1] = [1000 1]
	z4.biasPlus(m_b3);			///
	Matd a4 = tanh(z4);					/// [1000 1]
	Matd z5 = a4 * m_W4;
	z5.biasPlus(m_b4);
	return z5;
}

double nnmodel::L2() const
{
	Matd y = forward(m_X);
	Matd d = m_y - y;
	d = elemwiseMult(d, d);
	double L2 = d.sum() * 1./d.rows;

	return L2;
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

Matd &nnmodel::w4()
{
	return m_W4;
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

Matd &nnmodel::b4()
{
	return m_b4;
}

void nnmodel::pass()
{
	pass_batch(m_X, m_y);
}

void nnmodel::pass_batch(int batch)
{
	if(!m_X.total() || !m_y.total() || m_X.rows != m_y.rows || !batch || batch > m_X.rows)
		return;

	Matd X(batch, m_X.cols), y(batch, m_y.cols);

	std::vector<int> indexes;
	indexes.resize(batch);
	std::uniform_int_distribution<int> ud(0, m_X.rows - 1);
	std::map<int, bool> set;
	for(int i = 0; i < batch; i++){
		int v = ud(m_generator);
		while(set.find(v) != set.end()){
			v = ud(m_generator);
		}
		set[v] = true;
		indexes[i] = v;
	}
	X = m_X.getRows(indexes);
	y = m_y.getRows(indexes);

	pass_batch(X, y);

//	y = forward(m_X);
//	Matd d = m_y - y;
//	d = elemwiseMult(d, d);
//	m_L2 = d.sum() * 1./d.rows;
}

void nnmodel::pass_batch(const Matd &X, const Matd y)
{
	if(!m_W1.total() || !m_W2.total()
			|| !m_b1.total() || !m_b2.total()
			|| !X.total() || !y.total()
			|| m_W1.rows != X.cols)
		return;

	/// forward
	Matd z2 = X * m_W1;				/// [1000 2] * [2 5] = [1000 5]
	z2.biasPlus(m_b1);				///
	Matd a2 = tanh(z2);				/// [1000 5]
	Matd z3 = a2 * m_W2;			/// [1000 5] * [5 4] = [1000 4]
	z3.biasPlus(m_b2);				///
	Matd a3 = tanh(z3);				/// [1000 4]
	Matd z4 = a3 * m_W3;			/// [1000 4] * [4 3] = [1000 3]
	z4.biasPlus(m_b3);
	Matd a4 = tanh(z4);				/// [1000 3]
	Matd z5 = a4 * m_W4;			/// [1000 3] * [3 1] = [1000 1]
	z5.biasPlus(m_b4);
	Matd a5 = z5;

	/// backward
	Matd d5 = a5 - y;

	Matd sz4 = elemwiseMult(a4, a4);
	sz4 = 1. - sz4;
	Matd sz3 = elemwiseMult(a3, a3);
	sz3 = 1. - sz3;
	Matd sz2 = elemwiseMult(a2, a2);
	sz2 = 1. - sz2;

	int m = X.rows;

	Matd dW4, dW3, dW2, dW1;
	Matd dB4, dB3, dB2, dB1;
//	for(int i = 0; i < m; i++){

//	}
	/// layer 5 ->4
	Matd d4 = d5 * m_W4.t();
	d4 = elemwiseMult(d4, sz4);
	dW4 = a4.t() * d5;
	dW4 *= 1./m;
	dB4 = (sumRows(d5) * (1./m)).t();

	/// layer 4 -> 3
	Matd d3 = d4 * m_W3.t();		/// [1000 1] * [1 4] = [1000 4]
	d3 = elemwiseMult(d3, sz3);		/// [1000 4]
	dW3 = a3.t() * d4;				/// [4 1000] * [1000 1] = [4 1]
	dW3 *= 1./m;
	dB3 = (sumRows(d4) * (1./m)).t();

	/// layer 3 -> 2
	Matd d2 = d3 * m_W2.t();		///
	d2 = elemwiseMult(d2, sz2);		///
	dW2 = a2.t() * d3;				///
	dW2 *= 1./m;
	dB2 = (sumRows(d3) * (1./m)).t();

	/// layer 2 -> 1
	dW1 = X.t() * d2;				///
	dW1 *= 1./m;
	dB1 = (sumRows(d2) * (1./m)).t();

	m_dB1 = (m_betha * m_dB1) + (1 - m_betha) * (dB1);
	m_dB2 = (m_betha * m_dB2) + (1 - m_betha) * (dB2);
	m_dB3 = (m_betha * m_dB3) + (1 - m_betha) * (dB3);
	m_dB4 = (m_betha * m_dB4) + (1 - m_betha) * (dB4);
	m_dW1 = (m_betha * m_dW1) + (1 - m_betha) * (dW1);
	m_dW2 = (m_betha * m_dW2) + (1 - m_betha) * (dW2);
	m_dW3 = (m_betha * m_dW3) + (1 - m_betha) * (dW3);
	m_dW4 = (m_betha * m_dW4) + (1 - m_betha) * (dW4);

	m_W1 -= m_alpha * (m_dW1);
	m_b1 -= m_alpha * (m_dB1);
	m_W2 -= m_alpha * (m_dW2);
	m_b2 -= m_alpha * (m_dB2);
	m_W3 -= m_alpha * (m_dW3);
	m_b3 -= m_alpha * (m_dB3);
	m_W4 -= m_alpha * (m_dW4);
	m_b4 -= m_alpha * (m_dB4);
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

	m_W3 = Matd(m_layer2, m_layer3);
	m_W3.randn(0., 0.1, seed);
	m_b3 = Matd::ones(m_layer3, 1);
	m_b3.randn(0, 0.1, seed);

	m_W4 = Matd(m_layer3, m_y.cols);
	m_W4.randn(0., 0.1, seed);
	m_b4 = Matd::ones(m_y.cols, 1);
	m_b4.randn(0, 0.1, seed);

	m_dB1 = Matd::zeros(m_b1.size());
	m_dB2 = Matd::zeros(m_b2.size());
	m_dB3 = Matd::zeros(m_b3.size());
	m_dB4 = Matd::zeros(m_b4.size());
	m_dW1 = Matd::zeros(m_W1.size());
	m_dW2 = Matd::zeros(m_W2.size());
	m_dW3 = Matd::zeros(m_W3.size());
	m_dW4 = Matd::zeros(m_W4.size());
}
