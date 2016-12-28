#include "nnmodel.h"
#include  <map>
#include <iostream>

using namespace ct;

nnmodel::nnmodel()
	: m_alpha(0.01)
	, m_betha1(0.9)
	, m_betha2(0.99)
	, m_iteration(0)
	, m_inputs(0)
	, m_outputs(0)
	, m_layer1(20)
	, m_layer2(60)
	, m_layer3(15)
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
			m_layer1 = neurons;
			break;
		default:
			m_layer2 = neurons;
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

void nnmodel::setBetha1(double v)
{
	m_betha1 = v;
}

void nnmodel::setBetha2(double v)
{
	m_betha2 = v;
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
	Matd y = forward_model(m_X);
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

Matd nnmodel::w(int index)
{
	return m_W[index];
}

Matd nnmodel::b(int index)
{
	return m_b[index];
}

int nnmodel::count() const
{
	return m_W.size();
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

	m_dB1 = (m_betha1 * m_dB1) + (1 - m_betha1) * (dB1);
	m_dB2 = (m_betha1 * m_dB2) + (1 - m_betha1) * (dB2);
	m_dB3 = (m_betha1 * m_dB3) + (1 - m_betha1) * (dB3);
	m_dB4 = (m_betha1 * m_dB4) + (1 - m_betha1) * (dB4);
	m_dW1 = (m_betha1 * m_dW1) + (1 - m_betha1) * (dW1);
	m_dW2 = (m_betha1 * m_dW2) + (1 - m_betha1) * (dW2);
	m_dW3 = (m_betha1 * m_dW3) + (1 - m_betha1) * (dW3);
	m_dW4 = (m_betha1 * m_dW4) + (1 - m_betha1) * (dW4);

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

void nnmodel::setLayers(const std::vector<int> &layers)
{
	m_layers = layers;
}

void nnmodel::init_model(int seed)
{
	if(m_X.empty() || m_y.empty() || m_layers.empty() || m_layers.back() != m_y.cols)
		return;

	int input = m_X.cols;
	int output = m_layers[0];

	m_W.resize(m_layers.size());
	m_b.resize(m_layers.size());

	m_mW.resize(m_layers.size());
	m_mb.resize(m_layers.size());

	m_vW.resize(m_layers.size());
	m_vb.resize(m_layers.size());

	for(size_t i = 0; i < m_layers.size(); i++){
		output = m_layers[i];

		double n = 1./sqrt(input);

		m_W[i] = Matd(input, output);
		m_W[i].randn(0., n, seed);
		m_b[i] = Matd::ones(output, 1);
		m_b[i].randn(0, n, seed);

		m_mW[i] = Matd::zeros(input, output);
		m_vW[i] = Matd::zeros(input, output);

		m_mb[i] = Matd::zeros(output, 1);
		m_vb[i] = Matd::zeros(output, 1);

		input = output;
	}
}

Matd nnmodel::forward_model(const Matd &X) const
{
	if(m_W.empty() || m_b.empty() || m_layers.empty() || m_layers.back() != m_y.cols)
		return Matd(0, 0);
	Matd x = X, z, a;

	for(int i = 0; i < m_layers.size(); i++){
		z = x * m_W[i];
		z.biasPlus(m_b[i]);
		if(i < m_layers.size() - 1){
			a = tanh(z);
			x = a;
		}else
			a = z;
	}
	return a;
}

void nnmodel::pass_batch_model(const Matd &X, const Matd y)
{
	if(m_W.empty() || m_b.empty() || m_mW.empty() || m_layers.empty() ||
			m_mb.empty() || m_vW.empty() || m_vb.empty() || m_layers.back() != m_y.cols){
		std::cout << "wrong parameters of model\n";
		return;
	}

	/// forward

	std::vector< Matd > z, a;
	z.resize(m_layers.size());
	a.resize(m_layers.size() + 1);

	a[0] = X;
	for(int i = 0; i < m_layers.size(); i++){
		z[i] = a[i] * m_W[i];
		z[i].biasPlus(m_b[i]);
		if(i < m_layers.size() - 1){
			a[i + 1] = tanh(z[i]);
		}else
			a[i + 1] = z[i];
	}

	std::vector< Matd > dW, dB;
	dW.resize(m_layers.size());
	dB.resize(m_layers.size());

	double m = X.rows;
	Matd d = a.back() - y;

	/// backward

	for(int i = m_layers.size() - 1; i > -1; --i){
		Matd sz = elemwiseMult(a[i], a[i]);
		sz = 1. - sz;

		Matd di = d * m_W[i].t();
		di = elemwiseMult(di, sz);
		dW[i] = a[i].t() * d;
		dW[i] *= 1./m;
		dB[i] = (sumRows(d) * (1./m)).t();
		d = di;
	}

	m_iteration++;
	double sb1 = 1. / (1. - pow(m_betha1, m_iteration));
	double sb2 = 1. / (1. - pow(m_betha2, m_iteration));
	double eps = 10e-8;

	for(size_t i = 0; i < m_layers.size(); ++i){
		m_mW[i] = m_betha1 * m_mW[i] + (1. - m_betha1) * dW[i];
		m_mb[i] = m_betha1 * m_mb[i] + (1. - m_betha1) * dB[i];

		m_vW[i] = m_betha2 * m_vW[i] + (1. - m_betha2) * elemwiseSqr(dW[i]);
		m_vb[i] = m_betha2 * m_vb[i] + (1. - m_betha2) * elemwiseSqr(dB[i]);

		Matd mWs = m_mW[i] * sb1;
		Matd mBs = m_mb[i] * sb1;
		Matd vWs = m_vW[i] * sb2;
		Matd vBs = m_vb[i] * sb2;

		vWs.sqrt(); vBs.sqrt();
		vWs += eps; vBs += eps;
		mWs = elemwiseDiv(mWs, vWs);
		mBs = elemwiseDiv(mBs, vBs);

		m_W[i] -= m_alpha * mWs;
		m_b[i] -= m_alpha * mBs;
	}
}

void nnmodel::pass_batch_model(int batch)
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

	pass_batch_model(X, y);
}

void nnmodel::init_weights(int seed)
{
	double m = m_X.cols;

	m_W1 = Matd(m_X.cols, m_layer1);
	m_W1.randn(0., 1./sqrt(m), seed);
	m_b1 = Matd::ones(m_layer1, 1);
	m_b1.randn(0, 1./sqrt(m), seed);

	m_W2 = Matd(m_layer1, m_layer2);
	m_W2.randn(0., 1./sqrt(m_layer1), seed);
	m_b2 = Matd::ones(m_layer2, 1);
	m_b2.randn(0, 1./sqrt(m_layer1), seed);

	m_W3 = Matd(m_layer2, m_layer3);
	m_W3.randn(0., 1./sqrt(m_layer2), seed);
	m_b3 = Matd::ones(m_layer3, 1);
	m_b3.randn(0, 1./sqrt(m_layer2), seed);

	m_W4 = Matd(m_layer3, m_y.cols);
	m_W4.randn(0., 1./sqrt(m_layer3), seed);
	m_b4 = Matd::ones(m_y.cols, 1);
	m_b4.randn(0, 1./sqrt(m_layer3), seed);

	m_dB1 = Matd::zeros(m_b1.size());
	m_dB2 = Matd::zeros(m_b2.size());
	m_dB3 = Matd::zeros(m_b3.size());
	m_dB4 = Matd::zeros(m_b4.size());
	m_dW1 = Matd::zeros(m_W1.size());
	m_dW2 = Matd::zeros(m_W2.size());
	m_dW3 = Matd::zeros(m_W3.size());
	m_dW4 = Matd::zeros(m_W4.size());
}
