#include "nnmodel.h"
#include  <map>
#include <iostream>
#include "matops.h"

using namespace ct;

nnmodel::nnmodel()
	: m_resultModel(ESquarError)
{
}

void nnmodel::setData(const ct::Matf &X, const ct::Matf &y)
{
	if(X.rows != y.rows)
		return;
	m_X = X;
	m_y = y;
}

void nnmodel::setAlpha(double alpha)
{
	m_AdamOptimizer.setAlpha(alpha);
}

double nnmodel::alpha() const
{
	return m_AdamOptimizer.alpha();
}

uint32_t nnmodel::iteration() const
{
	return m_AdamOptimizer.iteration();
}

void nnmodel::setBetha1(double v)
{
	m_AdamOptimizer.setBetha1(v);
}

void nnmodel::setBetha2(double v)
{
	m_AdamOptimizer.setBetha2(v);
}

//ct::Matf nnmodel::forward(const ct::Matf &X) const
//{
//	Matf z2 = X * m_W1;			/// [1000 2] * [2 5] = [1000 5]
//	z2.biasPlus(m_b1);				///
//	Matf a2 = tanh(z2);			/// [1000 5]
//	Matf z3 = a2 * m_W2;			/// [1000 5] * [5 4] = [1000 4]
//	z3.biasPlus(m_b2);				///
//	Matf a3 = tanh(z3);			/// [1000 4]
//	Matf z4 = a3 * m_W3;			/// [1000 4] * [4 1] = [1000 1]
//	z4.biasPlus(m_b3);			///
//	Matf a4 = tanh(z4);					/// [1000 1]
//	Matf z5 = a4 * m_W4;
//	z5.biasPlus(m_b4);
//	return z5;
//}

double nnmodel::L2()
{
	Matf y = forward_model(m_X);
	Matf d = m_y - y;
	elemwiseMult(d, d);
	double L2 = d.sum() * 1./d.rows;

	return L2;
}

Matf nnmodel::w(int index)
{
	return m_mlp[index].W;//m_W[index];
}

Matf nnmodel::b(int index)
{
	return m_mlp[index].B;//m_b[index];
}

int nnmodel::count() const
{
	return (int)m_mlp.size();
}

//void nnmodel::pass()
//{
//	pass_batch(m_X, m_y);
//}

//void nnmodel::pass_batch(int batch)
//{
//	if(!m_X.total() || !m_y.total() || m_X.rows != m_y.rows || !batch || batch > m_X.rows)
//		return;

//	Matf X(batch, m_X.cols), y(batch, m_y.cols);

//	std::vector<int> indexes;
//	indexes.resize(batch);
//	std::uniform_int_distribution<int> ud(0, m_X.rows - 1);
//	std::map<int, bool> set;
//	for(int i = 0; i < batch; i++){
//		int v = ud(m_generator);
//		while(set.find(v) != set.end()){
//			v = ud(m_generator);
//		}
//		set[v] = true;
//		indexes[i] = v;
//	}
//	X = m_X.getRows(indexes);
//	y = m_y.getRows(indexes);

//	pass_batch(X, y);

////	y = forward(m_X);
////	Matf d = m_y - y;
////	d = elemwiseMult(d, d);
////	m_L2 = d.sum() * 1./d.rows;
//}

//void nnmodel::pass_batch(const Matf &X, const Matf y)
//{
//	if(!m_W1.total() || !m_W2.total()
//			|| !m_b1.total() || !m_b2.total()
//			|| !X.total() || !y.total()
//			|| m_W1.rows != X.cols)
//		return;

//	/// forward
//	Matf z2 = X * m_W1;				/// [1000 2] * [2 5] = [1000 5]
//	z2.biasPlus(m_b1);				///
//	Matf a2 = tanh(z2);				/// [1000 5]
//	Matf z3 = a2 * m_W2;			/// [1000 5] * [5 4] = [1000 4]
//	z3.biasPlus(m_b2);				///
//	Matf a3 = tanh(z3);				/// [1000 4]
//	Matf z4 = a3 * m_W3;			/// [1000 4] * [4 3] = [1000 3]
//	z4.biasPlus(m_b3);
//	Matf a4 = tanh(z4);				/// [1000 3]
//	Matf z5 = a4 * m_W4;			/// [1000 3] * [3 1] = [1000 1]
//	z5.biasPlus(m_b4);
//	Matf a5 = z5;

//	/// backward
//	Matf d5 = a5 - y;

//	Matf sz4 = elemwiseMult(a4, a4);
//	sz4 = 1. - sz4;
//	Matf sz3 = elemwiseMult(a3, a3);
//	sz3 = 1. - sz3;
//	Matf sz2 = elemwiseMult(a2, a2);
//	sz2 = 1. - sz2;

//	int m = X.rows;

//	Matf dW4, dW3, dW2, dW1;
//	Matf dB4, dB3, dB2, dB1;
////	for(int i = 0; i < m; i++){

////	}
//	/// layer 5 ->4
//	Matf d4 = d5 * m_W4.t();
//	d4 = elemwiseMult(d4, sz4);
//	dW4 = a4.t() * d5;
//	dW4 *= 1./m;
//	dB4 = (sumRows(d5) * (1./m)).t();

//	/// layer 4 -> 3
//	Matf d3 = d4 * m_W3.t();		/// [1000 1] * [1 4] = [1000 4]
//	d3 = elemwiseMult(d3, sz3);		/// [1000 4]
//	dW3 = a3.t() * d4;				/// [4 1000] * [1000 1] = [4 1]
//	dW3 *= 1./m;
//	dB3 = (sumRows(d4) * (1./m)).t();

//	/// layer 3 -> 2
//	Matf d2 = d3 * m_W2.t();		///
//	d2 = elemwiseMult(d2, sz2);		///
//	dW2 = a2.t() * d3;				///
//	dW2 *= 1./m;
//	dB2 = (sumRows(d3) * (1./m)).t();

//	/// layer 2 -> 1
//	dW1 = X.t() * d2;				///
//	dW1 *= 1./m;
//	dB1 = (sumRows(d2) * (1./m)).t();

//	m_dB1 = (m_betha1 * m_dB1) + (1 - m_betha1) * (dB1);
//	m_dB2 = (m_betha1 * m_dB2) + (1 - m_betha1) * (dB2);
//	m_dB3 = (m_betha1 * m_dB3) + (1 - m_betha1) * (dB3);
//	m_dB4 = (m_betha1 * m_dB4) + (1 - m_betha1) * (dB4);
//	m_dW1 = (m_betha1 * m_dW1) + (1 - m_betha1) * (dW1);
//	m_dW2 = (m_betha1 * m_dW2) + (1 - m_betha1) * (dW2);
//	m_dW3 = (m_betha1 * m_dW3) + (1 - m_betha1) * (dW3);
//	m_dW4 = (m_betha1 * m_dW4) + (1 - m_betha1) * (dW4);

//	m_W1 -= m_alpha * (m_dW1);
//	m_b1 -= m_alpha * (m_dB1);
//	m_W2 -= m_alpha * (m_dW2);
//	m_b2 -= m_alpha * (m_dB2);
//	m_W3 -= m_alpha * (m_dW3);
//	m_b3 -= m_alpha * (m_dB3);
//	m_W4 -= m_alpha * (m_dW4);
//	m_b4 -= m_alpha * (m_dB4);
//}

Matf nnmodel::resultModel(Matf &m)
{
	switch (m_resultModel) {
		case ESigmoid:
			return sigmoid(m);
		default:
			return m;
	}
}

Matf nnmodel::diffModel(Matf &m)
{
	switch (m_resultModel) {
		case ESigmoid:
		{
			Matf s = m * (1.f - m);
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

	m_mlp.resize(m_layers.size());
//	m_W.resize(m_layers.size());
//	m_b.resize(m_layers.size());

	for(size_t i = 0; i < m_layers.size(); i++){
		output = m_layers[i];

		m_mlp[i].init(input, output, i == m_layers.size() - 1? ct::LINEAR : ct::LEAKYRELU);
//		double n = 1./sqrt(input);

//		m_W[i] = Matf(input, output);
//		m_W[i].randn(0., n, seed);
//		m_b[i] = Matf::ones(output, 1);
//		m_b[i].randn(0, n, seed);

		input = output;
	}

	if(!m_AdamOptimizer.init(m_mlp)){
		std::cout << "optimizer not init\n";
	}
}

Matf nnmodel::forward_model(const Matf &X)
{
	if(m_mlp.empty() || m_layers.empty() || m_layers.back() != m_y.cols)
		return Matf(0, 0);
	Matf *x = (Matf*)&X, z, a;

	for(size_t i = 0; i < m_layers.size(); i++){
		m_mlp[i].forward(x);
		x = &m_mlp[i].Y();
//		z = x * m_W[i];
//		z.biasPlus(m_b[i]);
//		if(i < m_layers.size() - 1){
//			a = relu(z);
//			x = a;
//		}else
//			a = z;
	}
	m_mlp.back().Y().copyTo(a);
	return a;
}

void nnmodel::pass_batch_model(const Matf &X, const Matf y)
{
	if(m_mlp.empty() || m_layers.empty() ||
			m_layers.back() != m_y.cols){
		std::cout << "wrong parameters of model\n";
		return;
	}

	/// forward

	ct::Matf a = forward_model(X);

	Matf d = a - y;

	/// backward

	Matf *pd = &d;
	for(int i = (int)m_layers.size() - 1; i > -1; --i){
//		Matf sz = elemwiseMult(a[i], a[i]);
//		sz = 1. - sz;
//		Matf sz = derivRelu(a[i]);

//		Matf di = d * m_W[i].t();
//		elemwiseMult(di, sz);
//		dW[i] = a[i].t() * d;
//		dW[i] *= 1./m;
//		dB[i] = (sumRows(d) * (1./m)).t();
//		d = di;
		m_mlp[i].backward(*pd);
		pd = &m_mlp[i].DltA0;
	}

	if(!m_AdamOptimizer.pass(m_mlp)){
		std::cout << "optimizer not work\n";
	}
}

void nnmodel::pass_batch_model(int batch)
{
	if(!m_X.total() || !m_y.total() || m_X.rows != m_y.rows || !batch || batch > m_X.rows)
		return;

	Matf X(batch, m_X.cols), y(batch, m_y.cols);

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

