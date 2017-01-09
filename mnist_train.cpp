#include "mnist_train.h"

using namespace ct;

mnist_train::mnist_train()
{
	m_mnist = 0;
	m_AdamOptimizer.setAlpha(0.01);
	m_AdamOptimizer.setBetha2(0.99);
}

void mnist_train::setMnist(mnist_reader *mnist)
{
	m_mnist = mnist;
}

ct::Matd mnist_train::forward(const ct::Matd &X) const
{
	if(m_W.empty() || m_b.empty() || m_layers.empty())
		return Matd(0, 0);
	Matd x = X, z, a;

	for(size_t i = 0; i < m_layers.size(); i++){
		z = x * m_W[i];
		z.biasPlus(m_b[i]);
		if(i < m_layers.size() - 1){
			a = relu(z);
			x = a;
		}else
			a = softmax(z, 1);
	}
	return a;
}

Matd mnist_train::forward(int index, int count) const
{
	if(m_W.empty() || m_b.empty() || m_layers.empty())
		return Matd(0, 0);

	Matd X = m_X.getRows(index, count);

	return forward(X);
}

Matd mnist_train::forward_test(int index, int count) const
{
	if(!m_mnist || m_mnist->test().empty() || m_mnist->lb_test().empty())
		return Matd(0, 0);

	Matd X = Matd::zeros(count, m_X.cols);

	count = std::min(count, m_mnist->test().size() - index);

	for(int i = 0; i < count; i++){
		int id = index + i;
		QByteArray& data = m_mnist->test()[id];
		uint lb = m_mnist->lb_test()[id];

		for(int j = 0; j < data.size(); j++){
			X.at(i, j) = ((uint)data[j] > 0? 1. : 0.);
		}
	}

	return forward(X);
}

void mnist_train::setAlpha(double alpha)
{
	m_AdamOptimizer.setAlpha(alpha);
}

void mnist_train::setLayers(const std::vector<int> &layers)
{
	m_layers = layers;
}

uint mnist_train::iteration() const
{
	return m_AdamOptimizer.iteration();
}

double mnist_train::L2(int batch)
{
	std::vector<int> indexes;
	indexes.resize(batch);
	std::uniform_int_distribution<int> ud(0, m_mnist->train().size() - 1);
	std::map<int, bool> set;
	for(int i = 0; i < batch; i++){
		int v = ud(m_generator);
		while(set.find(v) != set.end()){
			v = ud(m_generator);
		}
		set[v] = true;
		indexes[i] = v;
	}

	Matd X = m_X.getRows(indexes);
	Matd yp = m_y.getRows(indexes);

	Matd y = forward(X);

	double m = X.rows;

	Matd d = yp - y;

	Matd l2 = elemwiseMult(d, d);
	l2 = sumRows(l2);
	l2 *= 1./m;
	return l2.sum();
}

double mnist_train::L2test(int batch)
{
	if(!m_mnist || m_mnist->test().empty() || m_mnist->lb_test().empty())
		return -1;

	std::vector<int> indexes;
	indexes.resize(batch);
	std::uniform_int_distribution<int> ud(0, m_mnist->test().size() - 1);
	std::map<int, bool> set;
	for(int i = 0; i < batch; i++){
		int v = ud(m_generator);
		while(set.find(v) != set.end()){
			v = ud(m_generator);
		}
		set[v] = true;
		indexes[i] = v;
	}

	Matd X = Matd::zeros(batch, m_X.cols);
	Matd yp = Matd::zeros(batch, m_y.cols);

	for(int i = 0; i < batch; i++){
		int id = indexes[i];
		QByteArray& data = m_mnist->test()[id];
		uint lb = m_mnist->lb_test()[id];

		for(int j = 0; j < data.size(); j++){
			X.at(i, j) = ((uint)data[j] > 0? 1. : 0.);
		}
		yp.at(i, lb) = 1.;
	}

	Matd y = forward(X);

	double m = X.rows;

	Matd d = yp - y;

	Matd l2 = elemwiseMult(d, d);
	l2 = sumRows(l2);
	l2 *= 1./m;
	return l2.sum();
}

double mnist_train::cross_entropy(int batch)
{
	std::vector<int> indexes;
	indexes.resize(batch);
	std::uniform_int_distribution<int> ud(0, m_mnist->train().size() - 1);
	std::map<int, bool> set;
	for(int i = 0; i < batch; i++){
		int v = ud(m_generator);
		while(set.find(v) != set.end()){
			v = ud(m_generator);
		}
		set[v] = true;
		indexes[i] = v;
	}

	Matd X = m_X.getRows(indexes);
	Matd yp = m_y.getRows(indexes);

	Matd y = forward(X);

	double m = X.rows;

	Matd ce = elemwiseMult(yp, log(y));
	ce = ce + elemwiseMult(1. - yp, log(1. - y));
	ce = sumRows(ce);
	ce *= -1./m;
	return ce.sum();
}

void mnist_train::init(int seed)
{
	if(!m_mnist || !m_layers.size() || !m_mnist->train().size() || !m_mnist->lb_train().size())
		return;

	const int out_cols = 10;

	m_X = Matd::zeros(m_mnist->train().size(), m_mnist->train()[0].size());
	m_y = Matd::zeros(m_mnist->lb_train().size(), out_cols);

	for(int i = 0; i < m_mnist->train().size(); i++){
		int yi = m_mnist->lb_train()[i];
		m_y.at(i, yi) = 1.;

		QByteArray &data = m_mnist->train()[i];
		for(int j = 0; j < data.size(); j++){
			m_X.at(i, j) = ((uint)data[j] > 0)? 1. : 0.;
		}
	}

	int input = m_X.cols;
	int output = m_layers[0];

	m_W.resize(m_layers.size());
	m_b.resize(m_layers.size());

	for(size_t i = 0; i < m_layers.size(); i++){
		output = m_layers[i];

		double n = 1./sqrt(input);

		m_W[i] = Matd(input, output);
		m_W[i].randn(0., n, seed);
		m_b[i] = Matd::ones(output, 1);
		m_b[i].randn(0, n, seed);

		input = output;
	}

	if(!m_AdamOptimizer.init(m_layers, m_X.cols)){
		std::cout << "optimizer not init\n";
	}
}

void mnist_train::pass_batch(int batch)
{
	if(!batch || !m_mnist || !m_mnist->train().size() || m_mnist->train().size() < batch)
		return;

	Matd X, y;

	std::vector<int> indexes;
	indexes.resize(batch);
	std::uniform_int_distribution<int> ud(0, m_mnist->train().size() - 1);
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
}

void mnist_train::pass_batch(const Matd &X, const Matd &y)
{
	if(m_W.empty() || m_b.empty() || m_layers.empty() ||
			m_layers.back() != y.cols){
		std::cout << "wrong parameters of model\n";
		return;
	}

	/// forward

	std::vector< Matd > z, a;
	z.resize(m_layers.size());
	a.resize(m_layers.size() + 1);

	a[0] = X;
	for(size_t i = 0; i < m_layers.size(); i++){
		z[i] = a[i] * m_W[i];
		z[i].biasPlus(m_b[i]);
		if(i < m_layers.size() - 1){
			a[i + 1] = relu(z[i]);
		}else
			a[i + 1] = softmax(z[i], 1);
	}

	std::vector< Matd > dW, dB;
	dW.resize(m_layers.size());
	dB.resize(m_layers.size());

	double m = X.rows;
	Matd d = a.back() - y;

	/// backward

	for(int i = m_layers.size() - 1; i > -1; --i){
//		Matd sz = elemwiseMult(a[i], a[i]);
//		sz = 1. - sz;
		Matd sz = derivRelu(a[i]);

		Matd di = d * m_W[i].t();
		di = elemwiseMult(di, sz);
		dW[i] = a[i].t() * d;
		dW[i] *= 1./m;
		dB[i] = (sumRows(d) * (1./m)).t();
		d = di;
	}

	if(!m_AdamOptimizer.pass(dW, dB, m_W, m_b)){
		std::cout << "optimizer not work\n";
	}
}
