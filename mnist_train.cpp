#include "mnist_train.h"

using namespace ct;

mnist_train::mnist_train()
{
	m_mnist = 0;
	m_lambda = 0.01f;
	m_AdamOptimizer.setAlpha(0.01f);
	m_AdamOptimizer.setBetha2(0.999f);
}

void mnist_train::setMnist(mnist_reader *mnist)
{
	m_mnist = mnist;
}

Matf mnist_train::forward(const ct::Matf &X) const
{
	if(m_W.empty() || m_b.empty() || m_layers.empty())
		return Matf(0, 0);
	Matf x = X, z, a;

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

Matf mnist_train::forward(int index, int count) const
{
	if(m_W.empty() || m_b.empty() || m_layers.empty())
		return Matf(0, 0);

	Matf X = m_X.getRows(index, count);

	return forward(X);
}

Matf mnist_train::forward_test(int index, int count) const
{
	if(!m_mnist || m_mnist->test().empty() || m_mnist->lb_test().empty())
		return Matf(0, 0);

	Matf X = Matf::zeros(count, m_X.cols);

	count = std::min(count, m_mnist->test().size() - index);

	for(int i = 0; i < count; i++){
		int id = index + i;
		QByteArray& data = m_mnist->test()[id];
		//uint lb = m_mnist->lb_test()[id];

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

	Matf X = m_X.getRows(indexes);
	Matf yp = m_y.getRows(indexes);

	Matf y = forward(X);

	double m = X.rows;

	Matf d = yp - y;

	Matf l2 = elemwiseMult(d, d);
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

	Matf X = Matf::zeros(batch, m_X.cols);
	Matf yp = Matf::zeros(batch, m_y.cols);

	for(int i = 0; i < batch; i++){
		int id = indexes[i];
		QByteArray& data = m_mnist->test()[id];
		uint lb = m_mnist->lb_test()[id];

		for(int j = 0; j < data.size(); j++){
			X.at(i, j) = ((uint)data[j] > 0? 1. : 0.);
		}
		yp.at(i, lb) = 1.;
	}

	Matf y = forward(X);

	double m = X.rows;

	Matf d = yp - y;

	Matf l2 = elemwiseMult(d, d);
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

	Matf X = m_X.getRows(indexes);
	Matf yp = m_y.getRows(indexes);

	Matf y = forward(X);

	float m = X.rows;

	Matf ce = elemwiseMult(yp, log(y));
	ce = ce + elemwiseMult(1.f - yp, log(1.f - y));
	ce = sumRows(ce);
	ce *= -1.f/m;
	return ce.sum();
}

void mnist_train::getEstimate(int batch, double &l2, double &accuracy)
{
	if(m_X.empty() || m_y.empty())
		return;

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

	Matf X = m_X.getRows(indexes);
	Matf yp = m_y.getRows(indexes);

	Matf y = forward(X);

	double m = X.rows;

	Matf d = yp - y;

	Matf ml2 = elemwiseMult(d, d);
	ml2 = sumRows(ml2);
	ml2 *= 1./m;

	//////////// l2
	l2 = ml2.sum();

	int right = 0;
	for(int i = 0; i < m; i++){
		int k1 = y.argmax(i, 1);
		int k2 = yp.argmax(i, 1);
		right += (int)(k1 == k2);
	}
	///////////// accuracy
	accuracy = (double)right / m;
}

void mnist_train::getEstimateTest(int batch, double &l2, double &accuracy)
{
	if(!m_mnist || m_mnist->test().empty() || m_mnist->lb_test().empty())
		return;

	std::vector<int> indexes;

	if(batch > 0){

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

	}else{
		batch = m_mnist->test().size();
		indexes.resize(batch);

		for(size_t i = 0; i < batch; i++){
			indexes[i] = (int)i;
		}
	}

	Matf X = Matf::zeros(batch, m_X.cols);
	Matf yp = Matf::zeros(batch, m_y.cols);

	for(int i = 0; i < batch; i++){
		int id = indexes[i];
		QByteArray& data = m_mnist->test()[id];
		uint lb = m_mnist->lb_test()[id];

		for(int j = 0; j < data.size(); j++){
			X.at(i, j) = ((uint)data[j] > 0? 1. : 0.);
		}
		yp.at(i, lb) = 1.;
	}

	Matf y = forward(X);

	float m = X.rows;

	Matf d = yp - y;

	Matf ml2 = elemwiseMult(d, d);
	ml2 = sumRows(ml2);
	ml2 *= 1.f/m;

	//////////// l2
	l2 = ml2.sum();

	int right = 0;
	for(int i = 0; i < m; i++){
		int k1 = y.argmax(i, 1);
		int k2 = yp.argmax(i, 1);
		right += (int)(k1 == k2);
	}
	///////////// accuracy
	accuracy = (double)right / m;
}

void mnist_train::init(int seed)
{
	if(!m_mnist || !m_layers.size() || !m_mnist->train().size() || !m_mnist->lb_train().size())
		return;

	const int out_cols = 10;

	m_X = Matf::zeros(m_mnist->train().size(), m_mnist->train()[0].size());
	m_y = Matf::zeros(m_mnist->lb_train().size(), out_cols);

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

		m_W[i] = Matf(input, output);
		m_W[i].randn(0., n, seed);
		m_b[i] = Matf::ones(output, 1);
		m_b[i].randn(0, n, seed);

		input = output;
	}

	if(!m_AdamOptimizer.init(m_layers, m_X.cols)){
		std::cout << "optimizer not init\n";
	}
}

template< typename T >
void translate(int x, int y, int w, int h, T *X)
{
	std::vector<T>d;
	d.resize(w * h);

#pragma omp parallel for
	for(int i = 0; i < h; i++){
		int newi = i + x;
		if(newi >= 0 && newi < h){
			for(int j = 0; j < w; j++){
				int newj = j + y;
				if(newj >= 0 && newj < w){
					d[newi * w + newj] = X[i * w + j];
				}
			}
		}
	}
	for(int i = 0; i < d.size(); i++){
		X[i] = d[i];
	}
}

template< typename T >
void rotate_mnist(int w, int h, T angle, T *X)
{
	T cw = w / 2;
	T ch = h / 2;

	std::vector<T> d;
	d.resize(w * h);

	for(int y = 0; y < h; y++){
		for(int x = 0; x < w; x++){
			T x1 = x - cw;
			T y1 = y - ch;

			T nx = x1 * cos(angle) + y1 * sin(angle);
			T ny = -x1 * sin(angle) + y1 * cos(angle);
			nx += cw; ny += ch;
			int ix = nx, iy = ny;
			if(ix >= 0 && ix < w && iy >= 0 && iy < h){
				T c = X[y * w + x];
				d[iy * w + ix] = c;
			}
		}
	}
	for(int i = 0; i < d.size(); i++){
		X[i] = d[i];
	}
}

void mnist_train::pass_batch(int batch)
{
	if(!batch || !m_mnist || !m_mnist->train().size() || m_mnist->train().size() < batch)
		return;

	Matf X, y;

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

#if 1
	std::uniform_int_distribution<int> udtr(-5, 5);
	std::uniform_real_distribution<float> uar(-7, 7);

#pragma omp parallel for
	for(int i = 0; i < X.rows; i++){
		float *Xi = &X.at(i, 0);
		int x = udtr(m_generator);
		int y = udtr(m_generator);
		float ang = uar(m_generator);
		ang = angle2rad(ang);

		rotate_mnist<float>(28, 28, ang, Xi);
		translate<float>(x, y, 28, 28, Xi);
	}
#endif
	pass_batch(X, y);
}

void mnist_train::pass_batch(const Matf &X, const Matf &y)
{
	if(m_W.empty() || m_b.empty() || m_layers.empty() ||
			m_layers.back() != y.cols){
		std::cout << "wrong parameters of model\n";
		return;
	}

	/// forward

	std::vector< Matf > z, a;
	z.resize(m_layers.size());
	a.resize(m_layers.size() + 1);

	a[0] = X;

	Matf D1, Dt1, D2, Dt2, D3, Dt3;

	for(size_t i = 0; i < m_layers.size(); i++){
		z[i] = a[i] * m_W[i];
		z[i].biasPlus(m_b[i]);

		if(i == 0){
			dropout(z[i], 0.5f, D1, Dt1);
		}
		if(i == 1){
			dropout(z[i], 0.5f, D2, Dt2);
		}
		if(i == 2){
			dropout(z[i], 0.5f, D3, Dt3);
		}
		if(i < m_layers.size() - 1){
			a[i + 1] = relu(z[i]);
		}else
			a[i + 1] = softmax(z[i], 1);
	}

	std::vector< Matf > dW, dB;
	dW.resize(m_layers.size());
	dB.resize(m_layers.size());

	float m = X.rows;
	Matf d = a.back() - y;

	/// backward

	for(int i = (int)m_layers.size() - 1; i > -1; --i){
//		Matf sz = elemwiseMult(a[i], a[i]);
//		sz = 1. - sz;
		Matf sz = derivRelu(a[i]);

		//Matf di = d * m_W[i].t();
		Matf di;
		matmulT2(d, m_W[i], di);
		di = elemwiseMult(di, sz);
		//dW[i] = a[i].t() * d;
		matmulT1(a[i], d, dW[i]);
		dW[i] *= 1./m;
		dW[i] += (m_lambda/m * m_W[i]);

		if(i == 2){
			dropout_transpose(dW[i], Dt3);
		}
		if(i == 1){
			dropout_transpose(dW[i], Dt2);
		}
		if(i == 0){
			dropout_transpose(dW[i], Dt1);
		}

		dB[i] = (sumRows(d) * (1.f/m)).t();
		d = di;
	}

	if(!m_AdamOptimizer.pass(dW, dB, m_W, m_b)){
		std::cout << "optimizer not work\n";
	}
}
