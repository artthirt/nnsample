#include "mnist_conv.h"

#include "mnist_utils.h"

using namespace ct;

const int imageWidth = 28;
const int imageHeight = 28;

mnist_conv::mnist_conv()
{
	m_iteration = 0;
	m_mnist = 0;
	m_count_cnvW.push_back(8);
//	m_count_cnvW.push_back(2);
//	m_count_cnvW.push_back(3);
//	m_count_cnvW.push_back(4);
	m_conv_length = (int)m_count_cnvW.size();

	setConvLength(m_count_cnvW);
}

void mnist_conv::setMnist(mnist_reader *mnist)
{
	m_mnist = mnist;
}

void mnist_conv::setConvLength(const std::vector<int> &count_cnvW)
{
	if(!count_cnvW.size())
		return;
	m_conv_length = (int)count_cnvW.size();
	m_count_cnvW = count_cnvW;

	time_t tm;
	time(&tm);
	ct::generator.seed(tm);

	m_cnv.resize(m_conv_length);
	int prev = 1;
	ct::Size szA0(imageWidth, imageHeight);
	for(int i = 0; i < m_cnv.size(); ++i){
		m_cnv[i].resize(prev);
		for(int j = 0; j < m_cnv[i].size(); ++j){
			m_cnv[i][j].init(m_count_cnvW[i], szA0);
			m_cnv[i][j].setAlpha(0.1);
		}
		szA0 = m_cnv[i][0].szA2;
		prev = m_count_cnvW[i] * prev;
	}
}

std::vector<std::vector<convnn::convnn<float> > > &mnist_conv::cnv()
{
	return m_cnv;
}

std::vector<std::vector<Matf> > mnist_conv::cnvW()
{
	std::vector< std::vector < Matf > > res;

	res.resize(m_cnv.size());

	for(int i = 0; i < m_cnv.size(); ++i){
		for(int j = 0; j < m_cnv[i].size(); ++j){
			for(int k = 0; k < m_cnv[i][j].W.size(); ++k){
				res[i].push_back(m_cnv[i][j].W[k]);
			}
		}
	}
	return res;
}

Matf mnist_conv::forward(int index, int count)
{
	if(m_W.empty() || m_b.empty() || m_layers.empty())
		return Matf(0, 0);

	Matf X = m_mnist->X().getRows(index, count);

	return forward(X);
}

Matf mnist_conv::forward_test(int index, int count)
{
	if(!m_mnist || m_mnist->test().empty() || m_mnist->lb_test().empty())
		return Matf(0, 0);

	Matf X = Matf::zeros(count, m_mnist->X().cols);

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

void mnist_conv::setAlpha(double alpha)
{
	m_AdamOptimizer.setAlpha(alpha);
}

void mnist_conv::setLayers(const std::vector<int> &layers)
{
	m_layers = layers;
}

uint mnist_conv::iteration() const
{
	return m_iteration;
}

void mnist_conv::getEstimate(int batch, double &l2, double &accuracy)
{
	if(m_mnist->X().empty() || m_mnist->y().empty())
		return;

	Matf X;
	Matf yp;

	getXy(X, yp, batch);

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

void mnist_conv::getEstimateTest(int batch, double &l2, double &accuracy)
{
	if(!m_mnist || m_mnist->test().empty() || m_mnist->lb_test().empty())
		return;

	Matf X, yp;
	getXyTest(X, yp, batch);

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

void mnist_conv::pass_batch(int batch)
{
	if(!batch || !m_mnist || !m_mnist->train().size() || m_mnist->train().size() < batch)
		return;

	Matf X, y;

	getXy(X, y, batch);

	std::uniform_int_distribution<int> udtr(-3, 3);
	std::uniform_real_distribution<float> uar(-5, 5);

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

	pass_batch(X, y);

}

void mnist_conv::getX(Matf &X, int batch)
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

	X = m_mnist->X().getRows(indexes);
}

void mnist_conv::getXyTest(Matf &X, Matf &yp, int batch)
{
	if(batch < 0)
		batch = m_mnist->test().size();

	std::vector<int> indexes;

	getBatchIds(indexes, batch);

	X = Matf::zeros(batch, m_mnist->X().cols);
	yp = Matf::zeros(batch, m_mnist->y().cols);

	for(int i = 0; i < batch; i++){
		int id = indexes[i];
		QByteArray& data = m_mnist->test()[id];
		uint lb = m_mnist->lb_test()[id];

		for(int j = 0; j < data.size(); j++){
			X.at(i, j) = ((uint)data[j] > 0? 1. : 0.);
		}
		yp.at(i, lb) = 1.;
	}
}

void mnist_conv::getXy(Matf &X, Matf &y, int batch)
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

	X = m_mnist->X().getRows(indexes);
	y = m_mnist->y().getRows(indexes);
}

void mnist_conv::randX(Matf &X)
{
#if 1
	std::uniform_int_distribution<int> udtr(-3, 3);
	std::uniform_real_distribution<float> uar(-3, 3);

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

}

void mnist_conv::getBatchIds(std::vector<int> &indexes, int batch)
{
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

		for(int i = 0; i < batch; i++){
			indexes[i] = (int)i;
		}
	}
}

///*************

inline float reLu(float v)
{
	return std::max(v, 0.f);
}

inline float sigmoid(float v)
{
	return 1.f/std::exp(-v);
}

inline float deriv_sigm(float sigm)
{
	return sigm * (1 - sigm);
}

inline float grad_reLu(float v)
{
	return v > 0? 1 : 0;
}

template< typename T >
inline ct::Mat_<T>derivSigmoid(const ct::Mat_<T>& sigm)
{
	return sigm * (1.f - sigm);
}

///*************

void mnist_conv::init(int seed)
{
	if(!m_mnist || !m_layers.size() || !m_mnist->train().size() || !m_mnist->lb_train().size() || m_cnv.empty())
		return;

	m_cnv_out_size = m_cnv.back()[0].szA2;
	m_cnv_out_len = m_cnv.back().size() * m_cnv.back()[0].szA2.area() * m_count_cnvW.back();

	qDebug("--- input to MLP: %d ----", m_cnv_out_len);

	int input = m_cnv_out_len;
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

	if(!m_AdamOptimizer.init(m_layers, m_cnv_out_len)){
		std::cout << "optimizer not init\n";
	}
}

Matf mnist_conv::forward(const ct::Matf &X)
{
	if(m_W.empty() || m_b.empty() || m_layers.empty())
		return Matf(0, 0);

	Matf X_out;

	conv(X, X_out);

	Matf x = X_out, z, a;

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

void mnist_conv::conv(const Matf &X, Matf &X_out)
{
	if(X.empty())
		return;

	for(int i = 0; i < m_cnv.size(); ++i){
		std::vector< convnn::convnn< float > >& ls = m_cnv[i];

		if(i == 0){
			convnn::convnn< float >& m0 = ls[0];
			m0.forward(X, reLu);
		}else{
			for(int j = 0; j < m_cnv[i - 1].size(); ++j){
				int off1 = j * m_count_cnvW[i - 1];
				convnn::convnn< float >& m0 = m_cnv[i - 1][j];
				for(int k = 0; k < m_count_cnvW[i - 1]; ++k){
					int col = off1 + k;
					convnn::convnn< float >& mi = ls[col];
					mi.forward(m0.A2[k], reLu);
				}
			}
		}
	}

	convnn::convnn<float>::hconcat(m_cnv.back(), X_out);
}

void mnist_conv::pass_batch(const Matf &X, const Matf &y)
{
	if(m_W.empty() || m_b.empty() || m_layers.empty() ||
			m_layers.back() != y.cols){
		std::cout << "wrong parameters of model\n";
		return;
	}

	/// forward

	//// CONV

	Matf cnv_a;
	conv(X, cnv_a);

	//// MLP

	std::vector< Matf > z, a;

	z.resize(m_layers.size());
	a.resize(m_layers.size() + 1);

	a[0] = cnv_a;

	std::vector< Matf > D;
	Matf Wi;
	D.resize(3);

	for(size_t i = 0; i < m_layers.size(); i++){
		if(i < D.size()){
			dropout(m_W[i].rows, m_W[i].cols, 0.9f, D[i]);
			Wi = elemwiseMult(m_W[i], D[i]);
			z[i] = a[i] * Wi;
		}else{
			z[i] = a[i] * m_W[i];
		}
		z[i].biasPlus(m_b[i]);

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

	//// Backward

	for(int i = (int)m_layers.size() - 1; i > -1; --i){
//		Matf sz = elemwiseMult(a[i], a[i]);
//		sz = 1. - sz;
		Matf di, sz;

		{
			sz = derivRelu(a[i]);

			//Matf di = d * m_W[i].t();
			matmulT2(d, m_W[i], di);
			di = elemwiseMult(di, sz);
		}
		//dW[i] = a[i].t() * d;
		matmulT1(a[i], d, dW[i]);

		dW[i] *= 1./m;
		//dW[i] += (m_lambda/m * m_W[i]);

		if(i < D.size()){
			dW[i] = elemwiseMult(dW[i], D[i]);
		}

		dB[i] = (sumRows(d) * (1.f/m)).t();

		d = di;
	}

	/// convolution
	{
		std::vector< Matf > ds;

		nn::hsplit(d, m_cnv.back().size() * m_cnv.back()[0].W.size(), ds);

		for(int i = m_cnv.size() - 1; i > -1; i--){
			std::vector< convnn::convnn<float > >& lrs = m_cnv[i];
			std::vector< Matf > di;

			qDebug("LR[%d]-----", i);

			for(int j = 0; j < lrs.size(); ++j){
				convnn::convnn<float > &cnv = lrs[j];

				std::vector< Matf >dsi;

				for(int k = 0; k < cnv.W.size(); ++k){
					dsi.push_back(ds[j * cnv.W.size() + k]);
				}

				cnv.backward< Matf (*)(const Matf& mat) >(dsi, derivRelu);
				di.push_back(cnv.DltA0);
				for(int k = 0; k < cnv.W.size(); ++k){
					std::string sw = cnv.W[k];
					qDebug("W[%d:%d]:\n%s", j, k, sw.c_str());
				}
			}
			ds = di;

			qDebug("----");
		}
	}

	m_AdamOptimizer.pass(dW, dB, m_W, m_b);
	m_iteration = m_AdamOptimizer.iteration();
}
