#include "mnist_conv.h"

#include "mnist_utils.h"

using namespace ct;

mnist_conv::mnist_conv()
{
	m_iteration = 0;
	m_mnist = 0;
	m_count_cnvW.push_back(10);
	m_count_cnvW.push_back(10);
	m_count_cnvW.push_back(10);
	m_conv_length = m_count_cnvW.size();

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
	m_conv_length = count_cnvW.size();
	m_cnvW.resize(m_conv_length);
	m_cnvB.resize(m_conv_length);
	m_count_cnvW = count_cnvW;

	time_t tm;
	time(&tm);
	ct::generator.seed(tm);

	std::normal_distribution<float> nrm(0, 0.01);

	for(int i = 0; i < m_conv_length; ++i){
		m_cnvW[i].resize(m_count_cnvW[i]);
		m_cnvB[i].resize(m_count_cnvW[i]);

		for(int j = 0; j < m_cnvW[i].size(); ++j){
			Matf &W = m_cnvW[i][j];
			W = Matf::zeros(3, 3);
			W.randn(0, 0.1);

			m_cnvB[i][j] = nrm(ct::generator);
		}
	}
	m_MomentOptimizer.resize(m_conv_length);
//	for(int i = 0; i < m_MomentOptimizer.size(); ++i){
//		m_MomentOptimizer[i].setAlpha(0.1);
//	}
}

std::vector<std::vector<Matf> > &mnist_conv::cnvW()
{
	return m_cnvW;
}

Matf mnist_conv::forward(const ct::Matf &X) const
{
	if(m_W.empty() || m_b.empty() || m_layers.empty())
		return Matf(0, 0);

	Matf X_out;

	conv(X, X_out, 28, 28);

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

	std::vector<int> indexes;

	getBatchIds(indexes);

	Matf X = Matf::zeros(batch, m_mnist->X().cols);
	Matf yp = Matf::zeros(batch, m_mnist->y().cols);

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

void mnist_conv::init(int seed)
{
	if(!m_mnist || !m_layers.size() || !m_mnist->train().size() || !m_mnist->lb_train().size() || m_cnvW.empty())
		return;

	m_cnv_out_size = ct::Size(28 - 2 * m_conv_length, 28 - 2 * m_conv_length);

	int input = m_cnv_out_size.area();
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

	if(!m_AdamOptimizer.init(m_layers, m_cnv_out_size.area())){
		std::cout << "optimizer not init\n";
	}
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

inline float reLu(float v)
{
	return std::max(v, 0.f);
}

void mnist_conv::conv(const Matf &X, Matf &X_out, int w, int h)const
{
	if(X.empty())
		return;

	Matf a = X;
	Mati indexes;

	std::vector< ct::Matf > Res;

	ct::Size sz(w, h);
	for(int i = 0; i < m_conv_length; ++i){
		sz = nn::conv2D(a, sz, 1, m_cnvW[i], m_cnvB[i], Res, reLu);
		nn::max_pool(Res, a, indexes);
	}
	X_out = a;
}

void mnist_conv::pass_batch(const Matf &X, const Matf &y)
{
	if(m_W.empty() || m_b.empty() || m_layers.empty() ||
			m_layers.back() != y.cols){
		std::cout << "wrong parameters of model\n";
		return;
	}

	std::vector< Matf > z, a;

	/// forward

	////CONV
	int w = 28;
	int h = 28;

	std::vector< Matf > cnv_a;
	std::vector< Mati > indexes;

	std::vector< std::vector< ct::Matf > > Res;
	std::vector< ct::Size > szs;

	cnv_a.resize(m_conv_length + 1);
	Res.resize(m_conv_length);
	indexes.resize(m_conv_length);
	szs.resize(m_conv_length + 1);

	ct::Size sz(w, h);

	cnv_a[0] = X;
	szs[0] = sz;

	for(int i = 0; i < m_conv_length; ++i){
		sz = nn::conv2D(cnv_a[i], sz, 1, m_cnvW[i], m_cnvB[i], Res[i], reLu);
		szs[i + 1] = sz;
		nn::max_pool(Res[i], cnv_a[i + 1], indexes[i]);
	}

	////

	z.resize(m_layers.size());
	a.resize(m_layers.size() + 1);

	a[0] = cnv_a.back();

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
		/*if(i > 0)*/{
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

//		if(i > 0)
		d = di;
	}

	//// deriv conv
	{
		std::vector< std::vector<Matf> > gradW;
		std::vector< std::vector<float> > gradB;
		gradW.resize(m_cnvW.size());
		gradB.resize(m_cnvW.size());

		for(int i = m_cnvW.size() - 1;i > -1; --i){
			Matf di, sz, di2;
			if(i > 0){
//				if(i != m_cnvW.size() - 1){
					sz = derivRelu(cnv_a[i + 1]);
					di = elemwiseMult(d, sz);
//				}else{
//					di = d;
//				}
				nn::deriv_prev_cnv(di, m_cnvW[i], indexes[i], szs[i + 1], szs[i], di2);
				di = di2;
			}

			ct::Size szW(m_cnvW[i][0].cols, m_cnvW[i][0].rows);

			nn::deriv_conv2D(cnv_a[i], d, indexes[i], szs[i], szs[i + 1], szW, m_cnvW[i].size(), 1, gradW[i], gradB[i]);

			if(i > 0)
				d = di;
		}

		for(int i = 0; i < gradW.size(); ++i){
			qDebug("-----L[%d]----\n", i);
			m_MomentOptimizer[i].pass(gradW[i], gradB[i], m_cnvW[i], m_cnvB[i]);
			for(int j = 0; j < gradW[i].size(); ++j){
				std::string str = m_cnvW[i][j];
				//qDebug("cnvW[%d]\n%s", j, str.c_str());
				qDebug("cnvB[%d] = %f", j, m_cnvB[i][j]);
			}
		}
		qDebug("---------");

	}

	m_AdamOptimizer.pass(dW, dB, m_W, m_b);
	m_iteration = m_AdamOptimizer.iteration();
}
