#include "mnist_conv.h"

#include "mnist_utils.h"

#include "qt_work_mat.h"

using namespace ct;

const int imageWidth = 28;
const int imageHeight = 28;

mnist_conv::mnist_conv()
{
	m_iteration = 0;
	m_mnist = 0;
	m_count_cnvW.push_back(8);
	m_count_cnvW.push_back(3);
//	m_count_cnvW.push_back(1);
	m_conv_length = (int)m_count_cnvW.size();
	m_seed = 1;

	setConvLength(m_count_cnvW);
}

void mnist_conv::setMnist(mnist_reader *mnist)
{
	m_mnist = mnist;
}

void mnist_conv::setConvLength(const std::vector<int> &count_cnvW, std::vector<int> *weight_sizes)
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
	for(size_t i = 0; i < m_cnv.size(); ++i){
		m_cnv[i].resize(prev);
		for(size_t j = 0; j < m_cnv[i].size(); ++j){

			if(weight_sizes && weight_sizes->size() == m_count_cnvW.size()){
				m_cnv[i][j].setWeightSize((*weight_sizes)[i]);
			}

			m_cnv[i][j].init(m_count_cnvW[i], szA0);
		}
		szA0 = m_cnv[i][0].szA2;
		prev = m_count_cnvW[i] * prev;
	}

#ifdef _USE_GPU
	m_gpu_model.setConvLength(count_cnvW, weight_sizes);
#endif
}

std::vector<std::vector<convnn::convnn<float> > > &mnist_conv::cnv()
{
	return m_cnv;
}

std::vector<std::vector<Matf> > mnist_conv::cnvW()
{
	std::vector< std::vector < Matf > > res;

	if(m_use_gpu){
		res.resize(m_gpu_model.cnv().size());

		for(size_t i = 0; i < m_gpu_model.cnv().size(); ++i){
			for(size_t j = 0; j < m_gpu_model.cnv()[i].size(); ++j){
				for(size_t k = 0; k < m_gpu_model.cnv()[i][j].W.size(); ++k){
					ct::Matf Wf;
					gpumat::convert_to_mat(m_gpu_model.cnv()[i][j].W[k], Wf);
					res[i].push_back(Wf);
				}
			}
		}
	}else{
		res.resize(m_cnv.size());

		for(size_t i = 0; i < m_cnv.size(); ++i){
			for(size_t j = 0; j < m_cnv[i].size(); ++j){
				for(size_t k = 0; k < m_cnv[i][j].W.size(); ++k){
					res[i].push_back(m_cnv[i][j].W[k]);
				}
			}
		}
	}
	return res;
}

std::vector<ct::Matf> mnist_conv::getA() const
{
	return m_a;
}

void mnist_conv::setA(const std::vector<ct::Matf> &value)
{
	m_a = value;
}

Matf mnist_conv::forward(int index, int count, bool use_gpu)
{
	if(m_W.empty() || m_b.empty() || m_layers.empty())
		return Matf(0, 0);

	Matf X = m_mnist->X().getRows(index, count);

	return forward(X, use_gpu);
}

Matf mnist_conv::forward_test(int index, int count, bool use_gpu)
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

	return forward(X, use_gpu);
}

void mnist_conv::setAlpha(double alpha)
{
	m_AdamOptimizer.setAlpha(alpha);

	for(size_t i = 0; i < m_cnv.size(); ++i){
		for(size_t j = 0; j < m_cnv[i].size(); ++j){
			m_cnv[i][j].setAlpha(alpha);
		}
	}

	m_gpu_model.setAlpha(alpha);
}

void mnist_conv::setLayers(const std::vector<int> &layers)
{
	m_layers = layers;

	m_gpu_model.setLayers(layers);
}

uint mnist_conv::iteration() const
{
	return m_use_gpu? m_gpu_model.iteration() : m_iteration;
}

void mnist_conv::getEstimate(int batch, double &l2, double &accuracy, bool use_gpu)
{
	if(m_mnist->X().empty() || m_mnist->y().empty())
		return;

	Matf X;
	Matf yp;

	getXy(X, yp, batch);

	Matf y = forward(X, use_gpu);

	double m = X.rows;

	Matf d = yp - y;

	elemwiseMult(d, d);
	d = sumRows(d);
	d *= 1./m;

	//////////// l2
	l2 = d.sum();

	int right = 0;
	for(int i = 0; i < m; i++){
		int k1 = y.argmax(i, 1);
		int k2 = yp.argmax(i, 1);
		right += (int)(k1 == k2);
	}
	///////////// accuracy
	accuracy = (double)right / m;
}

void mnist_conv::getEstimateTest(double &l2, double &accuracy, bool use_gpu)
{
	if(!m_mnist || m_mnist->test().empty() || m_mnist->lb_test().empty())
		return;

	size_t batch_count = 10;
	size_t test_size = m_mnist->test().size();

	size_t batch_size = test_size / batch_count;

	l2 = 0;
	int right = 0;

	for(size_t i = 0; i < batch_count; ++i){

		Matf X, yp;
		getXyTest(X, yp, batch_size, false, i * batch_size);

		Matf y = forward(X, use_gpu);

		Matf d = yp - y;

		int m = X.rows;

		elemwiseMult(d, d);
		d = sumRows(d);
		d *= 1.f/m;

		//////////// l2
		l2 += d.sum();

		for(int i = 0; i < m; i++){
			int k1 = y.argmax(i, 1);
			int k2 = yp.argmax(i, 1);
			right += (int)(k1 == k2);
		}

	}

	l2 /= (double)batch_count;

	///////////// accuracy
	accuracy = (double)right / test_size;
}

void mnist_conv::pass_batch(int batch, bool use_gpu)
{
	if(!batch || !m_mnist || !m_mnist->train().size() || m_mnist->train().size() < batch)
		return;

	Matf X, y;

	getXy(X, y, batch);

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

	m_use_gpu = use_gpu;
	if(!use_gpu){
		pass_batch(X, y);
	}else{
		gpumat::convert_to_gpu(X, gX);
		gpumat::convert_to_gpu(y, gY);

		if(!m_gpu_model.isInit()){
			m_gpu_model.init_gpu(m_layers);
		}

		m_gpu_model.pass_batch_gpu(gX, gY);
	}

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

void mnist_conv::getXyTest(Matf &X, Matf &yp, int batch, bool use_rand, int beg)
{
	if(batch < 0)
		batch = m_mnist->test().size();

	if(use_rand){

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
	}else{
		int cnt_batch = std::min(m_mnist->test().size() - beg, batch);

		X = Matf::zeros(cnt_batch, m_mnist->X().cols);
		yp = Matf::zeros(cnt_batch, m_mnist->y().cols);

		int id = beg;
		for(int i = 0; i < cnt_batch; i++, id++){
			QByteArray& data = m_mnist->test()[id];
			uint lb = m_mnist->lb_test()[id];

			for(int j = 0; j < data.size(); j++){
				X.at(i, j) = ((uint)data[j] > 0? 1. : 0.);
			}
			yp.at(i, lb) = 1.;
		}

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
	std::uniform_int_distribution<int> udtr(-5, 5);
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

	m_gpu_model.init_gpu(m_layers);
}

Matf mnist_conv::forward(const ct::Matf &X, bool use_gpu)
{
	if(m_W.empty() || m_b.empty() || m_layers.empty())
		return Matf(0, 0);

	m_use_gpu = use_gpu;
	if(use_gpu){
		if(!m_gpu_model.isInit())
			m_gpu_model.init_gpu(m_layers);
		return m_gpu_model.forward_gpu(X);
	}

	Matf X_out;

	conv(X, X_out, false);

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

void mnist_conv::conv(const Matf &X, Matf &X_out, bool saved)
{
	if(X.empty())
		return;

	for(size_t i = 0; i < m_cnv.size(); ++i){
		std::vector< convnn::convnn< float > >& ls = m_cnv[i];

		if(i == 0){
			convnn::convnn< float >& m0 = ls[0];
			m0.forward(&X, reLu);
		}else{
			for(size_t j = 0; j < m_cnv[i - 1].size(); ++j){
				size_t off1 = j * m_count_cnvW[i - 1];
				convnn::convnn< float >& m0 = m_cnv[i - 1][j];
				for(int k = 0; k < m_count_cnvW[i - 1]; ++k){
					size_t col = off1 + k;
					convnn::convnn< float >& mi = ls[col];
					mi.forward(&m0.A2[k], reLu);
				}
			}
		}
	}

	convnn::convnn<float>::hconcat(m_cnv.back(), X_out);

	if(!saved){
		for(size_t i = 0; i < m_cnv.size(); ++i){
			for(size_t j = 0; j < m_cnv[i].size(); ++j){
				m_cnv[i][j].clear();
			}
		}
	}
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

	conv(X, m_cnv_a);

	//// MLP

	m_z.resize(m_layers.size());
	m_a.resize(m_layers.size() + 1);

	m_a[0] = m_cnv_a;

	std::vector< Matf > D;
	Matf Wi;
	D.resize(3);

	for(size_t i = 0; i < m_layers.size(); i++){
		if(i < D.size()){
			dropout(m_W[i].rows, m_W[i].cols, 0.9f, D[i]);
			elemwiseMult(m_W[i], D[i], Wi);
			m_z[i] = m_a[i] * Wi;
		}else{
			m_z[i] = m_a[i] * m_W[i];
		}
		m_z[i].biasPlus(m_b[i]);

		if(i < m_layers.size() - 1){
			m_a[i + 1] = relu(m_z[i]);
		}else
			m_a[i + 1] = softmax(m_z[i], 1);
	}

	m_dW.resize(m_layers.size());
	m_dB.resize(m_layers.size());

	float m = X.rows;
	m_d = m_a.back() - y;

	//// Backward

	for(int i = (int)m_layers.size() - 1; i > -1; --i){
//		Matf sz = elemwiseMult(a[i], a[i]);
//		sz = 1. - sz;
		Matf di, sz;

		{
			sz = derivRelu(m_a[i]);

			//Matf di = d * m_W[i].t();
			matmulT2(m_d, m_W[i], di);
			elemwiseMult(di, sz);
		}
		//dW[i] = a[i].t() * d;
		matmulT1(m_a[i], m_d, m_dW[i]);

		m_dW[i] *= 1./m;
		//dW[i] += (m_lambda/m * m_W[i]);

		if(i < (int)D.size()){
			elemwiseMult(m_dW[i], D[i]);
		}

		m_dB[i] = (sumRows(m_d) * (1.f/m)).t();

		m_d = di;
	}

	/// convolution
	{
		nn::hsplit(m_d, m_cnv.back().size() * m_cnv.back()[0].W.size(), m_ds);

		for(int i = m_cnv.size() - 1; i > -1; i--){
			std::vector< convnn::convnn<float > >& lrs = m_cnv[i];

//			qDebug("LR[%d]-----", i);
			size_t kidx = 0;

			for(size_t j = 0; j < lrs.size(); ++j){
				convnn::convnn<float > &cnv = lrs[j];

				size_t kfirst = kidx;
				kidx += (cnv.W.size());

				if(i == m_cnv.size() - 1)
					cnv.backward< Matf (*)(const Matf& mat) >(m_ds, derivRelu, kfirst, kidx, i == 0);
				else
					cnv.backward< Matf (*)(const Matf& mat) >(m_cnv[i + 1], derivRelu, kfirst, kidx, i == 0);

//				qt_work_mat::q_save_mat(cnv.gradW[0], "gradW.txt");
//				qt_work_mat::q_save_mat(cnv.dA1[0], "dA1.txt");
//				qt_work_mat::q_save_mat(cnv.A1[0], "A1.txt");
//				qt_work_mat::q_save_mat(cnv.A0, "A0.txt");
//				for(int k = 0; k < cnv.W.size(); ++k){
//					std::string sw = cnv.W[k];
//					qDebug("W[%d:%d]:\n%s", j, k, sw.c_str());
//				}
			}
//			qDebug("----");
		}
	}

	m_AdamOptimizer.pass(m_dW, m_dB, m_W, m_b);
	m_iteration = m_AdamOptimizer.iteration();
}
