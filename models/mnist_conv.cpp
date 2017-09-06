#include "mnist_conv.h"

#include "matops.h"
#include "mnist_utils.h"

#include "qt_work_mat.h"

const int cnv_size = 3;

using namespace ct;

const int imageWidth = 28;
const int imageHeight = 28;

mnist_conv::mnist_conv()
{
	m_iteration = 0;
	m_mnist = 0;
//	m_count_cnvW.push_back(1);
	m_seed = 1;

	setConvLength();
}

void mnist_conv::setMnist(mnist_reader *mnist)
{
	m_mnist = mnist;
}

void mnist_conv::setConvLength()
{

	time_t tm;
	time(&tm);
	ct::generator.seed(tm);

	m_cnv.resize(cnv_size);
	int prev = 1;
	ct::Size szA0(imageWidth, imageHeight);

	m_cnv[0].init(szA0, 1, 1, 32, ct::Size(3, 3), ct::LEAKYRELU, true, true, false);
	m_cnv[1].init(m_cnv[0].szOut(), 32, 1, 64, ct::Size(3, 3), ct::LEAKYRELU, true, true, true);
	m_cnv[2].init(m_cnv[1].szOut(), 64, 1, 96, ct::Size(3, 3), ct::LEAKYRELU, false, true, true);

#ifdef _USE_GPU
	m_gpu_model.setConvLength();
#endif
}

std::vector<conv2::convnn<float> > &mnist_conv::cnv()
{
	return m_cnv;
}

std::vector< Matf > mnist_conv::cnvW()
{
	std::vector< Matf > res;

	if(m_use_gpu){
		res.resize(m_gpu_model.cnv().size());

		for(size_t i = 0; i < m_gpu_model.cnv().size(); ++i){
			ct::Matf Wf;
			gpumat::convert_to_mat(m_gpu_model.cnv()[i].W, Wf);
			res.push_back(Wf);
		}
	}else{
		res.resize(m_cnv.size());

		for(size_t i = 0; i < m_cnv.size(); ++i){
			res.push_back(m_cnv[i].W);
		}
	}
	return res;
}

void mnist_conv::save_model(bool gpu)
{
	if(gpu){
		m_gpu_model.save_model();
		return;
	}

	std::fstream fs;
	fs.open(name_model_cnv, std::ios_base::out | std::ios_base::binary);

	if(!fs.is_open()){
		printf("File %s not open\n", name_model_cnv.c_str());
		return;
	}

//	write_vector(fs, m_cnvlayers);
//	write_vector(fs, m_layers);

//	fs.write((char*)&m_szA0, sizeof(m_szA0));

	int cnvs = m_cnv.size(), mlps = m_mlp.size();

	/// size of convolution array
	fs.write((char*)&cnvs, sizeof(cnvs));
	/// size of mlp array
	fs.write((char*)&mlps, sizeof(mlps));

	for(size_t i = 0; i < m_cnv.size(); ++i){
		conv2::convnnf &cnv = m_cnv[i];
		cnv.write2(fs);
	}

	for(size_t i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].write2(fs);
	}

	printf("model saved.\n");
}

void mnist_conv::load_model(bool gpu)
{
	if(gpu){
		m_gpu_model.load_model();
		return;
	}

	std::fstream fs;
	fs.open(name_model_cnv, std::ios_base::in | std::ios_base::binary);

	if(!fs.is_open()){
		printf("File %s not open\n", name_model_cnv.c_str());
		return;
	}

//	read_vector(fs, m_cnvlayers);
//	read_vector(fs, m_layers);

//	fs.read((char*)&m_szA0, sizeof(m_szA0));

//	setConvLayers(m_cnvlayers, m_szA0);

	int cnvs, mlps;

	/// size of convolution array
	fs.read((char*)&cnvs, sizeof(cnvs));
	/// size of mlp array
	fs.read((char*)&mlps, sizeof(mlps));

	printf("Load model: conv size %d, mlp size %d", cnvs, mlps);

	m_cnv.resize(cnvs);
	m_mlp.resize(mlps);

	printf("conv\n");
	for(size_t i = 0; i < m_cnv.size(); ++i){
		conv2::convnnf &cnv = m_cnv[i];
		cnv.read2(fs);
		printf("layer %d: rows %d, cols %d\n", i, cnv.W.rows, cnv.W.cols);
	}

	printf("mlp\n");
	for(size_t i = 0; i < m_mlp.size(); ++i){
		ct::mlpf &mlp = m_mlp[i];
		mlp.read2(fs);
		printf("layer %d: rows %d, cols %d\n", i, mlp.W.rows, mlp.W.cols);
	}

	printf("model loaded.\n");
}

Matf mnist_conv::forward(int index, int count, bool use_gpu, bool train)
{
	if(m_layers.empty())
		return Matf(0, 0);

	Matf X = m_mnist->X().getRows(index, count);
	std::vector< ct::Matf > vX;
	vX.resize(X.rows);
	for(int i = 0; i < X.rows; ++i){
		vX[i] = X.row(i);
	}

	return forward(vX, false, use_gpu);
}

Matf mnist_conv::forward_test(int index, int count, bool use_gpu)
{
	if(!m_mnist || m_mnist->test().empty() || m_mnist->lb_test().empty())
		return Matf(0, 0);

	std::vector< Matf > X;

	count = std::min(count, m_mnist->test().size() - index);

	X.resize(count);
	for(int i = 0; i < count; i++){
		int id = index + i;
		QByteArray& data = m_mnist->test()[id];
		//uint lb = m_mnist->lb_test()[id];
		X[i].setSize(1, m_mnist->X().cols);

		for(int j = 0; j < data.size(); j++){
			X[i].ptr()[j] = ((uint)data[j] > 0? 1. : 0.);
		}
	}

	return forward(X, false, use_gpu, false);
}

void mnist_conv::setAlpha(double alpha)
{
	m_optim.setAlpha(alpha);
	m_cnv_optim.setAlpha(alpha);

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

	//std::vector< Matf > X;
	Matf yp, y;

	const int batchi = 50;
	int index = 0;
	int xcnt = 0;

	while(index < batch){
		std::vector< Matf > Xi;
		ct::Matf ypi;
		getXy(Xi, ypi, batchi);
		yp.push_back(ypi);
		xcnt += Xi.size();

		Matf yi = forward(Xi, false, use_gpu, false);
		y.push_back(yi);
		index += batchi;
	}

	double m = xcnt;

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

	size_t test_size = m_mnist->test().size();
	size_t batch_count = test_size/50;

	size_t batch_size = test_size / batch_count;

	l2 = 0;
	int right = 0;

	for(size_t i = 0; i < batch_count; ++i){

		std::vector< Matf > X;
		ct::Matf yp;
		getXyTest(X, yp, batch_size, false, i * batch_size);

		Matf y = forward(X, false, use_gpu);

		Matf d = yp - y;

		int m = X.size();

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

	std::vector< Matf > X;
	Matf y;

	getXy(X, y, batch);

	std::uniform_int_distribution<int> udtr(-5, 5);
	std::uniform_real_distribution<float> uar(-7, 7);

#pragma omp parallel for
	for(int i = 0; i < X.size(); i++){
		float *Xi = X[i].ptr();
		int x = udtr(m_generator);
		int y = udtr(m_generator);
		float ang = uar(m_generator);
		ang = angle2rad(ang);

		rotate_mnist<float>(28, 28, ang, Xi);
		translate<float>(x, y, 28, 28, Xi);
	}

#ifndef _USE_GPU
	use_gpu = false;
#endif

	m_use_gpu = use_gpu;
	if(!use_gpu){
		pass_batch(X, y);
	}else{
#ifdef _USE_GPU
		gpumat::cnv2gpu(X, gX);
		gpumat::convert_to_gpu(y, gY);

		if(!m_gpu_model.isInit()){
			m_gpu_model.init_gpu(m_layers);
		}

		m_gpu_model.pass_batch_gpu(gX, gY);
#endif
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

void mnist_conv::getXyTest(std::vector< Matf > &X, Matf &yp, int batch, bool use_rand, int beg)
{
	if(batch < 0)
		batch = m_mnist->test().size();

	if(use_rand){

		std::vector<int> indexes;

		getBatchIds(indexes, batch);

		//X = Matf::zeros(batch, m_mnist->X().cols);
		yp = Matf::zeros(batch, m_mnist->y().cols);

		X.resize(batch);

		for(int i = 0; i < batch; i++){
			X[i].setSize(1, m_mnist->X().cols);
			int id = indexes[i];
			QByteArray& data = m_mnist->test()[id];
			uint lb = m_mnist->lb_test()[id];

			for(int j = 0; j < data.size(); j++){
				X[i].ptr()[j] = ((uint)data[j] > 0? 1. : 0.);
			}
			yp.at(i, lb) = 1.;
		}
	}else{
		int cnt_batch = std::min(m_mnist->test().size() - beg, batch);

		X.resize(cnt_batch);
		//Matf::zeros(cnt_batch, m_mnist->X().cols);
		yp = Matf::zeros(cnt_batch, m_mnist->y().cols);

		int id = beg;
		for(int i = 0; i < cnt_batch; i++, id++){
			X[i].setSize(1, m_mnist->X().cols);
			QByteArray& data = m_mnist->test()[id];
			uint lb = m_mnist->lb_test()[id];

			for(int j = 0; j < data.size(); j++){
				X[i].ptr()[j] = ((uint)data[j] > 0? 1. : 0.);
			}
			yp.at(i, lb) = 1.;
		}

	}
}

void mnist_conv::getXy(std::vector< Matf > &X, Matf &y, int batch)
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

	ct::Matf mX = m_mnist->X().getRows(indexes);
	y = m_mnist->y().getRows(indexes);

	X.resize(mX.rows);
	for(int i = 0; i < X.size(); ++i){
		X[i] = mX.row(i);
	}
}

void mnist_conv::randX(Matf &X)
{
#if 1
	std::uniform_int_distribution<int> udtr(-10, 10);
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

void mnist_conv::setDropout(size_t count, float prob)
{
	for(size_t i = 0; i < std::min(count, m_mlp.size() - 1); ++i){
		m_mlp[i].setDropout(true);
		m_mlp[i].setDropout(prob);
	}
}

void mnist_conv::clearDropout()
{
	for(size_t i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].setDropout(false);
	}
}

void mnist_conv::init(int seed)
{
	if(!m_mnist || !m_layers.size() || !m_mnist->train().size() || !m_mnist->lb_train().size() || m_cnv.empty())
		return;

	m_cnv_out_size = m_cnv.back().szOut();
	m_cnv_out_len = m_cnv.back().outputFeatures();

	qDebug("--- input to MLP: %d ----", m_cnv_out_len);

	int input = m_cnv_out_len;
	int output = m_layers[0];

	m_mlp.resize(m_layers.size());

	for(size_t i = 0; i < m_layers.size(); i++){
		output = m_layers[i];

		mlp<float>& _mlp = m_mlp[i];

		_mlp.init(input, output, i == m_layers.size() - 1? ct::SOFTMAX : ct::LEAKYRELU);

		input = output;
	}

	if(!m_optim.init(m_mlp)){
		std::cout << "optimizer not init\n";
	}
	m_cnv_optim.init(m_cnv);

	m_gpu_model.init_gpu(m_layers);
}

Matf mnist_conv::forward(const std::vector< ct::Matf > &X, bool use_dropout, bool use_gpu, bool train)
{
	if(m_mlp.empty() || m_layers.empty() || X.empty())
		return Matf(0, 0);

	m_use_gpu = use_gpu;
	if(use_gpu){
		if(!m_gpu_model.isInit())
			m_gpu_model.init_gpu(m_layers);
		gpumat::cnv2gpu(X, gX);
		return m_gpu_model.forward_gpu(gX);
	}

	for(conv2::convnnf& item: m_cnv){
		item.setTrainMode(train);
	}

	conv(X, m_Xout);

	Matf *pA = &m_Xout;

	if(!use_dropout)
		clearDropout();

	for(size_t i = 0; i < m_layers.size(); i++){
		mlp<float>& _mlp = m_mlp[i];

		_mlp.forward(pA);
		pA = &_mlp.A1;
	}
	return m_mlp.back().A1;
}

void mnist_conv::conv(const std::vector< Matf > &X, Matf &X_out)
{
	if(X.empty())
		return;

	std::vector< Matf > *pX = (std::vector< Matf > *)&X;
	for(size_t i = 0; i < m_cnv.size(); ++i){
		m_cnv[i].forward(pX);
		pX = &m_cnv[i].XOut();
	}

	conv2::vec2mat(m_cnv.back().XOut(), X_out);

	//ct::convnn<float>::hconcat(m_cnv.back(), X_out);
}

void mnist_conv::pass_batch(const std::vector< Matf > &X, const Matf &y)
{
	if(m_mlp.empty() || m_layers.empty() ||
			m_layers.back() != y.cols){
		std::cout << "wrong parameters of model\n";
		return;
	}

	/// forward
	setDropout(2, 0.95f);

	forward(X, true, false);

	Matf d = m_mlp.back().A1 - y;

	/// backward

	Matf* pD = &d;

	for(int i = (int)m_layers.size() - 1; i > -1; --i){
		mlp<float>& _mlp = m_mlp[i];

		_mlp.backward(*pD);
		pD = &_mlp.DltA0;
	}

	{
		//ct::hsplit(m_mlp.front().DltA0, m_cnv.back().size() * m_cnv.back()[0].W.size(), m_ds);
		conv2::mat2vec(m_mlp.front().DltA0, m_cnv.back().szK, m_ds);

		std::vector< ct::Matf > *pD = &m_ds;

		for(int i = m_cnv.size() - 1; i > -1; --i){
			m_cnv[i].backward(*pD, i == 0);
			pD = &m_cnv[i].Dlt;
		}
	}

	m_optim.pass(m_mlp);
	m_cnv_optim.pass(m_cnv);
	m_iteration = m_optim.iteration();
}
