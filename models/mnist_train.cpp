#include "mnist_train.h"

#ifdef _USE_GPU
#include "helper_gpu.h"
#endif

#include "mnist_utils.h"

using namespace ct;

mnist_train::mnist_train()
{
	m_mnist = 0;
	m_lambda = 0.00001f;
	m_iteration = 0;
	m_optim.setAlpha(0.01f);
	m_optim.setBetha2(0.999f);
#ifdef _USE_GPU
	m_dropout_count = 5;
#endif
}

mnist_train::~mnist_train()
{

}

void mnist_train::setMnist(mnist_reader *mnist)
{
	m_mnist = mnist;
}

Matf mnist_train::forward(int index, int count, bool use_gpu)
{
	if(m_mlp.empty() || m_layers.empty())
		return Matf(0, 0);

	Matf X = m_mnist->X().getRows(index, count);

	return forward(X, false, use_gpu);
}

Matf mnist_train::forward_test(int index, int count, bool use_gpu)
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

	return forward(X, false, use_gpu);
}

void mnist_train::setAlpha(double alpha)
{
	m_optim.setAlpha(alpha);
#ifdef _USE_GPU
	m_gpu_adam.setAlpha(alpha);
#endif
}

void mnist_train::setLayers(const std::vector<int> &layers)
{
	m_layers = layers;
}

uint mnist_train::iteration() const
{
	return m_iteration;
}

void mnist_train::getEstimate(int batch, double &l2, double &accuracy, bool use_gpu)
{
	if(m_mnist->X().empty() || m_mnist->y().empty())
		return;

	Matf X;
	Matf yp;

	getXy(X, yp, batch);

	Matf y = forward(X, false, use_gpu);

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

void mnist_train::getEstimateTest(double &l2, double &accuracy, bool use_gpu)
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

		Matf y = forward(X, false, use_gpu);

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

void mnist_train::init(int seed)
{
	if(!m_mnist || !m_layers.size() || !m_mnist->train().size() || !m_mnist->lb_train().size())
		return;

	ct::generator.seed(seed);

	int input = m_mnist->X().cols;
	int output = m_layers[0];

	m_mlp.resize(m_layers.size());

	for(size_t i = 0; i < m_layers.size(); i++){
		output = m_layers[i];

		mlp<float>& _mlp = m_mlp[i];

		_mlp.init(input, output);

		input = output;
	}

	if(!m_optim.init(m_mlp)){
		std::cout << "optimizer not init\n";
	}
}

void mnist_train::pass_batch(int batch)
{
	if(!batch || !m_mnist || !m_mnist->train().size() || m_mnist->train().size() < batch)
		return;

	Matf X, y;

	getXy(X, y, batch);

#if 1
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
#endif
	pass_batch(X, y);
}

template< typename T >
void read_mat(QFile&f, Mat_<T>& mat, QString& capt)
{
	QString str = f.readLine();
	str = str.trimmed();
	QStringList sl = str.split(' ');
	if(sl.size() == 3){
		capt = sl[0];
		int rows = sl[1].toInt();
		int cols = sl[2].toInt();

		mat.setSize(rows, cols);

		for(int i = 0; i < rows; i++){
			QString str = f.readLine();
			str = str.trimmed();
			QStringList sl = str.split(' ');
			if(sl.size() < cols)
				return;
			for(int j = 0; j < cols; j++){
				mat.at(i, j) = (T)sl[j].toDouble();
			}
		}
	}
}

void mnist_train::load(const QString &fn)
{
	if(!QFile::exists(fn))
		return;

	QFile f(fn);
	if(!f.open(QIODevice::ReadOnly))
		return;

	QString str = f.readLine();
	str = str.trimmed();
	QStringList sl = str.split(' ');
	if(sl.size() != 2)
		return;

	if(sl[0] != "layers")
		return;

	int layers = sl[1].toInt();

	m_mlp.resize(layers);

	m_layers.resize(layers);

	for(int i = 0; i < layers; i++){
		ct::mlp<float>& _mlp = m_mlp[i];

		Matf mat1, mat2;
		QString capt1, capt2;
		read_mat(f, mat1, capt1);
		read_mat(f, mat2, capt2);
		if(capt1 == "W")
			_mlp.W = mat1;
		else
			_mlp.B = mat1;

		if(capt2 == "b")
			_mlp.B = mat2;
		else
			_mlp.W = mat2;

//		float w1 = m_W[i].max(), w2 = m_W[i].min();
//		float b1 = m_b[i].max(), b2 = m_b[i].min();
//		qDebug("max(w)=%f, max(b)=%f, min(w)=%f, min(b)=%f", w1, b1, w2, b2);

		m_layers[i] = _mlp.W.cols;
	}
	f.close();
}

template< typename T >
void write_mat(QFile &f, const Mat_<T>& mat, const QString& capt)
{
	QString str = capt + QString(" %1 %2").arg(mat.rows).arg(mat.cols) + "\n";
	f.write(str.toLatin1());
	for(int i = 0; i < mat.rows; i++){
		QString str;
		for(int j = 0; j < mat.cols; j++){
			str += QString::number(mat.at(i, j)) + " ";
		}
		str += "\n";
		f.write(str.toLatin1());
	}
}

void mnist_train::save(const QString &fn)
{
	QFile f(fn);

	if(!f.open(QIODevice::WriteOnly))
		return;

	f.write(QString("layers %1\n").arg(m_mlp.size()).toLatin1());
	for(uint i = 0; i < m_mlp.size(); i++){
		write_mat(f, m_mlp[i].W, "W");
		write_mat(f, m_mlp[i].B, "b");
	}

	f.close();
}

double mnist_train::pass_batch_autoencoder(int batch, bool use_gpu)
{
	if(!batch || !m_mnist || !m_mnist->train().size() || m_mnist->train().size() < batch)
		return -1;

	Matf X;

	Matf a, z;

	qDebug("<<<<<begin>>>>");

#ifndef _USE_GPU
	use_gpu = false;
#else
	gpumat::GpuMat gX;
#endif

	if(use_gpu){
#ifdef _USE_GPU
//		if(enc_gpu.empty()){
//			enc_gpu.resize(m_layers.size() - 1);
//			int input = m_mnist->X().cols;
//			for(uint i = 0; i < enc_gpu.size(); i++){
//				gpumat::SimpleAutoencoder& enc = enc_gpu[i];
//				enc.init(m_gW[i], m_gb[i], input, m_layers[i], &gpumat::reLu, &gpumat::deriv_reLu);
//				input = m_layers[i];
//			}
//		}
#endif
	}else{
		if(enc.empty()){
			enc.resize(m_layers.size() - 1);
			int input = m_mnist->X().cols;
			for(uint i = 0; i < enc.size(); i++){
				enc[i].init(m_mlp[i].W, m_mlp[i].B, input, m_layers[i], &relu, &derivRelu);
				input = m_layers[i];
			}
		}
	}

	double res = 0;

	for(int i = 0; i < (int)m_layers.size() - 1; i++){

		if(use_gpu){
#ifdef _USE_GPU
//			getX(X, batch);
//			randX(X);

//			gpumat::convert_to_gpu(X, gX);
//			g_a[0] = gX;
//			for(int j = 0; j < i; j++){
//				gpumat::matmul(g_a[j], m_gW[j], g_z[j]);
//				gpumat::biasPlus(g_z[j], m_gb[j]);
//				gpumat::reLu(g_z[j], g_a[j + 1]);
////				z = a[j] * m_W[j];
////				z.biasPlus(m_b[j]);
////				a = relu(z);
//			}

//			enc_gpu[i].pass(g_a[i]);

//			float l2 = enc_gpu[i].l2(g_a[i]);
//			res += l2;
//			qDebug("l[%d]: l2=%f; W.rows=%d; W.cols=%d", i, l2, enc_gpu[i].W[0].rows, enc_gpu[i].W[0].cols);
//			m_gW[i] = enc_gpu[i].W[0];
//			m_gb[i] = enc_gpu[i].b[0];
#endif
		}else{
			getX(X, batch);
			randX(X);
			a = X;

			Matf* pA = &a;
			for(int j = 0; j < i; ++j){
				m_mlp[j].forward(pA, ct::RELU, false);
				pA = &m_mlp[j].A1;
			}

			enc[i].pass(*pA);

			float l2 = enc[i].l2(*pA);
			res += l2;
			qDebug("l2=%f; W.rows=%d; W.cols=%d", l2, enc[i].W[0].rows, enc[i].W[0].cols);
			m_mlp[i].W = enc[i].W[0];
			m_mlp[i].B = enc[i].b[0];
		}
	}
	qDebug("<<<<<end>>>>");
	return res;
}

void mnist_train::copyWbMat2GpuMat()
{
#ifdef _USE_GPU
	for(size_t i = 0; i < m_mlp.size(); i++){
		gpumat::convert_to_gpu(m_mlp[i].W, m_gpu_mlp[i].W);
		gpumat::convert_to_gpu(m_mlp[i].B, m_gpu_mlp[i].B);
	}
#endif
}

void mnist_train::copyWbGpuMat2Mat()
{
#ifdef _USE_GPU
	for(size_t i = 0; i < m_mlp.size(); i++){
		gpumat::convert_to_mat(m_gpu_mlp[i].W, m_mlp[i].W);
		gpumat::convert_to_mat(m_gpu_mlp[i].B, m_mlp[i].B);
	}
#endif
}

void mnist_train::init_weights(int seed)
{
	init(seed);
#ifdef _USE_GPU
	init_gpu();
#endif
}

void mnist_train::getX(Matf &X, int batch)
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

void mnist_train::getXyTest(Matf &X, Matf &yp, int batch, bool use_rand, int beg)
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

void mnist_train::getXy(Matf &X, Matf &y, int batch)
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

void mnist_train::randX(Matf &X)
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

void mnist_train::setDropout(size_t count, float prob)
{
	for(size_t i = 0; i < std::min(count, m_mlp.size() - 1); ++i){
		m_mlp[i].setDropout(true, prob);
	}
}

void mnist_train::clearDropout()
{
	for(size_t i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].setDropout(false);
	}
}

void mnist_train::getBatchIds(std::vector<int> &indexes, int batch)
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

///*************************

Matf mnist_train::forward(const ct::Matf &X, bool use_dropout, bool use_gpu)
{
	if(m_mlp.empty() || m_layers.empty())
		return Matf(0, 0);

#ifdef _USE_GPU
	if(use_gpu)
		return forward_gpu(X);
#endif

	/////////////////

	m_X = X;

//	qDebug("---CPU---");
	ct::etypefunction func = ct::RELU;

	if(!use_dropout)
		clearDropout();

	Matf *pA = &m_X;

	for(size_t i = 0; i < m_layers.size(); i++){
		mlp<float>& _mlp = m_mlp[i];

		if(i == m_layers.size() - 1)
			func = ct::SOFTMAX;

		_mlp.forward(pA, func);
		pA = &_mlp.A1;
	}
	return m_mlp.back().A1;
}

void mnist_train::pass_batch(const Matf &X, const Matf &y)
{
	if(m_mlp.empty() || m_layers.empty() ||
			m_layers.back() != y.cols){
		std::cout << "wrong parameters of model\n";
		return;
	}

	/// forward
	setDropout(2, 0.95f);

	forward(X, true);

	Matf d = m_mlp.back().A1 - y;

	/// backward

	Matf* pD = &d;

	for(int i = (int)m_layers.size() - 1; i > -1; --i){
		mlp<float>& _mlp = m_mlp[i];

		_mlp.backward(*pD, i == 0);
		pD = &_mlp.DltA0;
	}

	if(!m_optim.pass(m_mlp)){
		std::cout << "optimizer not work\n";
	}
	m_iteration = m_optim.iteration();
}

#ifdef _USE_GPU

Matf mnist_train::forward_gpu(const Matf &X, bool use_dropout, bool converToMatf)
{
	if(m_layers.empty() || X.empty())
		return Matf(0, 0);

	gpumat::convert_to_gpu(X, m_gX);

	gpumat::etypefunction func = gpumat::RELU;

	if(!use_dropout)
		clearGpuDropout();

	gpumat::GpuMat *pA = &m_gX;

	for(size_t i = 0; i < m_layers.size(); i++){
		gpumat::mlp& _mlp = m_gpu_mlp[i];

		if(i == m_layers.size() - 1)
			func = gpumat::SOFTMAX;

		_mlp.forward(pA, func);
		pA = &_mlp.A1;
	}

	if(converToMatf){
		Matf a;
		gpumat::convert_to_mat(m_gpu_mlp.back().A1, a);
		return a;
	}
	return Matf(0, 0);

}

Matf mnist_train::forward_gpu(const gpumat::GpuMat &X, bool use_dropout, bool converToMatf)
{
	if(m_layers.empty() || X.empty())
		return Matf(0, 0);

	gpumat::etypefunction func = gpumat::RELU;

	if(!use_dropout)
		clearGpuDropout();

	gpumat::GpuMat *pA = (gpumat::GpuMat*)&X;

	for(size_t i = 0; i < m_layers.size(); i++){
		gpumat::mlp& _mlp = m_gpu_mlp[i];

		if(i == m_layers.size() - 1)
			func = gpumat::SOFTMAX;

		_mlp.forward(pA, func);
		pA = &_mlp.A1;
	}

	if(converToMatf){
		Matf a;
		gpumat::convert_to_mat(m_gpu_mlp.back().A1, a);
		return a;
	}
	return Matf(0, 0);

}

void mnist_train::init_gpu()
{
	if(!m_mnist || !m_layers.size() || !m_mnist->train().size() || !m_mnist->lb_train().size())
		return;

	int input = m_mnist->X().cols;
	int output = m_layers[0];

	m_gpu_mlp.resize(m_layers.size());

	for(size_t i = 0; i < m_layers.size(); i++){
		output = m_layers[i];

		gpumat::mlp& _mlp = m_gpu_mlp[i];

		_mlp.init(input, output, gpumat::GPU_FLOAT);

		input = output;
	}

	m_gpu_adam.init(m_gpu_mlp);
}

void mnist_train::pass_batch_gpu(int batch)
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

	X = m_mnist->X().getRows(indexes);
	y = m_mnist->y().getRows(indexes);

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
	gpumat::convert_to_gpu(X, m_gX);
	gpumat::convert_to_gpu(y, m_gy);

	pass_batch_gpu(m_gX, m_gy);
}

void mnist_train::pass_batch_gpu(const gpumat::GpuMat &X, const gpumat::GpuMat &y)
{
	if(m_gpu_mlp.empty() || m_layers.empty() ||
			m_layers.back() != y.cols){
		std::cout << "wrong parameters of model\n";
		return;
	}

	/// forward
	setGpuDropout(2, 0.95f);

	forward_gpu(X, true, false);

	gpumat::sub(m_gpu_mlp.back().A1, y, g_d);

	/// backward

	gpumat::GpuMat* pD = &g_d;

	for(int i = (int)m_layers.size() - 1; i > -1; --i){
		gpumat::mlp& _mlp = m_gpu_mlp[i];

		_mlp.backward(*pD, i == 0);
		pD = &_mlp.DltA0;
	}

	m_gpu_adam.pass(m_gpu_mlp);

	m_iteration = m_gpu_adam.iteration();

}

void mnist_train::save_gpu_matricies()
{
	for(size_t i = 0; i < m_gpu_mlp.size(); ++i){
		gpumat::save_gmat(m_gpu_mlp[i].W, QString("./weigths/g_W%1.txt").arg(i).toStdString());
		gpumat::save_gmat(m_gpu_mlp[i].B, QString("./weigths/g_b%1.txt").arg(i).toStdString());
	}
}

void mnist_train::setGpuDropout(size_t count, float prob)
{
	for(size_t i = 0; i < std::min(count, m_gpu_mlp.size() - 1); ++i){
		m_gpu_mlp[i].setDropout(true, prob);
	}
}

void mnist_train::clearGpuDropout()
{
	for(size_t i = 0; i < m_gpu_mlp.size(); ++i){
		m_gpu_mlp[i].setDropout(false);
	}
}

#endif
