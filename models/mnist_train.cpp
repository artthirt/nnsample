#include "mnist_train.h"

#ifdef _USE_GPU
#include "helper_gpu.h"
#endif

using namespace ct;

#define PRINT_GMAT10(mat) {		\
	std::string s = mat.print(10);			\
	qDebug("%s\n", s.c_str());	\
}

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

void mnist_train::getEstimate(int batch, double &l2, double &accuracy, bool use_gpu)
{
	if(m_X.empty() || m_y.empty())
		return;

	std::vector<int> indexes;
	if(batch < 0){
		batch = m_mnist->train().size();
		indexes.resize(batch);
		for(int i = 0; i < batch; i++){
			indexes[i] = i;
		}
	}else{
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
	}

	Matf X = m_X.getRows(indexes);
	Matf yp = m_y.getRows(indexes);

#ifdef _USE_GPU
	Matf y;

	if(use_gpu){
		y = forward_gpu(X);
	}else{
		y = forward(X);
	}
#else
	Matf y = forward(X);
#endif
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

void mnist_train::getEstimateTest(int batch, double &l2, double &accuracy, bool use_gpu)
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

		for(int i = 0; i < batch; i++){
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

#ifdef _USE_GPU
	Matf y;

	if(use_gpu){
		y = forward_gpu(X);
	}else{
		y = forward(X);
	}
#else
	Matf y = forward(X);
#endif

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
	for(size_t i = 0; i < d.size(); i++){
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
	for(size_t i = 0; i < d.size(); i++){
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

	m_W.resize(layers);
	m_b.resize(layers);

	m_layers.resize(layers);

	for(int i = 0; i < layers; i++){
		Matf mat1, mat2;
		QString capt1, capt2;
		read_mat(f, mat1, capt1);
		read_mat(f, mat2, capt2);
		if(capt1 == "W")
			m_W[i] = mat1;
		else
			m_b[i] = mat1;

		if(capt2 == "b")
			m_b[i] = mat2;
		else
			m_W[i] = mat2;

		m_layers[i] = m_W[i].cols;
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

	f.write(QString("layers %1\n").arg(m_W.size()).toLatin1());
	for(uint i = 0; i < m_W.size(); i++){
		write_mat(f, m_W[i], "W");
		write_mat(f, m_b[i], "b");
	}

	f.close();
}

void mnist_train::pass_batch_autoencoder(int batch)
{
	if(!batch || !m_mnist || !m_mnist->train().size() || m_mnist->train().size() < batch)
		return;

	Matf X;

	Matf a, z;

	qDebug("<<<<<begin>>>>");

	if(enc.empty()){
		enc.resize(m_layers.size() - 1);
		int input = m_X.cols;
		for(uint i = 0; i < enc.size(); i++){
			enc[i].init(m_W[i], m_b[i], input, m_layers[i], &relu, &derivRelu);
			input = m_layers[i];
		}
	}

	for(int i = 0; i < (int)m_layers.size() - 1; i++){

		getX(X, batch);
		a = X;
		for(int j = 0; j < i; j++){
			z = a * m_W[j];
			z.biasPlus(m_b[j]);
			a = relu(z);
		}

		enc[i].pass(a);

		float l2 = enc[i].l2(a);
		qDebug("l2=%f; W.rows=%d; W.cols=%d", l2, enc[i].W[0].rows, enc[i].W[0].cols);
		m_W[i] = enc[i].W[0];
		m_b[i] = enc[i].b[0];
	}
	qDebug("<<<<<end>>>>");
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

	X = m_X.getRows(indexes);
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
		Matf di, sz;
		if(i > 0){
			sz = derivRelu(a[i]);

			//Matf di = d * m_W[i].t();
			matmulT2(d, m_W[i], di);
			di = elemwiseMult(di, sz);
		}
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

		if(i > 0)
			d = di;
	}

	if(!m_AdamOptimizer.pass(dW, dB, m_W, m_b)){
		std::cout << "optimizer not work\n";
	}
}

#ifdef _USE_GPU

double mnist_train::L2_gpu(int batch)
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

	Matf y = forward_gpu(X);

	double m = X.rows;

	Matf d = yp - y;

	Matf l2 = elemwiseMult(d, d);
	l2 = sumRows(l2);
	l2 *= 1./m;
	return l2.sum();

}

double mnist_train::L2test_gpu(int batch)
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

	Matf y = forward_gpu(X);

	double m = X.rows;

	Matf d = yp - y;

	Matf l2 = elemwiseMult(d, d);
	l2 = sumRows(l2);
	l2 *= 1./m;
	return l2.sum();
}

Matf mnist_train::forward_gpu(int index, int count)
{
	if(m_W.empty() || m_b.empty() || m_layers.empty())
		return Matf(0, 0);

	Matf X = m_X.getRows(index, count);

	return forward_gpu(X);
}

Matf mnist_train::forward_gpu(const Matf &X)
{
	if(m_gW.empty() || m_gb.empty() || m_layers.empty())
		return Matf(0, 0);

	Matf a;

	if(g_a.empty()){
		g_z.resize(m_layers.size());
		g_a.resize(m_layers.size() + 1);
	}

	gpumat::convert_to_gpu(X, g_a[0]);

	for(size_t i = 0; i < m_layers.size(); i++){
		gpumat::matmul(g_a[i], m_gW[i], g_z[i]);
		gpumat::biasPlus(g_z[i], m_gb[i]);
		if(i < m_layers.size() - 1){
			gpumat::reLu(g_z[i], g_a[i + 1]);
		}else
			gpumat::softmax(g_z[i], 1, g_a[i + 1], partZ);
	}

	gpumat::convert_to_mat(g_a.back(), a);

	return a;
}

Matf mnist_train::forward_test_gpu(int index, int count)
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

	return forward_gpu(X);
}

void mnist_train::init_gpu(int seed)
{
	if(!m_mnist || !m_layers.size() || !m_mnist->train().size() || !m_mnist->lb_train().size())
		return;

	int input = m_X.cols;
	int output = m_layers[0];

	m_gW.resize(m_layers.size());
	m_gb.resize(m_layers.size());

	for(size_t i = 0; i < m_layers.size(); i++){
		output = m_layers[i];

		double n = 1./sqrt(input);

		Matf Wi = Matf(input, output);
		Matf bi = Matf::ones(output, 1);
		Wi.randn(0., n, seed);
		bi.randn(0, n, seed);

		gpumat::convert_to_gpu(Wi, m_gW[i]);
		gpumat::convert_to_gpu(bi, m_gb[i]);

		input = output;
	}

	if(!m_gpu_adam.init(m_layers, m_X.cols, gpumat::GPU_FLOAT)){
		std::cout << "optimizer not init\n";
	}
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

	X = m_X.getRows(indexes);
	y = m_y.getRows(indexes);

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
	gpumat::convert_to_gpu(X, m_gX);
	gpumat::convert_to_gpu(y, m_gy);

	pass_batch_gpu(m_gX, m_gy);
}

void mnist_train::pass_batch_gpu(const gpumat::GpuMat &X, const gpumat::GpuMat &y)
{
	if(m_W.empty() || m_b.empty() || m_layers.empty() ||
			m_layers.back() != y.cols){
		std::cout << "wrong parameters of model\n";
		return;
	}

	/// forward

//	std::vector< Matf > z, a;
//	z.resize(m_layers.size());
//	a.resize(m_layers.size() + 1);
	if(m_gW.empty()){
		init_gpu(1);
	}
	PRINT_GMAT10(m_gW[0]);

	if(g_a.empty()){
		g_z.resize(m_layers.size());
		g_a.resize(m_layers.size() + 1);
	}

	g_a[0] = X;

//	Matf D1, Dt1, D2, Dt2, D3, Dt3;

	for(size_t i = 0; i < m_layers.size(); i++){
		gpumat::matmul(g_a[i], m_gW[i], g_z[i]);
		gpumat::biasPlus(g_z[i], m_gb[i]);
		//z[i] = a[i] * m_W[i];
		//z[i].biasPlus(m_b[i]);

//		if(i == 0){
//			dropout(z[i], 0.5f, D1, Dt1);
//		}
//		if(i == 1){
//			dropout(z[i], 0.5f, D2, Dt2);
//		}
//		if(i == 2){
//			dropout(z[i], 0.5f, D3, Dt3);
//		}
		if(i < m_layers.size() - 1){
			gpumat::reLu(g_z[i], g_a[i + 1]);
			//a[i + 1] = relu(z[i]);
		}else{
			gpumat::softmax(g_z[i], 1, g_a[i + 1], partZ);
			PRINT_GMAT10(g_z[i]);
			PRINT_GMAT10(g_a[i + 1]);
		}
			//a[i + 1] = softmax(z[i], 1);
	}

//	std::vector< Matf > dW, dB;
	g_dW.resize(m_layers.size());
	g_dB.resize(m_layers.size());

	float m = X.rows;

	gpumat::sub(g_a.back(), y, g_d);
	//Matf d = a.back() - y;
	PRINT_GMAT10(g_a.back());
	PRINT_GMAT10(y);
	PRINT_GMAT10(g_d);

	/// backward

	for(int i = (int)m_layers.size() - 1; i > -1; --i){
//		Matf sz = elemwiseMult(a[i], a[i]);
//		sz = 1. - sz;
//		Matf di, sz;
		if(i > 0){
			gpumat::deriv_reLu(g_a[i], g_sz);
			PRINT_GMAT10(g_sz);
			//sz = derivRelu(a[i]);

			//Matf di = d * m_W[i].t();
			PRINT_GMAT10(m_gW[i]);
			gpumat::matmulT2(g_d, m_gW[i], g_di);
			PRINT_GMAT10(g_di);
			//matmulT2(d, m_W[i], di);
			gpumat::elemiseMul(g_di, g_sz, g_di);
			PRINT_GMAT10(g_di);
			//di = elemwiseMult(di, sz);
		}
		//dW[i] = a[i].t() * d;
		gpumat::matmulT1(g_a[i], g_d, g_dW[i]);
		PRINT_GMAT10(g_dW[i]);
		//matmulT1(a[i], d, dW[i]);
		gpumat::mulval(g_dW[i], 1./m);
		PRINT_GMAT10(g_dW[i]);
		//dW[i] *= 1./m;
		//gpumat::add(g_dW[i], m_gW[i], m_lambda/m, 1.);
		//PRINT_GMAT10(g_dW[i]);
		//dW[i] += (m_lambda/m * m_W[i]);

//		if(i == 2){
//			dropout_transpose(dW[i], Dt3);
//		}
//		if(i == 1){
//			dropout_transpose(dW[i], Dt2);
//		}
//		if(i == 0){
//			dropout_transpose(dW[i], Dt1);
//		}

		gpumat::sumRows(g_d, g_dB[i], (1.f/m));
		PRINT_GMAT10(g_dB[i]);

		g_dB[i].swap_dims();
		//dB[i] = (sumRows(d) * (1.f/m)).t();

		if(i > 0)
			g_d = g_di;
	}

	m_gpu_adam.pass(g_dW, g_dB, m_gW, m_gb);

//	if(!m_AdamOptimizer.pass(dW, dB, m_W, m_b)){
//		std::cout << "optimizer not work\n";
//	}
}

#endif
