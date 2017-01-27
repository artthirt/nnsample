#include "helper_gpu.h"

#include <QDebug>

namespace gpumat{

void convert_to_gpu(const ct::Matf& mat, gpumat::GpuMat& gmat)
{
	if(mat.empty())
		return;
	gmat.resize(mat.rows, mat.cols, GPU_FLOAT);
	gmat.setData(mat.ptr());
}

void convert_to_gpu(const ct::Matd& mat, gpumat::GpuMat& gmat)
{
	if(mat.empty())
		return;
	gmat.resize(mat.rows, mat.cols, GPU_DOUBLE);
	gmat.setData(mat.ptr());
}

void convert_to_mat(const GpuMat &gmat, ct::Matf &mat)
{
	if(gmat.empty() || gmat.type != GPU_FLOAT)
		return;
	mat.setSize(gmat.rows, gmat.cols);
	gmat.getData((void*)mat.ptr());
}

void convert_to_mat(const GpuMat &gmat, ct::Matd &mat)
{
	if(gmat.empty() || gmat.type != GPU_DOUBLE)
		return;
	mat.setSize(gmat.rows, gmat.cols);
	gmat.getData((void*)mat.ptr());
}

///////////////////////////////

AdamOptimizer::AdamOptimizer()
{
	m_alpha = 0.001;
	m_betha1 = 0.9;
	m_betha2 = 0.999;
	m_iteration = 0;
}

double AdamOptimizer::alpha() const{
	return m_alpha;
}

void AdamOptimizer::setAlpha(double v){
	m_alpha = v;
}

double AdamOptimizer::betha1() const{
	return m_betha1;
}

void AdamOptimizer::setBetha1(double v){
	m_betha1 = v;
}

double AdamOptimizer::betha2() const{
	return m_betha2;
}

void AdamOptimizer::setBetha2(double v){
	m_betha2 = v;
}

uint32_t AdamOptimizer::iteration() const{
	return m_iteration;
}

bool AdamOptimizer::empty() const
{
	return m_mW.empty() || m_mb.empty();
}

bool AdamOptimizer::init(const std::vector<int> &layers, int samples, int type)
{
	if(!samples || layers.empty())
		return false;

	using namespace ct;

	m_iteration = 0;

	int input = samples;
	int output = layers[0];

	m_mW.resize(layers.size());
	m_mb.resize(layers.size());

	m_vW.resize(layers.size());
	m_vb.resize(layers.size());

	sW.resize(layers.size());
	sB.resize(layers.size());

	for(size_t i = 0; i < layers.size(); i++){
		output = layers[i];

		m_mW[i].resize(input, output, type);
		m_vW[i].resize(input, output, type);

		m_mb[i].resize(output, 1, type);
		m_vb[i].resize(output, 1, type);

		m_mW[i].zeros();
		m_vW[i].zeros();
		m_mb[i].zeros();
		m_vb[i].zeros();

		input = output;
	}
	return true;
}

bool AdamOptimizer::pass(const std::vector<GpuMat> &gradW, const std::vector<GpuMat> &gradB,
						 std::vector<GpuMat> &W, std::vector<GpuMat> &b)
{
	if(!gradW.size() || gradW.size() != gradB.size() || gradW.size() != W.size())
		return false;

	using namespace ct;

	m_iteration++;
	double sb1 = (1. / (1. - pow(m_betha1, m_iteration)));
	double sb2 = (1. / (1. - pow(m_betha2, m_iteration)));

	for(size_t i = 0; i < gradW.size(); ++i){

		gpumat::add(m_mW[i], gradW[i], m_betha1, (1. - m_betha1));
		gpumat::add(m_mb[i], gradB[i], m_betha1, (1. - m_betha1));
		//m_mW[i] = m_betha1 * m_mW[i] + (T)(1. - m_betha1) * gradW[i];
		//m_mb[i] = m_betha1 * m_mb[i] + (T)(1. - m_betha1) * gradB[i];

		gpumat::elemwiseSqr(gradW[i], sW[i]);
		gpumat::elemwiseSqr(gradB[i], sB[i]);

		gpumat::add(m_vW[i], sW[i], m_betha2, (1. - m_betha2));
		gpumat::add(m_vb[i], sB[i], m_betha2, (1. - m_betha2));
		//m_vW[i] = m_betha2 * m_vW[i] + (T)(1. - m_betha2) * elemwiseSqr(gradW[i]);
		//m_vb[i] = m_betha2 * m_vb[i] + (T)(1. - m_betha2) * elemwiseSqr(gradB[i]);

//		Mat_<T> mWs = m_mW[i] * sb1;
//		Mat_<T> mBs = m_mb[i] * sb1;
//		Mat_<T> vWs = m_vW[i] * sb2;
//		Mat_<T> vBs = m_vb[i] * sb2;

//		vWs.sqrt(); vBs.sqrt();
//		vWs += eps; vBs += eps;
//		mWs = elemwiseDiv(mWs, vWs);
//		mBs = elemwiseDiv(mBs, vBs);

		/// W = -alpha * (sb1 * mW / (sqrt(sb2 * vW) + eps))

		gpumat::sub_adamGrad(W[i], m_mW[i], m_vW[i], m_alpha, sb1, sb2);
		gpumat::sub_adamGrad(b[i], m_mb[i], m_vb[i], m_alpha, sb1, sb2);
		//W[i] -= m_alpha * mWs;
		//b[i] -= m_alpha * mBs;
	}
	return true;
}

///////////////////////////////////////////
///////////////////////////////////////////

SimpleAutoencoder::SimpleAutoencoder(){
	func = 0;
	deriv = 0;
	m_neurons = 0;
}

void SimpleAutoencoder::init(GpuMat &_W, GpuMat &_b, int samples, int neurons, SimpleAutoencoder::tfunc fn, SimpleAutoencoder::tfunc dfn)
{
	func = fn;
	deriv = dfn;
	m_neurons = neurons;

	std::vector< int > layers;
	layers.push_back(neurons);
	layers.push_back(samples);

	W.resize(2);
	b.resize(2);
	dW.resize(2);
	db.resize(2);

	adam.init(layers, samples, _W.type);

	W[0] = _W;
	b[0] = _b;

	transpose(_W, W[1]);
	b[1].resize(samples, 1, _W.type);
	b[1].zeros();

	//		W[0].randn(0, 0.1, 1);
	//		b[0].randn(0, 0.1, 1);
	//		W[1].randn(0, 0.1, 1);
	//		b[1].randn(0, 0.1, 1);
}

void SimpleAutoencoder::pass(const GpuMat &X)
{
	if(X.empty() || X.cols != W[0].rows || !func || !deriv)
		return;

	a[0] = X;
	for(int i = 0; i < 2; i++){
//		PRINT_GMAT10(a[i]);
//		PRINT_GMAT10(W[i]);
//		PRINT_GMAT10(b[i]);
		matmul_shared(a[i], W[i], z[i]);
//		W[i].save("W.txt");
//		a[i].save("a.txt");
//		z[i].save("z.txt");
//		PRINT_GMAT10(W[i]);
//		PRINT_GMAT10(z[i]);
		biasPlus(z[i], b[i]);
//		PRINT_GMAT10(z[i]);
		if(i == 0){
			(*func)(z[i], a[i + 1]);
//			PRINT_GMAT10(a[i + 1]);
		}else{
			a[i + 1] = z[i];
//			PRINT_GMAT10(a[i + 1]);
		}
	}

	double m = X.rows;

	sub(a[2], X, d);

//	PRINT_GMAT10(d);
	for(int i = 1; i > -1; --i){
		if(i > 0){
			(*deriv)(a[i], sz);
			matmulT2_shared(d, W[i], di);
//			PRINT_GMAT10(di);
			elemwiseMult(di, sz);
//			PRINT_GMAT10(di);
		}
//		a[i].save("ai.txt");
//		d.save("d.txt");
		matmulT1_shared(a[i], d, dW[i]);
		mulval(dW[i], 1./m);
//		dW[i].save("dWi.txt");
//		PRINT_GMAT10(d);
		sumRows_shared(d, db[i], 1./m);
//		PRINT_GMAT10(db[i]);
		db[i].swap_dims();
		if(i > 0)
			d = di;
	}
	transpose(dW[1], tw1);
	add(dW[0], tw1);
	transpose(dW[0], dW[1]);

	db[1].zeros();

//	PRINT_GMAT10(dW[0]);
//	PRINT_GMAT10(dW[1]);
//	PRINT_GMAT10(db[0]);
//	PRINT_GMAT10(db[1]);
	adam.pass(dW, db, W, b);
}

double SimpleAutoencoder::l2(const GpuMat &X)
{
	if(X.empty() || W[0].empty())
		return -1.;

	a[0] = X;
	for(int i = 0; i < 2; i++){
		matmul(a[i], W[i], z[i]);
		biasPlus(z[i], b[i]);
		if(i == 0){
			(*func)(z[i], a[i + 1]);
		}else{
			a[i + 1] = z[i];
		}
	}
	double m = X.rows;
	sub(a[2], X, d);
	elemwiseMult(d, d);
	double res = 0;
	if(d.type == GPU_FLOAT){
		ct::Matf df;
		convert_to_mat(d, df);
		res = df.sum() / m;

	}
	if(d.type == GPU_DOUBLE){
		ct::Matf dd;
		convert_to_mat(d, dd);
		res = dd.sum() / m;

	}
	return res;
}

}
