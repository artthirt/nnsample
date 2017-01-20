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

		gpumat::elemiseSqr(gradW[i], sW);
		gpumat::elemiseSqr(gradB[i], sB);

		gpumat::add(m_vW[i], sW, m_betha2, (1. - m_betha2));
		gpumat::add(m_vb[i], sB, m_betha2, (1. - m_betha2));
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

}
