#include "gpu_model.h"

#include <QDebug>

#include "qt_work_mat.h"

using namespace ct;

const int imageWidth = 28;
const int imageHeight = 28;

gpu_model::gpu_model()
{
	m_dropout_count = 3;
	m_iteration = 0;
	m_init = false;

	m_count_cnvW.push_back(8);
	m_count_cnvW.push_back(3);
//	m_count_cnvW.push_back(1);
	m_conv_length = (int)m_count_cnvW.size();

	setConvLength(m_count_cnvW);

}

bool gpu_model::isInit() const
{
	return m_init;
}

ct::Matf gpu_model::forward_gpu(const gpumat::GpuMat &X, bool use_dropout, bool converToMatf)
{
	if(m_layers.empty() || X.empty())
		return Matf(0, 0);

	gpumat::etypefunction func = gpumat::RELU;

	conv(X, g_Xout);

	if(!use_dropout)
		clearGpuDropout();

	gpumat::GpuMat *pA = (gpumat::GpuMat*)&g_Xout;

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

void gpu_model::init_gpu(const std::vector< int >& layers)
{
	if(!layers.size())
		return;

	m_layers = layers;

	m_cnv_out_size = m_cnv.back()[0].szA2;
	m_cnv_out_len = m_cnv.back().size() * m_cnv.back()[0].szA2.area() * m_count_cnvW.back();

	qDebug("--- input to MLP: %d ----", m_cnv_out_len);

	int input = m_cnv_out_len;
	int output = m_layers[0];

	m_gpu_mlp.resize(m_layers.size());

	for(size_t i = 0; i < m_layers.size(); i++){
		output = m_layers[i];

		gpumat::mlp& _mlp = m_gpu_mlp[i];

		_mlp.init(input, output, gpumat::GPU_FLOAT);

		input = output;
	}

	m_gpu_adam.init(m_gpu_mlp);

	m_init = true;
}

void gpu_model::pass_batch_gpu(const gpumat::GpuMat &X, const gpumat::GpuMat &y)
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

		_mlp.backward(*pD);
		pD = &_mlp.DltA0;
	}

	/// convolution
	{
		gpumat::hsplit(m_gpu_mlp.front().DltA0, m_cnv.back().size() * m_cnv.back()[0].W.size(), ds);

		for(int i = m_cnv.size() - 1; i > -1; i--){
			std::vector< gpumat::convnn >& lrs = m_cnv[i];

			size_t kidx = 0;

			for(size_t j = 0; j < lrs.size(); ++j){
				gpumat::convnn &cnv = lrs[j];

				size_t kfirst = kidx;
				kidx += (cnv.W.size());

				if(i == m_cnv.size() - 1)
					cnv.backward(ds, gpumat::RELU, kfirst, kidx, i == 0);
				else
					cnv.backward(m_cnv[i + 1], gpumat::RELU, kfirst, kidx, i == 0);
			}
		}
	}

	m_gpu_adam.pass(m_gpu_mlp);

	m_iteration = m_gpu_adam.iteration();

}

uint gpu_model::iteration() const
{
	return m_iteration;
}

void gpu_model::setAlpha(double val)
{
	m_gpu_adam.setAlpha(val);

	for(size_t i = 0; i < m_cnv.size(); ++i){
		for(size_t j = 0; j < m_cnv[i].size(); ++j){
			gpumat::convnn& cnv = m_cnv[i][j];
			cnv.setAlpha(val);
		}
	}
}

void gpu_model::setLayers(const std::vector<int> &layers)
{
	init_gpu(layers);
}

std::vector<std::vector<gpumat::convnn> > &gpu_model::cnv()
{
	return m_cnv;
}

void gpu_model::conv(const gpumat::GpuMat &X, gpumat::GpuMat &X_out)
{
	if(X.empty())
		return;

	for(size_t i = 0; i < m_cnv.size(); ++i){
		std::vector< gpumat::convnn >& ls = m_cnv[i];

		if(i == 0){
			gpumat::convnn& m0 = ls[0];
			m0.forward(&X, gpumat::RELU);
		}else{
			for(size_t j = 0; j < m_cnv[i - 1].size(); ++j){
				size_t off1 = j * m_count_cnvW[i - 1];
				gpumat::convnn& m0 = m_cnv[i - 1][j];
				for(size_t k = 0; k < m_count_cnvW[i - 1]; ++k){
					size_t col = off1 + k;
					gpumat::convnn& mi = ls[col];
					mi.forward(&m0.A2[k], gpumat::RELU);
				}
			}
		}
	}

	m_adds.hconcat(m_cnv.back(), X_out);

//	if(!saved){
//		for(size_t i = 0; i < m_cnv.size(); ++i){
//			for(size_t j = 0; j < m_cnv[i].size(); ++j){
//				m_cnv[i][j].clear();
//			}
//		}
	//	}
}

void gpu_model::setConvLength(const std::vector<int> &count_cnvW, std::vector<int> *weight_sizes)
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
}

void gpu_model::setGpuDropout(size_t count, float prob)
{
	for(size_t i = 0; i < std::min(count, m_gpu_mlp.size() - 1); ++i){
		m_gpu_mlp[i].setDropout(true, prob);
	}
}

void gpu_model::clearGpuDropout()
{
	for(size_t i = 0; i < m_gpu_mlp.size(); ++i){
		m_gpu_mlp[i].setDropout(false);
	}
}
