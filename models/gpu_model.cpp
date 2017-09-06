#include "gpu_model.h"

#include <QDebug>

#include "qt_work_mat.h"

#include "mnist_utils.h"

using namespace ct;

const int imageWidth = 28;
const int imageHeight = 28;

const int cnv_size = 3;

gpu_model::gpu_model()
{
	m_dropout_count = 3;
	m_iteration = 0;
	m_init = false;

	setConvLength();
}

bool gpu_model::isInit() const
{
	return m_init;
}

ct::Matf gpu_model::forward_gpu(const std::vector< gpumat::GpuMat > &X, bool use_dropout, bool converToMatf)
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

	m_cnv_out_size = m_cnv.back().szOut();
	m_cnv_out_len = m_cnv.back().outputFeatures();

	qDebug("--- input to MLP: %d ----", m_cnv_out_len);

	int input = m_cnv_out_len;
	int output = m_layers[0];

	m_gpu_mlp.resize(m_layers.size());

	for(size_t i = 0; i < m_layers.size(); i++){
		output = m_layers[i];

		gpumat::mlp& _mlp = m_gpu_mlp[i];

		_mlp.init(input, output, gpumat::GPU_FLOAT, i == m_layers.size() - 1? gpumat::SOFTMAX : gpumat::LEAKYRELU);

		input = output;
	}

	m_mlp_optim.init(m_gpu_mlp);
	m_cnv_optim.init(m_cnv);

	m_init = true;
}

void gpu_model::pass_batch_gpu(const std::vector< gpumat::GpuMat > &X, const gpumat::GpuMat &y)
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
//		gpumat::hsplit(m_gpu_mlp.front().DltA0, m_cnv.back().size() * m_cnv.back()[0].W.size(), ds);
		gpumat::mat2vec(m_gpu_mlp.front().DltA0, m_cnv.back().szK, ds);

		std::vector< gpumat::GpuMat > *pVec = &ds;
		for(int i = m_cnv.size() - 1; i > -1; i--){
			m_cnv[i].backward(*pVec, i == 0);
			pVec = &m_cnv[i].Dlt;
		}
	}

	m_mlp_optim.pass(m_gpu_mlp);
	m_cnv_optim.pass(m_cnv);

	m_iteration = m_mlp_optim.iteration();

}

uint32_t gpu_model::iteration() const
{
	return m_iteration;
}

void gpu_model::setAlpha(double val)
{
	m_mlp_optim.setAlpha(val);
	m_cnv_optim.setAlpha(val);
}

void gpu_model::setLayers(const std::vector<int> &layers)
{
	init_gpu(layers);
}

std::vector<gpumat::convnn_gpu> &gpu_model::cnv()
{
	return m_cnv;
}

void gpu_model::save_model()
{
	std::fstream fs;
	fs.open(name_model_cnv, std::ios_base::out | std::ios_base::binary);

	if(!fs.is_open()){
		printf("File %s not open\n", name_model_cnv.c_str());
		return;
	}

//	write_vector(fs, m_cnvlayers);
//	write_vector(fs, m_layers);

//	fs.write((char*)&m_szA0, sizeof(m_szA0));

	int cnvs = m_cnv.size(), mlps = m_gpu_mlp.size();

	/// size of convolution array
	fs.write((char*)&cnvs, sizeof(cnvs));
	/// size of mlp array
	fs.write((char*)&mlps, sizeof(mlps));

	for(size_t i = 0; i < m_cnv.size(); ++i){
		gpumat::convnn_gpu &cnv = m_cnv[i];
		cnv.write2(fs);
	}

	for(size_t i = 0; i < m_gpu_mlp.size(); ++i){
		m_gpu_mlp[i].write2(fs);
	}

	printf("model saved.\n");
}

void gpu_model::load_model()
{
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
	m_gpu_mlp.resize(mlps);

	printf("conv\n");
	for(size_t i = 0; i < m_cnv.size(); ++i){
		gpumat::convnn_gpu &cnv = m_cnv[i];
		cnv.read2(fs);
		printf("layer %d: rows %d, cols %d\n", i, cnv.W.rows, cnv.W.cols);
	}

	printf("mlp\n");
	for(size_t i = 0; i < m_gpu_mlp.size(); ++i){
		gpumat::mlp &mlp = m_gpu_mlp[i];
		mlp.read2(fs);
		printf("layer %d: rows %d, cols %d\n", i, mlp.W.rows, mlp.W.cols);
	}

	printf("model loaded.\n");
}

void gpu_model::conv(const std::vector< gpumat::GpuMat > &X, gpumat::GpuMat &X_out)
{
	if(X.empty())
		return;

	std::vector< gpumat::GpuMat > *pVec = (std::vector< gpumat::GpuMat > *)&X;

	for(size_t i = 0; i < m_cnv.size(); ++i){
		m_cnv[i].forward(pVec);
		pVec = &m_cnv[i].XOut();
	}

	gpumat::vec2mat(m_cnv.back().XOut(), X_out);
//	m_adds.hconcat(m_cnv.back(), X_out);

//	if(!saved){
//		for(size_t i = 0; i < m_cnv.size(); ++i){
//			for(size_t j = 0; j < m_cnv[i].size(); ++j){
//				m_cnv[i][j].clear();
//			}
//		}
	//	}
}

void gpu_model::setConvLength()
{
	time_t tm;
	time(&tm);
	ct::generator.seed(tm);

	m_cnv.resize(cnv_size);

	ct::Size szA0(imageWidth, imageHeight);
	m_cnv[0].init(szA0, 1, 1, 32, ct::Size(3, 3), gpumat::LEAKYRELU, true, true, false);
	m_cnv[1].init(m_cnv[0].szOut(), 32, 1, 64, ct::Size(3, 3), gpumat::LEAKYRELU, true, true, true);
	m_cnv[2].init(m_cnv[1].szOut(), 64, 1, 96, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true);

}

void gpu_model::setGpuDropout(size_t count, float prob)
{
	for(size_t i = 0; i < std::min(count, m_gpu_mlp.size() - 1); ++i){
		m_gpu_mlp[i].setDropout(true);
		m_gpu_mlp[i].setDropout(prob);
	}
}

void gpu_model::clearGpuDropout()
{
	for(size_t i = 0; i < m_gpu_mlp.size(); ++i){
		m_gpu_mlp[i].setDropout(false);
	}
}
