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

ct::Matf gpu_model::forward_gpu(const ct::Matf &X)
{
	if(m_layers.empty() || X.empty())
		return Matf(0, 0);

	init_arrays();

	Matf a;

	gpumat::convert_to_gpu(X, m_gX);

	conv(m_gX, g_a[0]);

	for(size_t i = 0; i < m_layers.size(); i++){
		gpumat::matmul_shared(g_a[i], m_gW[i], g_z[i]);

		gpumat::biasPlus(g_z[i], m_gb[i]);
		if(i < m_layers.size() - 1){
			gpumat::reLu(g_z[i], g_a[i + 1]);
		}else{
			gpumat::softmax(g_z[i], 1, g_a[i + 1], partZ);
		}
	}
	gpumat::convert_to_mat(g_a.back(), a);

	return a;
}

void gpu_model::init_gpu(const std::vector< int >& layers, int seed)
{
	if(!layers.size())
		return;

	m_layers = layers;

	m_cnv_out_size = m_cnv.back()[0].szA2;
	m_cnv_out_len = m_cnv.back().size() * m_cnv.back()[0].szA2.area() * m_count_cnvW.back();

	qDebug("--- input to MLP: %d ----", m_cnv_out_len);

	int input = m_cnv_out_len;
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

	if(!m_gpu_adam.init(m_layers, m_cnv_out_len, gpumat::GPU_FLOAT)){
		std::cout << "optimizer not init\n";
	}
	m_init = true;
}

void gpu_model::pass_batch_gpu(const gpumat::GpuMat &X, const gpumat::GpuMat &y)
{
	if(m_gW.empty() || m_gb.empty() || m_layers.empty() ||
			m_layers.back() != y.cols){
		std::cout << "wrong parameters of model\n";
		return;
	}

	/// forward
	init_arrays();

	//// CONV

	conv(X, g_a[0]);

	//// MLP

	//g_a[0] = m_cnvA;

	if(m_DropoutT.empty()){
		m_Dropout.resize(m_dropout_count);
		m_DropoutT.resize(m_dropout_count);
	}

	int max_layers = m_dropout_count;//std::min((int)(m_layers.size() - 2), m_dropout_count);

	for(size_t i = 0; i < m_layers.size(); i++){
		if(i < (size_t)max_layers){
			Matf d;
			ct::dropout(m_gW[i].rows, m_gW[i].cols, 0.92f, d);
			gpumat::convert_to_gpu(d, m_Dropout[i]);
			m_DropoutT[i] = m_Dropout[i];

			gpumat::elemwiseMult(m_Dropout[i], m_gW[i]);
			gpumat::matmul_shared(g_a[i], m_Dropout[i], g_z[i]);
		}else{
			gpumat::matmul_shared(g_a[i], m_gW[i], g_z[i]);
		}
		gpumat::biasPlus(g_z[i], m_gb[i]);
		if(i < m_layers.size() - 1){
			gpumat::reLu(g_z[i], g_a[i + 1]);
		}else{
			gpumat::softmax(g_z[i], 1, g_a[i + 1], partZ);
		}
	}

	if(g_dW.empty()){
		g_dW.resize(m_layers.size());
		g_dB.resize(m_layers.size());
	}

	float m = X.rows;

	gpumat::sub(g_a.back(), y, g_d.back());

	/// backward

	for(int i = (int)m_layers.size() - 1; i > -1; --i){
		{
			gpumat::deriv_reLu(g_a[i], g_sz[i]);

			gpumat::matmulT2_shared(g_d[i + 1], m_gW[i], g_d[i]);
			gpumat::elemwiseMult(g_d[i], g_sz[i]);
		}
		gpumat::matmulT1_shared(g_a[i], g_d[i + 1], g_dW[i]);
		gpumat::mulval(g_dW[i], 1./m);

		if(i < max_layers){
			gpumat::elemwiseMult(g_dW[i], m_DropoutT[i]);
		}

		g_dB[i].swap_dims();

		gpumat::sumRows_shared(g_d[i + 1], g_dB[i], (1./m));

		g_dB[i].swap_dims();
	}

	/// convolution
	{
		ds.resize(m_cnv.size() + 1);

		gpumat::hsplit(g_d[0], m_cnv.back().size() * m_cnv.back()[0].W.size(), ds.back());

		for(int i = 0; i < ds.back().size(); ++i){
			qt_work_mat::q_save_mat(ds.back()[i], QString::number(i) + "_ds.txt");
		}

		for(int i = m_cnv.size() - 1; i > -1; i--){
			std::vector< gpumat::convnn >& lrs = m_cnv[i];
			std::vector< gpumat::GpuMat > di;

//			qDebug("LR[%d]-----", i);

			if(i > 0)
				ds[i].resize(lrs.size());

			int kidx = 0;
			int nidx = 0;

			for(size_t j = 0; j < lrs.size(); ++j){
				gpumat::convnn &cnv = lrs[j];

				int kfirst = kidx;
//				for(size_t k = 0; k < cnv.W.size(); ++k){
//					ds[i][kidx++] = ds[i + 1][j * cnv.W.size() + k];
//					//dsi.push_back(ds[j * cnv.W.size() + k]);
//				}
				kidx += j * cnv.W.size() + (cnv.W.size());

				for(int l = kfirst; l < kidx; ++l){
					qt_work_mat::q_save_mat(ds[i + 1][l], QString::number(l) + "_dsi.txt");
				}

				cnv.backward(ds[i + 1], gpumat::RELU, kfirst, kidx);
				ds[i][nidx++] = cnv.DltA0;
				qt_work_mat::q_save_mat(cnv.DltA0, "dltA0_" + QString::number(nidx) + ".txt");
			}
		}
	}

	m_gpu_adam.pass(g_dW, g_dB, m_gW, m_gb);

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
	init_gpu(layers, 1);
}

void gpu_model::conv(const gpumat::GpuMat &X, gpumat::GpuMat &X_out)
{
	if(X.empty())
		return;

	for(size_t i = 0; i < m_cnv.size(); ++i){
		std::vector< gpumat::convnn >& ls = m_cnv[i];

		if(i == 0){
			gpumat::convnn& m0 = ls[0];
			m0.forward(X, gpumat::RELU);
		}else{
			for(size_t j = 0; j < m_cnv[i - 1].size(); ++j){
				int off1 = j * m_count_cnvW[i - 1];
				gpumat::convnn& m0 = m_cnv[i - 1][j];
				for(int k = 0; k < m_count_cnvW[i - 1]; ++k){
					int col = off1 + k;
					gpumat::convnn& mi = ls[col];
					mi.forward(m0.A2[k], gpumat::RELU);
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

void gpu_model::init_arrays()
{
	if(g_z.size() != m_layers.size()){
		g_z.resize(m_layers.size());
		g_a.resize(m_layers.size() + 1);
		g_d.resize(m_layers.size() + 1);
		g_sz.resize(m_layers.size());
	}
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
