#ifndef CONVNN_H
#define CONVNN_H

#include "custom_types.h"
#include "nn.h"

namespace convnn{

template<typename T>
class convnn{
public:
	typedef std::vector< ct::Mat_<T> > tvmat;

	convnn(){
		stride = 1;
		weight_size = 3;
		m_init = false;
	}

	ct::Mat_<T> A0;
	ct::Mat_<T> DltA0;
	tvmat A1;
	tvmat A2;
	tvmat W;
	tvmat rotW;
	std::vector< T > B;
	tvmat Masks;
	ct::Size szA0;
	ct::Size szA1;
	ct::Size szA2;
	int stride;
	int weight_size;
	nn::AdamOptimizer<T> m_optim;

	void init(int count_weight, const ct::Size& _szA0){
		W.resize(count_weight);
		B.resize(count_weight);

		szA0 = _szA0;

		nn::get_cnv_sizes(szA0, ct::Size(weight_size, weight_size), stride, szA1, szA2);

		std::normal_distribution<T> nrm(T(0), T(0.01));

		for(int i = 0; i < count_weight; ++i){
			W[i].setSize(weight_size, weight_size);
			W[i].randn(0, 0.01);
			B[i] = 0;//nrm(ct::generator);
		}
		rotateW();

		m_init = true;
	}

	void setAlpha(T alpha){
		m_optim.setAlpha(alpha);
	}

	void rotateW(){
		rotW.resize(W.size());
		for(int i = 0; i < W.size(); ++i){
			ct::flip(W[i], rotW[i]);
		}
	}

	template< typename Func >
	bool forward(const ct::Mat_<T>& mat, Func func){
		if(!m_init)
			throw new std::invalid_argument("convnn::forward: not initialized");
		A0 = mat;
		nn::conv2D(A0, szA0, stride, W, B, A1, func);
		ct::Size sztmp;
		return nn::subsample(A1, szA1, A2, Masks, sztmp);
	}

	template< typename Func >
	void back2conv(const tvmat& A1, const tvmat& dA2, tvmat& dA1, Func func){
		dA1.resize(A1.size());
		for(int i = 0; i < A1.size(); i++){
			dA1[i] = ct::elemwiseMult(dA2[i], func(A1[i]));
		}
	}

	template< typename Func >
	void backward(const std::vector< ct::Mat_<T> >& Delta, Func func){
		if(!m_init)
			throw new std::invalid_argument("convnn::backward: not initialized");
		std::vector< ct::Mat_<T> > dA2, dA1, dA0;
		nn::upsample(Delta, szA2, szA1, Masks, dA2);

		back2conv(A1, dA2, dA1, func);

		tvmat gradW;
		std::vector< T > gradB;
		ct::Size szW(weight_size, weight_size);

		nn::deriv_conv2D(A0, dA1, szA0, szA1, szW, stride, gradW, gradB);

		nn::deriv_prev_cnv(dA1, W, szA1, szA0, DltA0);

//		DltA0.setSize(dA0[0].rows, dA0[0].cols);
//		DltA0.fill(0);
//		for(int i = 0; i < dA0.size(); ++i){
//			DltA0 += dA0[i];
//		}
//		ct::elemMult(DltA0, A0);

//		for(int k = 0; k < gradW.size(); ++k){
//			std::string sw = gradW[k];
//			qDebug("gW[%d]:\n%s", k, sw.c_str());

//			std::stringstream ss;
//			ss << "A1" << k << ".txt";
//			ct::save_mat(dA1[k], ss.str());
//			ss.str("");
//			ss << "A2" << k << ".txt";
//			ct::save_mat(dA2[k], ss.str());
//			ss.str("");
//			ss << "D" << k << ".txt";
//			ct::save_mat(Delta[k], ss.str());
//		}

		m_optim.pass(gradW, gradB, W, B);

		rotateW();
	}

	static void hconcat(const std::vector< convnn<T> > &cnv, ct::Mat_<T>& _out){
		if(cnv.empty())
			return;
		std::vector< ct::Mat_<T> > slice;

		for(int i = 0; i < cnv.size(); ++i){
			ct::Mat_< T > res;
			nn::hconcat(cnv[i].A2, res);
			slice.push_back(res);
		}
		nn::hconcat(slice, _out);
	}

private:
	bool m_init;
};

}

#endif // CONVNN_H
