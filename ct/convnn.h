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
	tvmat prevW;
	std::vector< T > B;
	std::vector< T > prevB;
	tvmat Masks;
	ct::Size szA0;
	ct::Size szA1;
	ct::Size szA2;
	int stride;
	int weight_size;
	nn::AdamOptimizer<T> m_optim;

	tvmat gradW;
	std::vector< T > gradB;
	std::vector< ct::Mat_<T> > dA2, dA1;

	void setWeightSize(int ws){
		weight_size = ws;
		if(W.size())
			init(W.size(), szA0);
	}

	void init(size_t count_weight, const ct::Size& _szA0){
		W.resize(count_weight);
		B.resize(count_weight);

		szA0 = _szA0;

		nn::get_cnv_sizes(szA0, ct::Size(weight_size, weight_size), stride, szA1, szA2);

		update_random();

		m_init = true;
	}

	void save_weight(){
		prevW.resize(W.size());
		prevB.resize(B.size());
		for(int i = 0; i < W.size(); ++i){
			W[i].copyTo(prevW[i]);
			prevB[i] = B[i];
		}
	}
	void restore_weights(){
		if(prevW.empty() || prevB.empty())
			return;
		for(int i = 0; i < W.size(); ++i){
			prevW[i].copyTo(W[i]);
			B[i] = prevB[i];
		}
	}

	void update_random(){
		for(int i = 0; i < W.size(); ++i){
			W[i].setSize(weight_size, weight_size);
			W[i].randn(0, 0.1);
			B[i] = (T)0.1;
		}
	}

	void setAlpha(T alpha){
		m_optim.setAlpha(alpha);
	}

	void clear(){
		A0.clear();
		A1.clear();
		A1.clear();
		Masks.clear();
	}

	template< typename Func >
	bool forward(const ct::Mat_<T>& mat, Func func){
		if(!m_init)
			throw new std::invalid_argument("convnn::forward: not initialized");
		A0 = mat;
		nn::conv2D(A0, szA0, stride, W, B, A1, func);
		ct::Size sztmp;
		bool res = nn::subsample(A1, szA1, A2, Masks, sztmp);
		return res;
	}

	template< typename Func >
	void back2conv(const tvmat& A1, const tvmat& dA2, tvmat& dA1, Func func){
		dA1.resize(A1.size());
		for(size_t i = 0; i < A1.size(); i++){
			ct::elemwiseMult(dA2[i], func(A1[i]), dA1[i]);
		}
	}

	template< typename Func >
	void backward(const std::vector< ct::Mat_<T> >& Delta, Func func){
		if(!m_init)
			throw new std::invalid_argument("convnn::backward: not initialized");
		nn::upsample(Delta, szA2, szA1, Masks, dA2);

		back2conv(A1, dA2, dA1, func);

		ct::Size szW(weight_size, weight_size);

		nn::deriv_conv2D(A0, dA1, szA0, szA1, szW, stride, gradW, gradB);

		nn::deriv_prev_cnv(dA1, W, szA1, szA0, DltA0);

		m_optim.pass(gradW, gradB, W, B);
	}

	static void hconcat(const std::vector< convnn<T> > &cnv, ct::Mat_<T>& _out){
		if(cnv.empty())
			return;
		std::vector< ct::Mat_<T> > slice;

		for(size_t i = 0; i < cnv.size(); ++i){
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
