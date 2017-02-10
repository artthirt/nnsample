#ifndef MLP_H
#define MLP_H

#include "custom_types.h"
#include "common_types.h"

namespace ct{

template< typename T >
class mlp;

template< typename T >
class AdamMlp{
public:
	AdamMlp(){
		m_alpha = (T)0.001;
		m_betha1 = (T)0.9;
		m_betha2 = (T)0.99;
		m_iteration = 0;
		m_init = false;
	}

	T alpha()const{
		return m_alpha;
	}

	void setAlpha(T v){
		m_alpha = v;
	}

	T betha1() const{
		return m_betha1;
	}

	void setBetha1(T v){
		m_betha1 = v;
	}

	T betha2() const{
		return m_betha2;
	}

	void setBetha2(T v){
		m_betha2 = v;
	}

	uint32_t iteration() const{
		return m_iteration;
	}

	bool init(std::vector< ct::mlp<T> >& Mlp){
		if(Mlp.empty())
			return false;

		using namespace ct;

		m_iteration = 0;

		m_mW.resize(Mlp.size());
		m_mb.resize(Mlp.size());

		m_vW.resize(Mlp.size());
		m_vb.resize(Mlp.size());

		for(size_t i = 0; i < Mlp.size(); i++){
			ct::mlp<T>& _mlp = Mlp[i];
			m_mW[i].setSize(_mlp.W.size());
			m_vW[i].setSize(_mlp.W.size());
			m_mW[i].fill(0);
			m_vW[i].fill(0);

			m_mb[i].setSize(_mlp.B.size());
			m_vb[i].setSize(_mlp.B.size());
			m_mb[i].fill(0);
			m_vb[i].fill(0);
		}
		m_init = true;
		return true;
	}

	bool pass(std::vector< ct::mlp<T> >& Mlp){

		using namespace ct;

		m_iteration++;
		T sb1 = (T)(1. / (1. - pow(m_betha1, m_iteration)));
		T sb2 = (T)(1. / (1. - pow(m_betha2, m_iteration)));
		T eps = (T)(10e-8);

		for(size_t i = 0; i < Mlp.size(); ++i){
			ct::mlp<T>& _mlp = Mlp[i];
			m_mW[i] = m_betha1 * m_mW[i] + (T)(1. - m_betha1) * _mlp.gW;
			m_mb[i] = m_betha1 * m_mb[i] + (T)(1. - m_betha1) * _mlp.gB;

			m_vW[i] = m_betha2 * m_vW[i] + (T)(1. - m_betha2) * elemwiseSqr(_mlp.gW);
			m_vb[i] = m_betha2 * m_vb[i] + (T)(1. - m_betha2) * elemwiseSqr(_mlp.gB);

			Mat_<T> mWs = m_mW[i] * sb1;
			Mat_<T> mBs = m_mb[i] * sb1;
			Mat_<T> vWs = m_vW[i] * sb2;
			Mat_<T> vBs = m_vb[i] * sb2;

			vWs.sqrt(); vBs.sqrt();
			vWs += eps; vBs += eps;
			mWs = elemwiseDiv(mWs, vWs);
			mBs = elemwiseDiv(mBs, vBs);

			_mlp.W -= m_alpha * mWs;
			_mlp.B -= m_alpha * mBs;
		}
		return true;
	}
private:
	uint32_t m_iteration;
	T m_betha1;
	T m_betha2;
	T m_alpha;
	bool m_init;

	std::vector< ct::Mat_<T> > m_mW;
	std::vector< ct::Mat_<T> > m_mb;
	std::vector< ct::Mat_<T> > m_vW;
	std::vector< ct::Mat_<T> > m_vb;

};

template< typename T >
class mlp{
public:
	Mat_<T> *pA0;
	Mat_<T> W;
	Mat_<T> B;
	Mat_<T> Z;
	Mat_<T> A1;
	Mat_<T> DA1;
	Mat_<T> D1;
	Mat_<T> DltA0;
	Mat_<T> Dropout;
	Mat_<T> WDropout;
	Mat_<T> gW;
	Mat_<T> gB;

	mlp(){
		m_func = RELU;
		m_init = false;
		m_is_dropout = false;
		m_prob = (T)0.95;
		pA0 = nullptr;
	}

	void setDropout(bool val, T p = (T)0.95){
		m_is_dropout = val;
		m_prob = p;
	}

	void init(int input, int output){
		double n = 1./sqrt(input);

		W.setSize(input, output);
		W.randn(0., n);
		B.setSize(output, 1);
		B.randn(0, n);

		m_init = true;
	}

	inline void apply_func(const ct::Mat_<T>& Z, ct::Mat_<T>& A, etypefunction func){
		switch (func) {
			default:
			case RELU:
				A = relu(Z);
				break;
			case SOFTMAX:
				A = softmax(Z, 1);
				break;
			case SIGMOID:
				A = sigmoid(Z);
				break;
			case TANH:
				A = tanh(Z);
				break;
		}
	}
	inline void apply_back_func(const ct::Mat_<T>& D1, ct::Mat_<T>& D2, etypefunction func){
		switch (func) {
			default:
			case RELU:
				v_derivRelu(A1, DA1);
				break;
			case SOFTMAX:
//				A = softmax(A, 1);
				D1.copyTo(D2);
				return;
			case SIGMOID:
				v_derivSigmoid(A1, DA1);
				break;
			case TANH:
				v_derivTanh(A1, DA1);
				break;
		}
		elemwiseMult(D1, DA1, D2);
	}

	etypefunction funcType() const{
		return m_func;
	}

	void forward(const ct::Mat_<T> *mat, etypefunction func = RELU, bool save_A0 = true){
		if(!m_init || !mat)
			throw new std::invalid_argument("mlp::forward: not initialized. wrong parameters");
		pA0 = (Mat_<T>*)mat;
		m_func = func;

		if(m_is_dropout){
			ct::dropout(W.rows, W.cols, m_prob, Dropout);
			elemwiseMult(W, Dropout, WDropout);
			Z = *pA0 * WDropout;
		}else{
			Z = *pA0 * W;
		}

		Z.biasPlus(B);
		apply_func(Z, A1, func);

		if(!save_A0)
			pA0 = nullptr;
	}
	void backward(const ct::Mat_<T> &Delta, bool last_layer = false){
		if(!pA0 || !m_init)
			throw new std::invalid_argument("mlp::backward: not initialized. wrong parameters");

		apply_back_func(Delta, DA1, m_func);

		T m = Delta.rows;

		matmulT1(*pA0, DA1, gW);
		gW *= (T) 1. / m;

		if(m_is_dropout){
			elemwiseMult(gW, Dropout);
		}

		gB = (sumRows(DA1) * (1.f / m));
		gB.swap_dims();

		if(!last_layer){
			matmulT2(DA1, W, DltA0);
		}
	}

private:
	bool m_init;
	bool m_is_dropout;
	T m_prob;
	etypefunction m_func;
};

}

#endif // MLP_H
