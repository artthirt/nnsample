#ifndef NN_H
#define NN_H

#include <custom_types.h>
#include <vector>

#ifndef __GNUC__
typedef unsigned int uint;
#endif

namespace nn{

template< typename T >
class AdamOptimizer{
public:
	AdamOptimizer(){
		m_alpha = 0.001;
		m_betha1 = 0.9;
		m_betha2 = 0.999;
		m_iteration = 0;
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

	bool init(const std::vector< int >& layers, int samples){
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

			m_mW[i] = Mat_<T>::zeros(input, output);
			m_vW[i] = Mat_<T>::zeros(input, output);

			m_mb[i] = Mat_<T>::zeros(output, 1);
			m_vb[i] = Mat_<T>::zeros(output, 1);

			input = output;
		}
		return true;
	}

	bool pass(const std::vector< ct::Mat_< T > >& gradW, const std::vector< ct::Mat_< T > >& gradB,
			  std::vector< ct::Mat_<T> >& W, std::vector< ct::Mat_<T> >& b){
		if(!gradW.size() || gradW.size() != gradB.size() || gradW.size() != W.size())
			return false;

		using namespace ct;

		m_iteration++;
		T sb1 = 1. / (1. - pow(m_betha1, m_iteration));
		T sb2 = 1. / (1. - pow(m_betha2, m_iteration));
		T eps = 10e-8;

		for(size_t i = 0; i < gradW.size(); ++i){
			m_mW[i] = m_betha1 * m_mW[i] + (T)(1. - m_betha1) * gradW[i];
			m_mb[i] = m_betha1 * m_mb[i] + (T)(1. - m_betha1) * gradB[i];

			m_vW[i] = m_betha2 * m_vW[i] + (T)(1. - m_betha2) * elemwiseSqr(gradW[i]);
			m_vb[i] = m_betha2 * m_vb[i] + (T)(1. - m_betha2) * elemwiseSqr(gradB[i]);

			Mat_<T> mWs = m_mW[i] * sb1;
			Mat_<T> mBs = m_mb[i] * sb1;
			Mat_<T> vWs = m_vW[i] * sb2;
			Mat_<T> vBs = m_vb[i] * sb2;

			vWs.sqrt(); vBs.sqrt();
			vWs += eps; vBs += eps;
			mWs = elemwiseDiv(mWs, vWs);
			mBs = elemwiseDiv(mBs, vBs);

			W[i] -= m_alpha * mWs;
			b[i] -= m_alpha * mBs;
		}
		return true;
	}
	bool pass(const ct::Mat_< T > *gradW, const ct::Mat_< T >* gradB,
			  ct::Mat_<T> *W, ct::Mat_<T> *b, int count){
		if(!count || ! gradW || !gradB || !W || !b)
			return false;

		using namespace ct;

		m_iteration++;
		T sb1 = 1. / (1. - pow(m_betha1, m_iteration));
		T sb2 = 1. / (1. - pow(m_betha2, m_iteration));
		T eps = 10e-8;

		for(int i = 0; i < count; ++i){
			m_mW[i] = m_betha1 * m_mW[i] + (T)(1. - m_betha1) * gradW[i];
			m_mb[i] = m_betha1 * m_mb[i] + (T)(1. - m_betha1) * gradB[i];

			m_vW[i] = m_betha2 * m_vW[i] + (T)(1. - m_betha2) * elemwiseSqr(gradW[i]);
			m_vb[i] = m_betha2 * m_vb[i] + (T)(1. - m_betha2) * elemwiseSqr(gradB[i]);

			Mat_<T> mWs = m_mW[i] * sb1;
			Mat_<T> mBs = m_mb[i] * sb1;
			Mat_<T> vWs = m_vW[i] * sb2;
			Mat_<T> vBs = m_vb[i] * sb2;

			vWs.sqrt(); vBs.sqrt();
			vWs += eps; vBs += eps;
			mWs = elemwiseDiv(mWs, vWs);
			mBs = elemwiseDiv(mBs, vBs);

			W[i] -= m_alpha * mWs;
			b[i] -= m_alpha * mBs;
		}
		return true;
	}
	bool empty() const{
		return m_mW.empty() || m_mb.empty() || m_vW.empty() || m_vb.empty();
	}


private:
	uint32_t m_iteration;
	T m_betha1;
	T m_betha2;
	T m_alpha;

	std::vector< ct::Mat_<T> > m_mW;
	std::vector< ct::Mat_<T> > m_mb;
	std::vector< ct::Mat_<T> > m_vW;
	std::vector< ct::Mat_<T> > m_vb;
};

template<class T>
class SimpleAutoencoder
{
public:

	typedef ct::Mat_<T> (*tfunc)(const ct::Mat_<T>& t);

	SimpleAutoencoder(){
		func = 0;
		deriv = 0;
		m_neurons = 0;
	}

	T m_alpha;
	int m_neurons;

	ct::Mat_<T> W[2];
	ct::Mat_<T> b[2];

	tfunc func;
	tfunc deriv;

	void init(ct::Mat_<T>& _W, ct::Mat_<T>& _b, int samples, int neurons, tfunc fn, tfunc dfn){
		using namespace ct;

		func = fn;
		deriv = dfn;
		m_neurons = neurons;

		std::vector< int > layers;
		layers.push_back(neurons);
		layers.push_back(samples);
		adam.init(layers, samples);

		W[0] = _W;
		b[0] = _b;

		W[1] = _W.t();
		b[1] = Mat_<T>::zeros(samples, 1);

//		W[0].randn(0, 0.1, 1);
//		b[0].randn(0, 0.1, 1);
//		W[1].randn(0, 0.1, 1);
		b[1].randn(0, 0.1, 1);
	}

	void pass(const ct::Mat_<T>& X){
		if(X.empty() || X.cols != W[0].rows || !func || !deriv)
			return;
		using namespace ct;

		Mat_<T> a[3];
		Mat_<T> z[2], dW[2], db[2], d, di, sz;
		a[0] = X;
		for(int i = 0; i < 2; i++){
			z[i] = a[i] * W[i];
			z[i].biasPlus(b[i]);
			if(i == 0){
				a[i + 1] = (*func)(z[i]);
			}else{
				a[i + 1] = z[i];
			}
		}

		T m = X.rows;

		d = a[2] - X;

		for(int i = 1; i > -1; --i){
			if(i > 0){
				sz = (*deriv)(a[i]);
				matmulT2(d, W[i], di);
				di = elemwiseMult(di, sz);
			}
			matmulT1(a[i], d, dW[i]);
			dW[i] *= (T)(1./m);
			db[i] = (sumRows(d) * (T)(1./m)).t();
			if(i > 0)
				d = di;
		}
		dW[0] += dW[1].t();
		dW[1] = dW[0].t();

		adam.pass(dW, db, W, b, 2);
	}
	T l2(const ct::Mat_<T>& X) const{
		using namespace ct;

		if(X.empty() || W[0].empty())
			return -1.;

		Mat_<T> a[3];
		Mat_<T> z[2], d;
		a[0] = X;
		for(int i = 0; i < 2; i++){
			z[i] = a[i] * W[i];
			z[i].biasPlus(b[i]);
			if(i == 0){
				a[i + 1] = (*func)(z[i]);
			}else{
				a[i + 1] = z[i];
			}
		}
		T m = X.rows;
		d = a[2] - X;
		d = elemwiseMult(d, d);
		T res = d.sum() / m;
		return res;
	}
	AdamOptimizer<T> adam;
private:
};

}

#endif // NN_H
