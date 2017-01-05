#ifndef NN_H
#define NN_H

#include <custom_types.h>
#include <vector>

namespace nn{

template< typename T >
class AdamOptimizer{
public:
	AdamOptimizer(){
		m_alpha = 0.01;
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

	uint iteration() const{
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

			double n = 1./sqrt(input);

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
		double sb1 = 1. / (1. - pow(m_betha1, m_iteration));
		double sb2 = 1. / (1. - pow(m_betha2, m_iteration));
		double eps = 10e-8;

		for(size_t i = 0; i < gradW.size(); ++i){
			m_mW[i] = m_betha1 * m_mW[i] + (1. - m_betha1) * gradW[i];
			m_mb[i] = m_betha1 * m_mb[i] + (1. - m_betha1) * gradB[i];

			m_vW[i] = m_betha2 * m_vW[i] + (1. - m_betha2) * elemwiseSqr(gradW[i]);
			m_vb[i] = m_betha2 * m_vb[i] + (1. - m_betha2) * elemwiseSqr(gradB[i]);

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

private:
	uint m_iteration;
	T m_betha1;
	T m_betha2;
	T m_alpha;

	std::vector< ct::Matd > m_mW;
	std::vector< ct::Matd > m_mb;
	std::vector< ct::Matd > m_vW;
	std::vector< ct::Matd > m_vb;
};

}

#endif // NN_H
