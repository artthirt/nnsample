#ifndef NN_H
#define NN_H

#include <custom_types.h>
#include <vector>
#include <exception>

#ifndef __GNUC__
typedef unsigned int uint;
#endif

namespace nn{

template< typename T >
class AdamOptimizer{
public:
	AdamOptimizer(){
		m_alpha = (T)0.001;
		m_betha1 = (T)0.9;
		m_betha2 = (T)0.999;
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
		T sb1 = (T)(1. / (1. - pow(m_betha1, m_iteration)));
		T sb2 = (T)(1. / (1. - pow(m_betha2, m_iteration)));
		T eps = (T)(10e-8);

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
		T sb1 = (T)(1. / (1. - pow(m_betha1, m_iteration)));
		T sb2 = (T)(1. / (1. - pow(m_betha2, m_iteration)));
		T eps = (T)(10e-8);

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

template< typename T >
class MomentOptimizer{
public:
	MomentOptimizer(){
		m_alpha = 0.01;
		m_betha = 0.9;
	}

	void setAlpha(T val){
		m_alpha = val;
	}
	void setBetha(T val){
		m_betha = val;
	}

	void pass(const std::vector< ct::Mat_<T> > &gradW, const std::vector< T > &gradB,
			  std::vector< ct::Mat_<T> > &W, std::vector< T > &B)
	{
		if(W.empty() || gradW.size() != W.size() || gradB.empty() || gradB.size() != gradW.size())
			throw new std::invalid_argument("MomentOptimizer: wrong parameters");
		if(m_mW.empty()){
			m_mW.resize(W.size());
			m_mb.resize(W.size());
			for(int i = 0; i < m_mW.size(); ++i){
				m_mW[i] = ct::Mat_<T>::zeros(W[i].rows, W[i].cols);
				m_mb[i] = 0;
			}
		}

		for(int i = 0; i < m_mW.size(); ++i){
			ct::Mat_<T> tmp = m_mW[i];
			tmp *= m_betha;
			tmp += (1.f - m_betha) * gradW[i];
			m_mW[i] = tmp;

			m_mb[i] = m_betha * m_mb[i] + (1.f - m_betha) * gradB[i];
		}
		for(int i = 0; i < m_mW.size(); ++i){
			W[i] += ((-m_alpha) * m_mW[i]);
			B[i] += ((-m_alpha) * m_mb[i]);
		}
	}

private:
	std::vector< ct::Mat_<T> > m_mW;
	std::vector< T > m_mb;

	T m_alpha;
	T m_betha;
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
//		b[1].randn(0, 0.1, 1);
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

		T m = (T)X.rows;

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
		db[1] = Mat_<T>::zeros(db[1].size());

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
		T m = (T)X.rows;
		d = a[2] - X;
		d = elemwiseMult(d, d);
		T res = d.sum() / m;
		return res;
	}
	AdamOptimizer<T> adam;
private:
};

template<typename T>
inline T linear_func(T val)
{
	return val;
}

namespace internal{

template< typename T >
void matmul_conv(const T* dA, int x, int y, int xres, int yres, int width,
				 int width_res, int row, const ct::Mat_<T> & W, ct::Mat_<T> &Res)
{
	T *dW = &(*W.val)[0];
	T *dRes = &(*Res.val)[0] + row * Res.cols;

	for(int i = 0; i < W.rows; ++i){
		for(int j = 0; j < W.cols; ++j){
			T sC = 0;
			for(int k = 0; k < W.cols; ++k){
				sC += dA[(y + i) * width + (x + j)] * dW[j * W.cols + k];
			}
//			qDebug("r=%d c=%d", yres + i, xres + j);
			dRes[(yres + i) * width_res + (xres + j)] = sC;
		}
	}
}

/**
 * @brief conv
 * @param dA
 * @param x
 * @param y
 * @param xres
 * @param yres
 * @param width
 * @param width_res
 * @param row
 * @param W
 * @param Res
 */
template< typename T, typename Func >
inline void conv(const T* dA, int x, int y, int xres, int yres,
				 int width, int height, int width_res, int row,
				 const ct::Mat_<T> & W, T* &dRes,
				 Func func)
{
	T *dW = &(*W.val)[0];
//	T *dRes = &(*Res.val)[0] + row * Res.cols;

	T sC = 0;
	for(int i = 0; i < W.rows; ++i){
		if(y + i < height){
			for(int j = 0; j < W.cols; ++j){
				if(x + j < width)
					sC += dA[(y + i) * width + (x + j)] * dW[i * W.cols + j];
	//			qDebug("a: r=%d c=%d", y + i, x + j);
			}
		}
	}
//	qDebug("r: r=%d c=%d", yres, xres);
	dRes[(yres) * width_res + (xres)] = func(sC);
}

/**
 * @brief conv2D
 * @param dA
 * @param width
 * @param width_res
 * @param height_res
 * @param stride
 * @param row
 * @param W
 * @param Res
 * @param func - nonlinear operation
 */
template< typename T, typename Func >
inline void conv2D(const T* dA, int width, int height,
				   int width_res, int height_res, int stride, int row,
				   const ct::Mat_<T> & W, T *dRes,
				   Func func)
{
	int y = 0;
#pragma omp parallel for
	for(int yr = 0; yr < height_res; ++yr){
		y = yr * stride;
		for(int x = 0, xr = 0; xr < width_res; x += stride, ++xr){
			conv<T>(dA, x, y, xr, yr, width, height, width_res, row, W, dRes, func);
		}
	}
}

///**********

/**
 * @brief deriv_conv
 * @param dA
 * @param dgA1
 * @param gradA1
 * @param x
 * @param y
 * @param xres
 * @param yres
 * @param width
 * @param height
 * @param width_res
 * @param gradW
 */
template< typename T>
inline void deriv_conv(const T* dA, const T *dgA1, int x, int y, int xres, int yres,
				 int width, int height, int width_res, ct::Mat_<T> &gradW)
{
	T *dgW = &(*gradW.val)[0];

	T sC = dgA1[(yres) * width_res + (xres)];
	for(int i = 0; i < gradW.rows; ++i){
		if(y + i < height){
			for(int j = 0; j < gradW.cols; ++j){
				if(x + j < width){
//					qDebug("[%d, %d] = %f", y + i, x + j, dA[(y + i) * width + (x + j)]);
					dgW[i * gradW.cols + j] += dA[(y + i) * width + (x + j)] * sC;
				}
	//			qDebug("a: r=%d c=%d", y + i, x + j);
			}
		}
	}
}

/**
 * @brief deriv_conv2D
 * @param dA			reference to data of one image
 * @param dgA1			gradient of next layer
 * @param dId			indicies of using weight matrix
 * @param width			width image
 * @param height		height image
 * @param width_res		width of next layer
 * @param height_res	height of next layer
 * @param stride		stride
 * @param gradW			result: vector or gradient of weight matricies
 */
template< typename T>
inline void deriv_conv2D(const T* dA, const T *dgA1, const int *dId,
					int width, int height,
					int width_res, int height_res, int stride,
					std::vector< ct::Mat_<T> > &gradW)
{
	int y = 0;
#pragma omp parallel for
	for(int yr = 0; yr < height_res; ++yr){
		y = yr * stride;
		for(int x = 0, xr = 0; xr < width_res; x += stride, ++xr){
			int j = dId[yr * width_res + xr];
			deriv_conv<T>(dA, dgA1, x, y, xr, yr, width, height, width_res, gradW[j]);
		}
	}
}


}

/**
 * @brief conv2D
 * @param images	mattrix of images in rows
 * @param width		width image
 * @param height	height image
 * @param stride	stride
 * @param W			weight matrix 3x3
 * @param Res		result of convolution
 * @param func		nonlinear operation
 * @return
 */
template< typename T, typename Func >
ct::Size conv2D(const ct::Mat_<T>& images,
				const ct::Size& szI,
				int stride,
				const std::vector< ct::Mat_<T> >& W,
				const std::vector< T >& B,
				std::vector< ct::Mat_<T> >&A,
				Func func)
{
	if(images.empty() || W.empty() || B.empty() || W.size() != B.size()){
		std::cout << "conv2D wrong parameters\n";
		return ct::Size(0, 0);
	}

	int w_rows = W[0].rows;
	int w_cols = W[0].cols;

	int m = images.rows;

	ct::Size szO;
	szO.width	= (szI.width - w_cols + 1) / stride;
	szO.height	= (szI.height - w_rows + 1) / stride;

	int sz = szO.area();

	A.resize(W.size());
	for(size_t i = 0; i < A.size(); i++){
		A[i].setSize(images.rows, sz);
	}

	T *dI = images.ptr();

#pragma omp parallel for
	for(int i = 0; i < m; ++i){
		T *dIi = &dI[i * images.cols];

#pragma omp parallel for
		for(int j = 0; j < A.size(); ++j){
			T *dRes = A[j].ptr();
			T *dResi = &dRes[i * A[j].cols];

#pragma omp parallel for
			for(int y_res = 0; y_res < szO.height; y_res++){
				int y = y_res * stride;
				for(int x_res = 0; x_res < szO.width; x_res++){
					int x = x_res * stride;
					T *dW = W[j].ptr();

					T sum = 0;
#pragma omp simd
					for(int a = 0; a < w_rows; ++a){
						if(y + a < szI.height){
							for(int b = 0; b < w_cols; ++b){
								if(x + b < szI.width){
									T w = dW[a * w_cols + b];
									T g = dIi[(y + a) * szI.width + x + b];
									sum += w * g;
								}
							}
						}
					}
					sum += B[j];
					dResi[y_res * szO.width + x_res] = func(sum);
				}
			}

//			internal::conv2D(dIi, width, height, width_res, height_res, stride, i, W[j], dResi, func);
		}
	}
	return szO;
}

/**
 * @brief max_pool
 * @param Layers
 * @param Res
 * @param indexes
 * @return
 */
template< typename T >
bool max_pool(const std::vector< ct::Mat_<T> >&Layers, ct::Mat_<T>& Res, ct::Mat_<int>& indexes)
{
	if(Layers.empty())
		return false;

	int rows = Layers[0].rows;
	int cols = Layers[0].cols;

	if(!rows || !cols)
		return false;

	for(int i = 1; i < Layers.size(); ++i){
		if(Layers[i].rows != rows || Layers[i].cols != cols)
			return false;
	}

	Res.setSize(rows, cols);
	indexes.setSize(rows, cols);

	T* dRes = Res.ptr();
	int* dI = indexes.ptr();

#pragma omp parallel for
	for(int i = 0; i < rows; ++i){
#pragma omp parallel for
		for(int j = 0; j < cols; ++j){
			int kI = 0;
			T maxV = Layers[0].ptr()[i * cols + j];
			for(int k = 1; k < Layers.size(); ++k){
				T* dL = Layers[k].ptr();
				if(dL[i * cols + j] > maxV){
					maxV = dL[i * cols + j];
					kI = k;
				}
			}
			dRes[i * cols + j] = maxV;
			dI[i * cols + j] = kI;
		}
	}

	return true;
}

/**
 * @brief deriv_conv2D
 * derivative of convolution
 * @param A0		current layer
 * @param gradA1	gradient from next layer
 * @param indexes	indexes of using weight matrix

 * @param gradW		result vector of gradient of weight matricies
 * @return
 */
template< typename T >
void deriv_conv2D(const ct::Mat_<T>& A0,
					  const ct::Mat_<T>& gradA1,
					  const ct::Mat_<int>& indexes,
					  const ct::Size& szA0,
					  const ct::Size& szA1,
					  const ct::Size &szW,
					  uint countW,
					  int stride,
					  std::vector< ct::Mat_<T> >&gradW,
					  std::vector< T >&gradB)
{
	if(A0.empty() || gradA1.empty() || !countW || !stride){
		std::cout << "deriv_conv2D wrong parameters\n";
	}

	gradW.resize(countW);
	gradB.resize(countW);
	for(int i = 0; i < gradW.size(); ++i){
		gradW[i] = ct::Mat_<T>::zeros(szW.height, szW.width);
		gradB[i] = 0;
	}

	int m = A0.rows;

	T *dA = &(*A0.val)[0];
	T *dgA1 = &(*gradA1.val)[0];
	int* dId = &(*indexes.val)[0];

//#pragma omp parallel for
//	for(int i = 0; i < m; ++i){
//		T *dAi = &dA[i * A0.cols];
//		T *dgA1i = &dgA1[i * gradA1.cols];
//		int *dIi = &dId[i * gradA1.cols];

////#pragma omp parallel for
//		internal::deriv_conv2D(dAi, dgA1i, dIi, width, height,
//							   width_res, height_res, stride, gradW);
//	}

	for(int i = 0; i < m; ++i){
		T *dAi		= &dA[A0.cols * i];
		T *dgA1i	= &dgA1[gradA1.cols * i];
		int *dIi	= &dId[indexes.cols * i];

#pragma omp parallel for
		for(int y = 0; y < szA1.height; ++y){
			int y0 = y * stride;
#pragma omp parallel for
			for(int x = 0; x < szA1.width; ++x){
				int x0 = x * stride;
				int id = dIi[szA1.width * y + x];
				T *dgW = gradW[id].ptr();
				T d = dgA1i[szA1.width * y + x];

#pragma omp simd
				for(int a = 0; a < szW.height; ++a){
					if(y0 + a < szA0.height){
						for(int b = 0; b < szW.width; ++b){
							if(x0 + b < szA0.width){
								T a0 = dAi[szA0.width * (y0 + a) + x0 + b];
								dgW[a * szW.width + b] += a0 * d;
							}
						}
					}
				}
			}
		}
	}

	for(int i = 0; i < gradW.size(); ++i){
		gradW[i] *= (T)(1./m);
	}

	for(int i = 0; i < m; ++i){
		T *dgA1i	= &dgA1[gradA1.cols * i];
		int *dIi	= &dId[indexes.cols * i];
		for(int y = 0; y < szA1.height; ++y){
			for(int x = 0; x < szA1.width; ++x){
				int id = dIi[szA1.width * y + x];

				gradB[id] += dgA1i[szA1.width * y + x];
			}
		}
	}
	for(int i = 0; i < gradB.size(); ++i){
		gradB[i] /= (T)m;
	}
}

template< typename T >
void deriv_prev_cnv(const ct::Mat_<T>& deriv,
					const std::vector< ct::Mat_<T> >& W,
					const ct::Mati& indexes,
					const ct::Size& sL, const ct::Size& sLsub1,
					ct::Mat_<T>& D)
{
	if(deriv.empty() || W.empty() || indexes.empty())
		return;

	int m = deriv.rows;
	int w_rows = W[0].rows;
	int w_cols = W[0].cols;

	D.setSize(deriv.rows, sLsub1.area());
	D.fill(0);

	T *dA = deriv.ptr();
	T *dD = D.ptr();
	int* dI = indexes.ptr();

	float sz = w_rows * w_cols;

#pragma omp parallel for
	for(int i = 0; i < m; ++i){
		T *dAi = &dA[i * deriv.cols];
		T *dDi = &dD[i * D.cols];
		int *dIi = &dI[i * indexes.cols];

#pragma omp parallel for
		for(int y = 0; y < sLsub1.height; ++y){
			for(int x = 0; x < sLsub1.width; ++x){

//				int xi = std::max(0, std::min(x, sL.width - 1));
//				int yi = std::max(0, std::min(y, sL.height - 1));
//				int id = dIi[yi * sL.width + xi];

//				T *dW = W[id].ptr();
//				if(id != 0)
//					qDebug("id=%d", id);

				T sum = 0;
#pragma omp simd
				for(int a = 0; a < w_rows; ++a){
					if(y - a >= 0 && y - a < sL.height){
						for(int b = 0; b < w_cols; ++b){
							if(x - b >=0 && x - b < sL.width){
								int idx = (y - a) * sL.width + (x - b);
								int id = dIi[idx];

								T *dW = W[id].ptr();

								T d = dAi[idx];
								T w = dW[(a) * w_cols + (b)];
								sum += d * w;
							}
						}
					}
				}
				dDi[y * sLsub1.width + x] += sum;// / (sz);
			}
		}
	}
}

}

#endif // NN_H
