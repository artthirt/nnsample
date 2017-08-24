#ifndef MNIST_UTILS_H
#define MNIST_UTILS_H

#include <vector>

const std::string name_model_cnv("model_conv.bin");

template< typename T >
void translate(int x, int y, int w, int h, T *X)
{
	std::vector<T>d;
	d.resize(w * h);

#pragma omp parallel for
	for(int i = 0; i < h; i++){
		int newi = i + x;
		if(newi >= 0 && newi < h){
			for(int j = 0; j < w; j++){
				int newj = j + y;
				if(newj >= 0 && newj < w){
					d[newi * w + newj] = X[i * w + j];
				}
			}
		}
	}
	for(size_t i = 0; i < d.size(); i++){
		X[i] = d[i];
	}
}

template< typename T >
void rotate_mnist(int w, int h, T angle, T *X)
{
	T cw = w / 2;
	T ch = h / 2;

	std::vector<T> d;
	d.resize(w * h);

	for(int y = 0; y < h; y++){
		for(int x = 0; x < w; x++){
			T x1 = x - cw;
			T y1 = y - ch;

			T nx = x1 * cos(angle) + y1 * sin(angle);
			T ny = -x1 * sin(angle) + y1 * cos(angle);
			nx += cw; ny += ch;
			int ix = nx, iy = ny;
			if(ix >= 0 && ix < w && iy >= 0 && iy < h){
				T c = X[y * w + x];
				d[iy * w + ix] = c;
			}
		}
	}
	for(size_t i = 0; i < d.size(); i++){
		X[i] = d[i];
	}
}


#endif // MNIST_UTILS_H
