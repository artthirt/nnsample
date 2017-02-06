#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include "gpumat.h"

namespace internal{

struct Mtx{
	int rows;
	int cols;
	u_char* data;

	Mtx(){
		rows = cols = 0;
		data = 0;
	}
	Mtx(int rows, int cols, void* data){
		this->rows = rows;
		this->cols = cols;
		this->data = (u_char*)data;
	}
	Mtx(const gpumat::GpuMat& mat){
		rows = mat.rows;
		cols = mat.cols;
		data = mat.data;
	}
};

}

#endif // CUDA_COMMON_H
