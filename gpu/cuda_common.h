#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include "gpumat.h"

/**
  size of block for cuda gpu
*/
#define BLOCKSIZE	32

namespace gpumat{

	enum etypefunction{
		RELU = 1,
	};

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

		struct SmallMtxArray{
			enum {maxcount = 64};
			SmallMtxArray(){

			}
			SmallMtxArray(const std::vector< GpuMat >& gmat){
				if(maxcount < gmat.size())
					throw new std::invalid_argument("not enough size of array for store matrices");

				count = gmat.size();
				for(int i = 0; i < count; ++i){
					mtx[i] = gmat[i];
				}
			}

			int count;
			internal::Mtx mtx[maxcount];
		};

	}/* @end internal */

}/* @end gpumat */

#endif // CUDA_COMMON_H
