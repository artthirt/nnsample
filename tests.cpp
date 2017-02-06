#include "tests.h"

#include <QDebug>

#include "custom_types.h"
#include "nn.h"

typedef std::vector< unsigned char > vchar;

#define CHECK_VALUE(val, str) if(!(val)) std::cout << str << std::endl

#define CALC_MAT(_void_, result, caption, count){	\
	double tcc = tick();							\
	for(int i = 0; i < count; ++i){					\
		_void_;										\
	}												\
	tcc = tick() - tcc;								\
	tcc /= count;									\
	std::string s = (std::string)result;			\
	qDebug("%s time(ms): %f", caption, tcc);		\
	qDebug("%s\n", s.c_str());						\
}
//	std::cout << caption << " time(ms): " << tcc;
//	std::cout << endl << s.c_str();
//	std::cout << endl;
//}

#define PRINT_IMAGE(mat, width, height)	{						\
	float* dcn = mat.ptr();										\
	for(int i = 0; i < height; ++i){							\
		QString s;												\
		for(int j = 0; j < width; ++j){							\
			s += QString::number(dcn[i * width + j]) + "\t";	\
		}														\
		qDebug("%s", s.toStdString().c_str());					\
	}															\
	qDebug(" ");												\
}


#define PRINT_MAT(m, caption) {					\
	std::string s = m;							\
	qDebug("%s:\n%s\n", caption, s.c_str());	\
}

#ifdef _MSC_VER

#include <Windows.h>

LARGE_INTEGER freq_pc()
{
	static LARGE_INTEGER pc = {0};
	if(pc.QuadPart)
		return pc;
	QueryPerformanceFrequency(&pc);
	return pc;
}

double tick()
{
	LARGE_INTEGER pc, freq;
	QueryPerformanceCounter(&pc);
	freq = freq_pc();
	double res = (double)pc.QuadPart / freq.QuadPart;
	return res * 1e6;
}

#else

double tick()
{
	struct timespec res;
	clock_gettime(CLOCK_MONOTONIC, &res);
	double dres = res.tv_nsec + res.tv_sec * 1e9;
	return (double)dres / 1000.;
}

#endif

void test_shared()
{
	sm::shared_memory<vchar> data = sm::make_shared<vchar>(), u;

	data.get()->resize(10);

	CHECK_VALUE(u.ref() == -1, "u ref need -1");
	CHECK_VALUE(data.ref() == 1, "data ref need 1");
	{
		sm::shared_memory<vchar> k = data, l, j, n;
		sm::shared_memory<vchar> h = sm::make_shared<vchar>();
		h.get()->resize(12);
		(*h)[0] = 7;
		(*k)[1] = 2;
		l = k;
		j = l;
		n = k;
		CHECK_VALUE(data.ref() == 5, "data ref need 5");
		{
			sm::shared_memory<vchar> o = n, r = k;
			CHECK_VALUE(data.ref() == 7, "data ref need 7");
			(*r)[4] = 5;
			o = h;
			(*o)[1] = 3;
			CHECK_VALUE(data.ref() == 6, "data ref need 6");
			CHECK_VALUE(h.ref() == 2, "h ref need 2");
		}
		u = h;
		CHECK_VALUE(u.ref() == 2, "u ref need 2");
		CHECK_VALUE(data.ref() == 5, "data ref need 5");
	}

	for(size_t e = 0; e < 100; e++){
		sm::shared_memory< vchar > t = u;
		t = u;
		t = u;
		u = t;
		(*t)[11] = (uint8_t)e + 1;
		CHECK_VALUE(u.ref() == 2, "u ref need 2");
	}

	CHECK_VALUE(u.ref() == 1, "u ref need 1");
	CHECK_VALUE(data.ref() == 1, "data ref need 1");

	(*data)[0] = 1;
}

const int ww = 20;
const int hh = 16;

void conv_mat(ct::Matf &A, ct::Matf &W1, ct::Matf &W2)
{
	float dW1[] = {1, 1, 1,
				 1, 1, 1,
				 1, 1, 1};
	float dW2[] = {2, 2, 2,
				 2, 2, 2,
				 2, 2, 2};

	ct::Matf im = ct::Matf::zeros(hh, ww);

#if 0
	im.randn(0, 10);
#else
	for(int i = 0; i < im.rows; ++i){
		for(int j = 0; j < im.cols; ++j){
			im.at(i, j) = i + j;
		}
	}
#endif
	A = ct::Matf(1, im.total(), im.ptr());
	W1 = ct::Matf(3, 3, dW1);
	W2 = ct::Matf(3, 3, dW2);
}

void test_mat()
{
	using namespace ct;
	using namespace std;

	double dA[] = {
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 6, 5, 4
	};
	double dB[] = {
		4, 3,
		1, 6,
		4, 2
	};
	double dCtest1[] = {
		45, 51,
		38, 54,
		39, 61,
		40, 68
	};

	Matd At(3, 4, dA), B(3, 2, dB), C, Ctest1(4, 2, dCtest1);
	matmulT1(At, B, C);
	CHECK_VALUE(C == Ctest1, "matmulT1 wrong");

	Matd A, Bt;
	A = At.t();
	Bt = B.t();

	matmulT2(A, Bt, C);
	CHECK_VALUE(C == Ctest1, "matmulT2 wrong");

	double data[4 * 5] = {
		0, 1, 2, 3, 4,
		5, 6, 7, 8, 9,
		10, 11, 12, 13, 14,
		15, 16, 17, 18, 19
	};
	ct::Matd m1 = ct::Matd(4, 5, data);
	ct::Matd m2 = m1.t();

	std::string s1 = m1;
	std::string s2 = m2;

	qDebug("m1:\n%s\nm2:\n%s", s1.c_str(), s2.c_str());

	ct::Matd sm;
	sm = ct::softmax(m1, 1);
	s1 = sm;
	qDebug("SOFTMAX:\n%s\n", s1.c_str());

	Matf ims, W1, W2;
	conv_mat(ims, W1, W2);

	std::vector< ct::Matf > vW, vcn, gradW;
	std::vector< float > gradB;
	vW.push_back(W1);
	vW.push_back(W2);

	std::vector<float> b;
	b.push_back(0);
	b.push_back(0);

	//ct::Size sz = nn::conv2DW3x3(ims, 35, 21, 1, vW, vcn);
	ct::Size sz, sz1;

	CALC_MAT(sz = nn::conv2D(ims, ct::Size(ww, hh), 1, vW, b, vcn, nn::linear_func<float>), ims, "CALC", 10);

	qDebug("IMAGE:\n");
	PRINT_IMAGE(ims, ww, hh);

	qDebug("CONV:\n");

	qDebug("LAYERS");
	PRINT_IMAGE(vcn[0], sz.width, sz.height);
	PRINT_IMAGE(vcn[1], sz.width, sz.height);

	std::vector< ct::Matf > pool, masks;
	ct::Matf D;

	nn::subsample(vcn, sz, pool, masks, sz1);

	qDebug("MAXPOOL");
	PRINT_IMAGE(pool[0], sz1.width, sz1.height);

//	auto gradRelu = [](float v){return v > 0? 1 : 0;};

	qDebug("END TEST");
}

#ifdef _USE_GPU

#include "gpumat.h"
#include "helper_gpu.h"
#include "convnn_gpu.h"

void internal_test_gpu()
{
	ct::Matf A, W1, W2, tmp;
	conv_mat(A, W1, W2);
	gpumat::GpuMat gA, gAi, gMi, gAi2, gD;
	std::vector< gpumat::GpuMat > gWs, gA1;
	std::vector< float > bs;
	gWs.resize(2);
	bs.resize(2, 0);
	gpumat::convert_to_gpu(A, gA);
	gpumat::convert_to_gpu(W1, gWs[0]);
	gpumat::convert_to_gpu(W2, gWs[1]);
	ct::Size szA0(ww, hh);

	ct::Size szO = gpumat::conv2D(gA, szA0, 1, gWs, bs, gA1), szAi;

	std::string s1 = gA1[0].print();
	std::string s2 = gA1[1].print();

	qDebug("***GPU CONV2D**\nA1[0]:\n%s\nA1[1]:\n%s\n******", s1.c_str(), s2.c_str());

	qDebug("Conv2d[0]");
	gpumat::convert_to_mat(gA1[0], tmp);
	PRINT_IMAGE(tmp, szO.width, szO.height);
	qDebug("Conv2d[1]");
	gpumat::convert_to_mat(gA1[1], tmp);
	PRINT_IMAGE(tmp, szO.width, szO.height);

	gpumat::subsample(gA1[0], szO, gAi, gMi, szAi);
	gpumat::convert_to_mat(gAi, tmp);
	qDebug("Subsample");
	PRINT_IMAGE(tmp, szAi.width, szAi.height);

	gpumat::convert_to_mat(gMi, tmp);
	qDebug("Mask");
	PRINT_IMAGE(tmp, szO.width, szO.height);

	gpumat::upsample(gAi, szAi, szO, gMi, gAi2);

	gpumat::convert_to_mat(gAi2, tmp);
	qDebug("Upsample");
	PRINT_IMAGE(tmp, szO.width, szO.height);

	std::vector< gpumat::GpuMat > derivs, ws;
	gpumat::GpuMat ggW;
	float ggB;
	derivs.push_back(gAi2);
	ws.push_back(gWs[0]);

	gpumat::deriv_prev_cnv(derivs, ws, szO, szA0, 1, gD);

	gpumat::convert_to_mat(gD, tmp);
	qDebug("DerivPrevLayer");
	PRINT_IMAGE(tmp, szA0.width, szA0.height);

	gpumat::deriv_conv2D(gA, gD, szA0, szO, gWs[0].sz(), 1, ggW, ggB);

	gpumat::convert_to_mat(ggW, tmp);
	qDebug("DerivPrevLayer: B=%f", ggB);
	ct::Size szW = ggW.sz();
	PRINT_IMAGE(tmp, szW.width, szW.height);
}

#endif

void test_gpu_mat()
{
#ifdef _USE_GPU
	internal_test_gpu();
#endif
}

