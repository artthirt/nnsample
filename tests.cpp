#include "tests.h"

#include <QDebug>
#include <cmath>

#include "common_types.h"
#include "custom_types.h"
#include "nn.h"
#include "matops.h"
#include "convnn2_gpu.h"

#include "qt_work_mat.h"

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
			s += QString::number(dcn[i * width + j], 'f', 5) + "\t";	\
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

	Matf A0, W1, W2, Ai, Ai2, Mi;
	conv_mat(A0, W1, W2);

	std::vector< ct::Matf > vW, vcn, gradW, A1;
	std::vector< float > gradB;
	vW.push_back(W1);
	vW.push_back(W2);

	std::vector<float> b;
	b.push_back(0);
	b.push_back(0);

	ct::Size szA0(ww, hh);
	//ct::Size sz = nn::conv2DW3x3(ims, 35, 21, 1, vW, vcn);
	ct::Size szO = ct::conv2D(A0, szA0, 1, vW, b, A1, ct::RELU), szAi;

	s1 = A1[0].print();
	s2 = A1[1].print();

	qDebug("***MAT CONV2D**\nA1[0]:\n%s\nA1[1]:\n%s\n******", s1.c_str(), s2.c_str());

	qDebug("Conv2d[0]");
	PRINT_IMAGE(A1[0], szO.width, szO.height);
	qDebug("Conv2d[1]");
	PRINT_IMAGE(A1[1], szO.width, szO.height);

	ct::subsample(A1[0], szO, Ai, Mi, szAi);
	qDebug("Subsample");
	PRINT_IMAGE(Ai, szAi.width, szAi.height);

	qDebug("Mask");
	PRINT_IMAGE(Mi, szO.width, szO.height);

	ct::upsample(Ai, szAi, szO, Mi, Ai2);

	qDebug("Upsample");
	PRINT_IMAGE(Ai2, szO.width, szO.height);

	std::vector< ct::Matf > derivs, ws;
	ct::Matf ggW, D;
	float ggB;
	derivs.push_back(Ai2);
	ws.push_back(vW[0]);

	ct::deriv_prev_cnv(derivs, ws, szO, szA0, D);

	qDebug("DerivPrevLayer");
	PRINT_IMAGE(D, szA0.width, szA0.height);

	ct::deriv_conv2D(A0, Ai2, szA0, szO, ws[0].size(), 1, ggW, ggB);

	qDebug("deriv_conv2D: B=%f", ggB);
	ct::Size szW = ggW.size();
	PRINT_IMAGE(ggW, szW.width, szW.height);

	float test_data[] = {
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1
	},
	test_data2[] = {
		2, 2,
		2, 2
	};
	ct::Matf Test(1, 16, test_data), Test2(1, 4, test_data2);

	ct::deriv_conv2D(Test, Test2, ct::Size(4, 4), ct::Size(2, 2), ws[0].size(), 1, ggW, ggB);
	qDebug("deriv_conv2D: B=%f", ggB);
	szW = ggW.size();
	PRINT_IMAGE(ggW, szW.width, szW.height);

	qDebug("END MAT TEST");
}

#ifdef _USE_GPU

#include "gpumat.h"
#include "helper_gpu.h"
#include "convnn_gpu.h"

std::string fromDouble(double val)
{
	std::stringstream ss;
	ss << val;
	return ss.str();
}

void internal_test_gpu()
{
	gpumat::BN bn;
	ct::BN<float> cbn;

	std::vector< gpumat::GpuMat > g_X, g_Y, g_Y1, g_D;
	std::vector< ct::Matf > Xs, Ys, Ys1, Ds;

	int cnt = 5;
	g_X.resize(cnt);

	int index = 0;

	std::string str, str2;

	int channels = 64;

	str2 = "Xs = [";
	for(gpumat::GpuMat& g_Xi: g_X){
		ct::Matf X(3136, channels);
		float *dX = X.ptr();
		for(int i = 0; i < X.total(); ++i){
			float val = (float)i/X.total() * 3.;
			val = 2 * sin(index + (val) + 1.5 * cos(index * 0.1));
			dX[i] = val;
		}
		gpumat::convert_to_gpu(X, g_Xi);
		Xs.push_back(X);
		str += "X_" + std::to_string(++index) + "=" + X.print() + ";\n";
		str2 += "reshape(X_" + std::to_string(index) + "', [1, 200]); ";
	}
	str2 += "];\n";
	std::fstream fs;
	fs.open("labX.m", std::ios_base::out);
	fs << str << std::endl << str2;
	fs.close();

	bn.X = &g_X;
	bn.Y = &g_Y;
	bn.D = &g_D;
	bn.channels = channels;

	bn.normalize();					//// <----------

//	gpumat::GpuMat g_tMean;
//	gpumat::transpose(bn.Mean, g_tMean);
	gpumat::save_gmat(bn.Mean, "Mean.txt");
	gpumat::save_gmat(bn.Var, "Var.txt");

	str = "";
	str2 = "Ys = [";
	index = 0;
	for(gpumat::GpuMat& g_Yi: g_Y){
		str += "Y_" + std::to_string(++index) + "=" + g_Yi.print() + ";\n";
		str2 += "Y_" + std::to_string(index) + "; ";
	}
	str2 += "];\n";
	fs.open("labY.m", std::ios_base::out);
	fs << str << std::endl << str2;
	fs.close();

	/////////////////////////////

	cbn.X = &Xs;
	cbn.Y = &Ys;
	cbn.D = &Ds;

	cbn.channels = channels;

	cbn.normalize();					//// <---------

	ct::save_mat(cbn.Mean, "cMean.txt");
	ct::save_mat(cbn.Var, "cVar.txt");

	str = "";
	str2 = "Ys = [";
	index = 0;
	for(ct::Matf& Yi: Ys){
		str += "Y_" + std::to_string(++index) + "=" + Yi.print() + ";\n";
		str2 += "Y_" + std::to_string(index) + "; ";
	}
	str2 += "];\n";
	fs.open("labcY.m", std::ios_base::out);
	fs << str << std::endl << str2;
	fs.close();

	Ds.resize(Xs.size());
	g_D.resize(Ds.size());

	index = 0;
	for(ct::Matf& Di: Ds){
		Ys[index].copyTo(Di);
//		Di.setSize(Xs[index].size());
//		Di.randn(0, 0.1);
		gpumat::convert_to_gpu(Di, g_D[index]);
		index++;
	}
//	cbn.Var.copyTo(cbn.gamma);
//	cbn.Mean.copyTo(cbn.betha);

#if 1
	cbn.denormalize();

	str = "";
	str2 = "Ds = [";
	index = 0;
	for(ct::Matf& Di: cbn.Dout){
		str += "D_" + std::to_string(++index) + "=" + Di.print() + ";\n";
		str2 += "D_" + std::to_string(index) + "; ";
	}
	str2 += "];\n";
	fs.open("labcD.m", std::ios_base::out);
	fs << str << std::endl << str2;
	fs.close();

//	ct::save_mat(cbn.Var, "cdmean.txt");
	ct::save_mat(cbn.dgamma, "cdgamma.txt");
	ct::save_mat(cbn.dbetha, "cdbetha.txt");
#else
	cbn.X = &Ys;
	cbn.Y = &Ys1;

	cbn.Var.copyTo(cbn.gamma);
	cbn.Mean.copyTo(cbn.betha);
	cbn.normalize();

	str = "";
	str2 = "Ys = [";
	index = 0;
	for(ct::Matf& Yi: Ys1){
		str += "Y_" + std::to_string(++index) + "=" + Yi.print() + ";\n";
		str2 += "Y_" + std::to_string(index) + "; ";
	}
	str2 += "];\n";
	fs.open("labcY1.m", std::ios_base::out);
	fs << str << std::endl << str2;
	fs.close();
#endif
	////

#if 1
	bn.denormalize();

	str = "";
	str2 = "Ds = [";
	index = 0;
	for(gpumat::GpuMat& Di: bn.Dout){
		str += "D_" + std::to_string(++index) + "=" + Di.print() + ";\n";
		str2 += "D_" + std::to_string(index) + "; ";
	}
	str2 += "];\n";
	fs.open("labD.m", std::ios_base::out);
	fs << str << std::endl << str2;
	fs.close();

//	gpumat::save_gmat(bn.Mean, "dmean.txt");
//	gpumat::save_gmat(bn.dVar, "dvar.txt");
	gpumat::save_gmat(bn.dgamma, "dgamma.txt");
	gpumat::save_gmat(bn.dbetha, "dbetha.txt");
	std::cout << "end\n";
#else

	bn.X = &g_Y;
	bn.Y = &g_Y1;

	bn.Var.copyTo(bn.gamma);
	bn.Mean.copyTo(bn.betha);

	bn.denormalize();;

//	gpumat::GpuMat g_tMean;
//	gpumat::transpose(bn.Mean, g_tMean);
	gpumat::save_gmat(bn.Mean, "Mean.txt");
	gpumat::save_gmat(bn.Var, "Var.txt");

	str = "";
	str2 = "Ys = [";
	index = 0;
	for(gpumat::GpuMat& g_Yi: g_Y1){
		str += "Y_" + std::to_string(++index) + "=" + g_Yi.print() + ";\n";
		str2 += "Y_" + std::to_string(index) + "; ";
	}
	str2 += "];\n";
	fs.open("labY1.m", std::ios_base::out);
	fs << str << std::endl << str2;
	fs.close();

#endif
}

#endif

void test_gpu_mat()
{
#ifdef _USE_GPU
	internal_test_gpu();
#endif
}

