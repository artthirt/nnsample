#include "mainwindow.h"
#include <QApplication>
#include <QDir>

#include "shared_memory.h"
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
//	std::cout << caption << " time(ms): " << tcc;	\
//	std::cout << endl << s.c_str();					\
//	std::cout << endl;								\
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
		(*t)[11] = e + 1;
		CHECK_VALUE(u.ref() == 2, "u ref need 2");
	}

	CHECK_VALUE(u.ref() == 1, "u ref need 1");
	CHECK_VALUE(data.ref() == 1, "data ref need 1");

	(*data)[0] = 1;
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

	float dW1[] = {1, 1, 1,
				 1, 1, 1,
				 1, 1, 1};
	float dW2[] = {2, 2, 2,
				 2, 2, 2,
				 2, 2, 2};

	int ww = 20, hh = 16;
	ct::Matf im = ct::Matf::zeros(hh, ww), ims, W1(3, 3, dW1), W2(3, 3, dW2);

#if 0
	im.randn(0, 10);
#else
	for(int i = 0; i < im.rows; ++i){
		for(int j = 0; j < im.cols; ++j){
			im.at(i, j) = i + j;
		}
	}
#endif
	ims = ct::Matf(1, im.total(), im.ptr());

	std::vector< ct::Matf > vW, vcn, gradW;
	std::vector< float > gradB;
	vW.push_back(W1);
	vW.push_back(W2);

	std::vector<float> b;
	b.push_back(0);
	b.push_back(0);

	//ct::Size sz = nn::conv2DW3x3(ims, 35, 21, 1, vW, vcn);
	ct::Size sz;

	CALC_MAT(sz = nn::conv2D(ims, ct::Size(ww, hh), 1, vW, b, vcn, nn::linear_func<float>), im, "IMAGE", 10);

//	PRINT_MAT(im, "IMAGE");

	qDebug("CONV:\n");

#define PRINT_IMAGE(mat, width, height)	{							\
	float* dcn = mat.ptr();										\
	for(int i = 0; i < height; ++i){								\
		QString s;													\
		for(int j = 0; j < width; ++j){							\
			s += QString::number(dcn[i * width + j]) + "\t";		\
		}															\
		qDebug("%s", s.toStdString().c_str());						\
	}																\
	qDebug("");														\
}

	qDebug("LAYERS");
	PRINT_IMAGE(vcn[0], sz.width, sz.height);
	PRINT_IMAGE(vcn[1], sz.width, sz.height);

	ct::Matf pool;
	ct::Mati indexes;

	nn::max_pool(vcn, pool, indexes);

	qDebug("MAXPOOL");
	PRINT_IMAGE(pool, sz.width, sz.height);

	nn::deriv_conv2D(ims, pool, indexes, ct::Size(ww, hh), sz, vW[0].size(), vW.size(), 1, gradW, gradB);

	for(int i = 0; i < gradW.size(); ++i){
		PRINT_MAT(gradW[i], QString("gW[%1]").arg(i).toStdString().c_str());
	}

	qDebug("END TEST");
}

int main(int argc, char *argv[])
{
	QString progpath = argv[0];
	QDir dir;
	dir.setPath(progpath);
	dir.cd("../");
	QDir::current().setCurrent(dir.canonicalPath());

	test_shared();
	test_mat();

	QApplication a(argc, argv);
	MainWindow w;
	w.show();

	return a.exec();
}
