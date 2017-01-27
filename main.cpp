#include "mainwindow.h"
#include <QApplication>
#include <QDir>

#include "shared_memory.h"
#include "custom_types.h"

typedef std::vector< unsigned char > vchar;

#define CHECK_VALUE(val, str) if(!(val)) std::cout << str << std::endl

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
