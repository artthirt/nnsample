#include "mainwindow.h"
#include <QApplication>
#include <QDir>

#include "shared_memory.h"
#include "custom_types.h"
#include "nn.h"

#include "tests.h"

int main(int argc, char *argv[])
{
	QString progpath = argv[0];
	QDir dir;
	dir.setPath(progpath);
	dir.cd("../");
	QDir::current().setCurrent(dir.canonicalPath());

	test_shared();
	test_mat();
	test_gpu_mat();

	QApplication a(argc, argv);
	MainWindow w;
	w.show();

	return a.exec();
}
