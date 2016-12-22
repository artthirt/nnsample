#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <math.h>
#include <random>

MainWindow::MainWindow(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::MainWindow)
{
	ui->setupUi(this);

	const int cnt = 1000;
	const int cnt2 = 5;
	m_X = ct::Matd(cnt, 2);
	m_y = ct::Matd(cnt, 1);

	std::uniform_real_distribution<double> ud(-10, 10);
	std::mt19937 gen;
	gen.seed(0);

	std::vector< ct::Vec3d > pts;

	const double k1 = 0.53;
	const double k2 = 0.17;
	const double k3 = 0.42;

	std::vector< ct::Vec2d > rand_pts;
	for(size_t i = 0; i < cnt2; i++){
		rand_pts.push_back(ct::Vec2d(ud(gen), ud(gen)));
	}

	for(int i = 0; i < cnt; i++){
		double x = ud(gen);
		double y = ud(gen);
		double w1 = x * y;
		double w2 = x + y;
		double e = exp(1.1 * x + x * x + y * y + 1.2 * y + 2.7 * w1 + 2.1 * w2);
		e = e / (1 + e);
		double z =
				+ k1 * sin(x) + k2 * cos(w1) + k3 * sin(w2);

		for(size_t k = 0; k < rand_pts.size(); k++){
			double s = (rand_pts[k] - ct::Vec2d(x, y)).norm();
			z += 3 / (1 + 0.1 * s * s);
		}

		pts.push_back(ct::Vec3d(x, y, z));

		m_X.at(i, 0) = x;
		m_X.at(i, 1) = y;
		m_y.at(i, 0) = z;
	}

	m_nn.setData(m_X, m_y);
	m_nn.init_weights();

	ui->widgetScene->add_graphic(pts, ct::Vec3d(1, 0, 0));
}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::on_pb_calculate_clicked()
{
	m_nn.pass();

	ct::Matd Y = m_nn.forward(m_X);

	std::vector< ct::Vec3d > pts;
	for(int i = 0; i < m_X.rows; i++){
		double x = m_X.at(i, 0);
		double y = m_X.at(i, 1);
		double z = Y.at(i, 0);
		pts.push_back(ct::Vec3d(x, y, z));
	}
	if(ui->widgetScene->count() < 2){
		ui->widgetScene->add_graphic(pts, ct::Vec3d(0, 1, 0));
	}else{
		ui->widgetScene->pts(1) = pts;
	}
}
