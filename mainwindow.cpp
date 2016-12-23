#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <math.h>
#include <random>

MainWindow::MainWindow(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::MainWindow)
{
	ui->setupUi(this);

	connect(&m_timer, SIGNAL(timeout()), this, SLOT(onTimeout()));
	m_timer.start(30);

	const int cnt = 3000;
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
	m_nn.init_weights(13);

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

	ui->dsb_alpha->setValue(m_nn.alpha());

	ui->widgetScene->add_graphic(pts, ct::Vec3d(0.7, 0, 0));
}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::on_pb_calculate_clicked()
{
	ct::Matd Y = m_nn.forward(m_X);

	std::vector< ct::Vec3d > pts;
	for(int i = 0; i < m_X.rows; i++){
		double x = m_X.at(i, 0);
		double y = m_X.at(i, 1);
		double z = Y.at(i, 0);
		pts.push_back(ct::Vec3d(x, y, z));
	}
	if(ui->widgetScene->count() < 2){
		ui->widgetScene->add_graphic(pts, ct::Vec3d(0, 0.7, 0));
		ui->widgetScene->add_graphicLines(ui->widgetScene->pts(0), pts, ct::Vec3d(0.8, 0.7, 0.3));
	}else{
		ui->widgetScene->pts(1) = pts;
		ui->widgetScene->pts2Line(0) = pts;
	}

	ui->widgetScene->set_update();
	qDebug("L2=%f", m_nn.L2());

	ui->lb_L2norm->setText(QString("L2=%1").arg(m_nn.L2(), 0, 'f', 6));

	std::string sw1 = m_nn.w1();
	std::string sw2 = m_nn.w2();
	std::string sw3 = m_nn.w3();
	std::string sw4 = m_nn.w4();

	std::string sb1 = m_nn.b1().t();
	std::string sb2 = m_nn.b2().t();
	std::string sb3 = m_nn.b3().t();
	std::string sb4 = m_nn.b4().t();

	QString sout;

	sout += QString("-----W3-------\n") + sw4.c_str();
	sout += "\n";
	sout +=QString( "-----b3-------\n") + sb4.c_str();
	sout += "\n";
	sout += QString("-----W3-------\n") + sw3.c_str();
	sout += "\n";
	sout +=QString( "-----b3-------\n") + sb3.c_str();
	sout += "\n";
	sout += QString("-----W2-------\n") + sw2.c_str();
	sout += "\n";
	sout += QString("-----b2-------\n") + sb2.c_str();
	sout += "\n";
	sout += QString("-----W1-------\n") + sw1.c_str();
	sout += "\n";
	sout += QString("-----b1-------\n") + sb1.c_str();
	sout += "\n";

	ui->pte_out->setPlainText(sout);

	m_nn.pass();
}

void MainWindow::onTimeout()
{
	if(ui->chb_auto->isChecked()){
		on_pb_calculate_clicked();
	}
}

void MainWindow::on_dsb_alpha_valueChanged(double arg1)
{
	m_nn.setAlpha(arg1);
}
