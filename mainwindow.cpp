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
	m_timer.start(1);

	const int cnt = 7000;
	const int cnt_val = 700;
	const int cnt2 = 5;
	m_X = ct::Matd(cnt, 2);
	m_X_val = ct::Matd(cnt_val, 2);
	m_y = ct::Matd(cnt, 1);

	ud = std::uniform_real_distribution<double>(-10, 10);

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
		double z =  e
				+ k1 * sin(0.23 * x) + k2 * cos(0.17 * w1) + k3 * sin(0.15 * w2);

		for(size_t k = 0; k < rand_pts.size(); k++){
			double s = (rand_pts[k] - ct::Vec2d(x, y)).norm();
			z += 3 / (1 + 0.1 * s * s);
		}

		pts.push_back(ct::Vec3d(x, y, z));

		m_X.at(i, 0) = x;
		m_X.at(i, 1) = y;
		m_y.at(i, 0) = z;
	}

	for(int i = 0; i < cnt_val; i++){
		double x = ud(gen);
		double y = ud(gen);
		m_X_val.at(i, 0) = x;
		m_X_val.at(i, 1) = y;
	}

	m_nn.setData(m_X, m_y);
	//m_nn.init_weights(13);
	std::vector<int> layers;
	layers.push_back(5);
	layers.push_back(16);
	layers.push_back(21);
	layers.push_back(28);
	layers.push_back(9);
	layers.push_back(1);
	m_nn.setLayers(layers);
	m_nn.init_model(0);

	m_iteration = 0;

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
	m_nn.pass_batch_model(150);
	m_iteration++;
	update_scene();
}

void MainWindow::onTimeout()
{
	if(ui->chb_auto->isChecked()){
		if((m_iteration % 20) == 0){
			update_scene();
		}

		m_nn.pass_batch_model(150);
		m_iteration++;
	}
}

void MainWindow::on_dsb_alpha_valueChanged(double arg1)
{
	m_nn.setAlpha(arg1);
}

void MainWindow::update_scene()
{
	ct::Matd Y = m_nn.forward_model(m_X), y_validate;

	y_validate = m_nn.forward_model(m_X_val);

	std::vector< ct::Vec3d > pts, pts_val;
	for(int i = 0; i < m_X.rows; i++){
		double x = m_X.at(i, 0);
		double y = m_X.at(i, 1);
		double z = Y.at(i, 0);
		pts.push_back(ct::Vec3d(x, y, z));
	}
	for(int i = 0; i < m_X_val.rows; i++){
		double x = m_X_val.at(i, 0);
		double y = m_X_val.at(i, 1);
		double z = y_validate.at(i, 0);
		pts_val.push_back(ct::Vec3d(x, y, z));
	}
	if(ui->widgetScene->count() < 2){
		ui->widgetScene->add_graphic(pts, ct::Vec3d(0, 0.7, 0));
		ui->widgetScene->add_graphic(pts_val, ct::Vec3d(0, 0.1, 0.7));
		ui->widgetScene->add_graphicLines(ui->widgetScene->pts(0), pts, ct::Vec3d(0.8, 0.7, 0.3));
	}else{
		ui->widgetScene->pts(1) = pts;
		ui->widgetScene->pts(2) = pts_val;
		ui->widgetScene->pts2Line(0) = pts;
	}

	ui->widgetScene->set_update();

	double L2 = m_nn.L2();

	qDebug("L2=%f", L2);

	ui->lb_L2norm->setText(QString("L2=%1;\tIteration=%2").arg(L2, 0, 'f', 6).arg(m_iteration));

	QString sout;
	for(int i = m_nn.count() - 1; i >= 0; --i){
		std::string sw = m_nn.w(i);
		std::string sb = m_nn.b(i).t();

		sout += QString("-----W%1-------\n").arg(i + 1) + sw.c_str();
		sout += "\n";
		sout += QString("-----b%1-------\n").arg(i + 1) + sb.c_str();
		sout += "\n";
	}

	ui->pte_out->setPlainText(sout);
}
