#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QDebug>

#include <math.h>
#include <random>
#include <QThreadPool>

///////////////////////////////

bool PassModel::setRequestLock()
{
	lock = true;
	return mutex.tryLock(10);
}

void PassModel::setRequestUnlock()
{
	mutex.unlock();
	lock = false;
}

void PassModel::run()
{
	while(!done){
		if(use && model && !lock){
			waiting = false;
			mutex.lock();

			model->pass_batch_model(count_batch);

			mutex.unlock();
		}else{
			waiting = true;
			QThread::currentThread()->msleep(5);
		}
	}
}

//////////////////////////////

MainWindow::MainWindow(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::MainWindow)
{
	ui->setupUi(this);

	connect(&m_timer, SIGNAL(timeout()), this, SLOT(onTimeout()));
	m_timer.start(1000);

	connect(&m_timer_mnist, SIGNAL(timeout()), this, SLOT(onTimeoutMnist()));
	m_timer_mnist.start(1);

	connect(&m_timer_pretraint, SIGNAL(timeout()), this, SLOT(onTimeoutPretrain()));
	m_timer_pretraint.start(10);

	const int cnt = 27000;
	const int cnt_val = 700;
	const int cnt2 = 5;
	m_X = ct::Matf(cnt, 2);
	m_X_val = ct::Matf(cnt_val, 2);
	m_y = ct::Matf(cnt, 1);

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
	layers.push_back(30);
	layers.push_back(60);
	layers.push_back(10);
//	layers.push_back(5);
//	layers.push_back(10);
	layers.push_back(1);
	m_nn.setLayers(layers);
	m_nn.init_model(0);

	m_runmodel = new PassModel(&m_nn, 200);
	QThreadPool::globalInstance()->start(m_runmodel);

	ui->dsb_alpha->setValue(m_nn.alpha());

	ui->widgetScene->add_graphic(pts, ct::Vec3d(0.7, 0, 0));

	m_mnist.load();
	m_mnist.init_train();
	ui->widgetMNIST->setMnist(&m_mnist);
	ui->widgetMNIST->update();

	std::vector<int> layers2, layers3, cnv_layers, ws;
//	layers2.push_back(600);
	layers2.push_back(600);
	layers2.push_back(10);

	m_mnist_train.setLayers(layers2);
	m_mnist_train.setMnist(&m_mnist);
	m_mnist_train.init_weights(time(0));

	layers3.push_back(500);
	layers3.push_back(10);

//	ws.push_back(5);

	ui->widgetMNISTCnv->setMnist(&m_mnist);
	ui->widgetMNISTCnv->update();

	m_mnist_cnv.setConvLength();
	m_mnist_cnv.setLayers(layers3);
	m_mnist_cnv.setMnist(&m_mnist);
	m_mnist_cnv.init(1);

//	m_mnist_cnv.load_model(true);
//	m_mnist_cnv.load_model();

	ui->sb_timeout_train->setValue(m_timer_mnist.interval());
}

MainWindow::~MainWindow()
{
	delete ui;

	if(m_runmodel){
		m_runmodel->done = true;
	}
	QThreadPool::globalInstance()->waitForDone();
}

void MainWindow::on_pb_calculate_clicked()
{
	if(ui->twtabs->currentIndex() == 0){
		if(m_runmodel->setRequestLock()){
			m_nn.pass_batch_model(150);
			update_scene();
			m_runmodel->setRequestUnlock();
		}
	}else if(ui->twtabs->currentIndex() == 2){
		m_mnist_train.pass_batch(300);
		update_mnist();
	}
}

void MainWindow::onTimeout()
{
	if(ui->chb_auto->isChecked()){

		qDebug() << "to lock";
		if(m_runmodel->setRequestLock()){
			qDebug() << "in lock";
			update_scene();
			m_runmodel->setRequestUnlock();
			qDebug() << "out lock";
		}
	}
}

void MainWindow::onTimeoutMnist()
{
	if(ui->pb_pass->isChecked()){
#ifdef _USE_GPU
		if(m_use_gpu){
			m_mnist_train.pass_batch_gpu(400);
		}else{
			m_mnist_train.pass_batch(100);
		}
#else
		m_mnist_train.pass_batch(100);
#endif
		qDebug() << "Iteration" << m_mnist_train.iteration();
		ui->lb_out->setText("Pass: #" + QString::number(m_mnist_train.iteration()));

		if((m_mnist_train.iteration() % 40) == 0){
			update_mnist();
		}
	}

	if(ui->pb_pass_cnv->isChecked()){
		pass_cnv();
		if((m_mnist_cnv.iteration() && m_mnist_cnv.iteration() % 40) == 0){
			on_pb_update_cnv_clicked();
		}
	}
}

void MainWindow::onTimeoutPretrain()
{
	if(ui->pb_pretrain->isChecked()){
		int batch = m_use_gpu? 500 : 500;
		double res = m_mnist_train.pass_batch_autoencoder(batch, m_use_gpu);
		ui->lb_out->setText(QString("l2_norm = %1").arg(res));
	}
}

void MainWindow::on_dsb_alpha_valueChanged(double arg1)
{
	m_nn.setAlpha(arg1);
	m_mnist_train.setAlpha(arg1);
	m_mnist_cnv.setAlpha(arg1);
}

void MainWindow::update_scene()
{
	ct::Matf Y = m_nn.forward_model(m_X), y_validate;

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

	ui->lb_L2norm->setText(QString("L2=%1;\tIteration=%2").arg(L2, 0, 'f', 9).arg(m_nn.iteration()));

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

void MainWindow::update_mnist()
{
	double l2, accuracy;
	m_mnist_train.getEstimate(2000, l2, accuracy, m_use_gpu);
	ui->lb_ce->setText(QString("L2=%1; Acc=%2;\tIteration=%3").arg(l2, 0, 'f', 9).arg(accuracy, 0, 'f', 5).arg(m_mnist_train.iteration()));

	uint index = ui->widgetMNIST->index();

	const int cnti = 50;

	if(ui->widgetMNIST->mode() == WidgetMNIST::TRAIN){

		int count = std::min((uint)2000, m_mnist.count_train_images() - index);

		ct::Matf y;
		int index1 = index;
		int count1 = index + count;
		while(index1 < count1){
			ct::Matf yi = m_mnist_train.forward(index1, cnti, m_use_gpu);
			index1 += cnti;
			y.push_back(yi);
		}

		QVector< uchar > data;

		data.resize(count);

		for(int i = 0; i < count; i++){
			data[i] = y.argmax(i, 1);
		}
		ui->widgetMNIST->updatePredictfromIndex(index, data);
	}else{
		uint count = std::min((uint)2000, m_mnist.count_test_images() - index);

		ct::Matf y;
		int index1 = index;
		int count1 = index1 + count;
		while(index1 < count1){
			ct::Matf yi = m_mnist_train.forward_test(index1, cnti, m_use_gpu);
			index1 += cnti;
			y.push_back(yi);
		}

		QVector< uchar > data;

		data.resize(count);

		for(uint i = 0; i < count; i++){
			data[i] = y.argmax(i, 1);
		}
		ui->widgetMNIST->updatePredictfromIndex(index, data);
	}
}

void MainWindow::pass_cnv()
{
//	ui->wdg_cnvW->set_prev_weight(m_mnist_cnv.cnvW());

	m_mnist_cnv.pass_batch(ui->sb_batch->value(),
						   ui->chb_use_gpu_cnv->isChecked());
//	on_pb_update_cnv_clicked();
	ui->lb_out_cnv->setText("Pass: #" + QString::number(m_mnist_cnv.iteration()));
}

void MainWindow::on_pb_next_clicked()
{
	ui->widgetMNIST->next();
}

void MainWindow::on_pb_toBegin_clicked()
{
	ui->widgetMNIST->toBegin();
}

void MainWindow::on_chb_auto_clicked(bool checked)
{
	if(!m_runmodel)
		return;
	m_runmodel->use = checked;
}

void MainWindow::on_pb_test_clicked()
{
	double l2, accuracy;
	m_mnist_train.getEstimateTest(l2, accuracy, m_use_gpu);
	ui->lb_out->setText("L2(test)=" + QString::number(l2) + "; Acc(test)=" + QString::number(accuracy));
}

void MainWindow::on_pb_changemodeMnist_clicked(bool checked)
{
	if(checked){
		ui->widgetMNIST->setTestMode();
	}else{
		ui->widgetMNIST->setTrainMode();
	}
}

void MainWindow::on_pb_save_clicked()
{
	m_mnist_train.save("model.txt");
}

void MainWindow::on_pb_load_clicked()
{
	m_mnist_train.load("model.txt");
}

void MainWindow::on_pb_passGPU_clicked()
{
#ifdef _USE_GPU
	double l2, accuracy;

	m_mnist_train.getEstimate(3000, l2, accuracy, true);

	m_mnist_train.pass_batch_gpu(400);
	m_mnist_train.getEstimate(3000, l2, accuracy, true);
	ui->lb_out->setText("L2=" + QString::number(l2) + "; Acc=" + QString::number(accuracy));

#endif
}

void MainWindow::on_chb_usegpu_clicked(bool checked)
{
	m_use_gpu = checked;
}

void MainWindow::on_pb_copy_mats_clicked()
{
	m_mnist_train.copyWbMat2GpuMat();
}

void MainWindow::on_pb_init_weights_clicked()
{
	m_mnist_train.init_weights();
}

void MainWindow::on_pb_copy_mats_2_clicked()
{
	m_mnist_train.copyWbGpuMat2Mat();
}

void MainWindow::on_pb_save_gpu_clicked()
{
	m_mnist_train.save_gpu_matricies();
}

void MainWindow::on_pb_pass_cnv_clicked()
{
//	pass_cnv();
}

void MainWindow::on_pb_test_cnv_clicked()
{
	double l2, accuracy;
	m_mnist_cnv.getEstimateTest(l2, accuracy, ui->chb_use_gpu_cnv->isChecked());
	ui->lb_out_cnv->setText("L2(test)=" + QString::number(l2) + "; Acc(test)=" + QString::number(accuracy));
}

void MainWindow::on_pb_update_cnv_clicked()
{
	const int cnt = 50;

	ui->wdg_cnvW->set_weight(m_mnist_cnv.cnvW());

	double l2, accuracy;
	m_mnist_cnv.getEstimate(2000, l2, accuracy, ui->chb_use_gpu_cnv->isChecked());
	ui->lb_l2cnv->setText(QString("L2=%1; Acc=%2;\tIteration=%3").arg(l2, 0, 'f', 9).arg(accuracy, 0, 'f', 5).arg(m_mnist_cnv.iteration()));

	uint index = ui->widgetMNISTCnv->index();

	if(ui->widgetMNISTCnv->mode() == WidgetMNIST::TRAIN){

		int count = std::min((uint)2000, m_mnist.count_train_images() - index);

		ct::Matf y;
		int index1 = index;
		int count1 = index + count;
		while(index1 < count1){
			ct::Matf yi = m_mnist_cnv.forward(index1, cnt, ui->chb_use_gpu_cnv->isChecked());
			index1 += cnt;
			y.push_back(yi);
		}

		QVector< uchar > data;

		data.resize(count);

		for(int i = 0; i < count; i++){
			data[i] = y.argmax(i, 1);
		}
		ui->widgetMNISTCnv->updatePredictfromIndex(index, data);
	}else{
		int count = std::min((uint)2000, m_mnist.count_test_images() - index);

		ct::Matf y;
		int index1 = index;
		int count1 = index + count;
		while(index1 < count1){
			ct::Matf yi = m_mnist_cnv.forward_test(index1, cnt, ui->chb_use_gpu_cnv->isChecked());
			y.push_back(yi);
			index1 += cnt;
		}

		QVector< uchar > data;

		data.resize(count);

		for(int i = 0; i < count; i++){
			data[i] = y.argmax(i, 1);
		}
		ui->widgetMNISTCnv->updatePredictfromIndex(index, data);
	}

	m_mnist_cnv.save_model(ui->chb_use_gpu_cnv->isChecked());
}

void MainWindow::closeEvent(QCloseEvent *event)
{
	QMainWindow::closeEvent(event);
}

void MainWindow::on_pb_mode_cnv_clicked(bool checked)
{
	if(checked){
		ui->widgetMNISTCnv->setTestMode();
	}else{
		ui->widgetMNISTCnv->setTrainMode();
	}

}


void MainWindow::on_sb_timeout_train_valueChanged(int arg1)
{
	m_timer_mnist.setInterval(arg1);
}
