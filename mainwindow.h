#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>

#include "custom_types.h"
#include "nnmodel.h"
#include "mnist_reader.h"
#include "mnist_train.h"

#include <QMutex>
#include <QRunnable>

namespace Ui {
class MainWindow;
}

class PassModel: public QRunnable{
public:
	PassModel(nnmodel* model, int batch){
		this->model = model;
		this->use = false;
		this->done = false;
		this->count_batch = batch;
		this->lock = false;
		this->waiting = false;
	}

	bool setRequestLock();
	void setRequestUnlock();

	nnmodel* model;
	QMutex mutex;

	int count_batch;
	bool use;
	bool done;
	bool lock;
	bool waiting;

protected:
	virtual void run();
};

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	explicit MainWindow(QWidget *parent = 0);
	~MainWindow();

private slots:
	void on_pb_calculate_clicked();
	void onTimeout();
	void onTimeoutMnist();
	void onTimeoutPretrain();

	void on_dsb_alpha_valueChanged(double arg1);

	void on_pb_load_clicked();

	void on_pb_next_clicked();

	void on_pb_toBegin_clicked();

	void on_chb_auto_clicked(bool checked);

	void on_pb_pass_clicked(bool checked);

	void on_pb_test_clicked();

	void on_pb_changemodeMnist_clicked(bool checked);

	void on_pb_pretrain_clicked(bool checked);

	void on_pb_save_clicked();

	void on_pb_passGPU_clicked();

	void on_pb_pass_gpu_clicked(bool checked);

private:
	Ui::MainWindow *ui;
	QTimer m_timer;
	QTimer m_timer_mnist;
	QTimer m_timer_pretraint;

	ct::Matd m_X;
	ct::Matd m_X_val;
	ct::Matd m_y;

	bool m_use_gpu;

	PassModel* m_runmodel;

	std::uniform_real_distribution<double> ud;
	std::mt19937 gen;

	nnmodel m_nn;
	mnist_train m_mnist_train;

	mnist_reader m_mnist;

	void update_scene();
	void update_mnist();
};

#endif // MAINWINDOW_H
