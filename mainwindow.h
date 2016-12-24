#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>

#include "custom_types.h"
#include "nnmodel.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	explicit MainWindow(QWidget *parent = 0);
	~MainWindow();

private slots:
	void on_pb_calculate_clicked();
	void onTimeout();

	void on_dsb_alpha_valueChanged(double arg1);

private:
	uint m_iteration;
	Ui::MainWindow *ui;
	QTimer m_timer;

	ct::Matd m_X;
	ct::Matd m_X_val;
	ct::Matd m_y;

	std::uniform_real_distribution<double> ud;
	std::mt19937 gen;

	nnmodel m_nn;
};

#endif // MAINWINDOW_H
