#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

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

private:
	Ui::MainWindow *ui;

	ct::Matd m_X;
	ct::Matd m_y;

	nnmodel m_nn;
};

#endif // MAINWINDOW_H
