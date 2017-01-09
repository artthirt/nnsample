#ifndef WIDGETMNIST_H
#define WIDGETMNIST_H

#include <QWidget>
#include <QFile>
#include <QByteArray>
#include <QVector>

#include "mnist_reader.h"

namespace Ui {
class WidgetMNIST;
}

class WidgetMNIST : public QWidget
{
	Q_OBJECT

public:
	enum {TEST, TRAIN};

	explicit WidgetMNIST(QWidget *parent = 0);
	~WidgetMNIST();

	void setTestMode();
	void setTrainMode();

	int mode() const;

	void setMnist(mnist_reader* mnist);

	uint index() const;

	void updatePredictfromIndex(uint index, const QVector<uchar>& predict);

	void next();
	void toBegin();

private:
	Ui::WidgetMNIST *ui;
	int m_mode;

	QVector< uchar > m_prediction_test;
	QVector< uchar > m_prediction_train;

	mnist_reader* m_mnist;

	uint m_index;

	// QWidget interface
protected:
	void paintEvent(QPaintEvent *event);
};

#endif // WIDGETMNIST_H
