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
	explicit WidgetMNIST(QWidget *parent = 0);
	~WidgetMNIST();

	void setMnist(mnist_reader* mnist);

	void next();
	void toBegin();

private:
	Ui::WidgetMNIST *ui;

	mnist_reader* m_mnist;

	uint m_index;

	// QWidget interface
protected:
	void paintEvent(QPaintEvent *event);
};

#endif // WIDGETMNIST_H
