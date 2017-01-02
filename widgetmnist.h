#ifndef WIDGETMNIST_H
#define WIDGETMNIST_H

#include <QWidget>
#include <QFile>
#include <QByteArray>
#include <QVector>

namespace Ui {
class WidgetMNIST;
}

class WidgetMNIST : public QWidget
{
	Q_OBJECT

public:
	explicit WidgetMNIST(QWidget *parent = 0);
	~WidgetMNIST();

	void load();
	void next();
	void toBegin();

private:
	Ui::WidgetMNIST *ui;

	QVector< QByteArray > m_mnist_train;
	QVector< uchar > m_mnist_labels_train;

	QVector< QByteArray > m_mnist_test;
	QVector< uchar > m_mnist_labels_test;

	uint m_index;
	uint m_count_train_images;
	uint m_count_test_images;

	uint readMnist(const QString &fn, QVector< QByteArray >& mnist);
	uint readMnist(QFile &file, QVector< QByteArray >& mnist);

	uint readMnistLabels(const QString &fn, QVector< uchar >& labels);
	uint readMnistLabels(QFile& file, QVector< uchar >& labels);

	void load_params();
	void save_params();

	// QWidget interface
protected:
	void paintEvent(QPaintEvent *event);
};

#endif // WIDGETMNIST_H
