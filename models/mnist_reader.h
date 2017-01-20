#ifndef MNIST_READER_H
#define MNIST_READER_H

#include <QFile>
#include <QVector>
#include <QByteArray>

class mnist_reader
{
public:
	enum{WidthIM = 28, HeightIM = 28};

	mnist_reader();

	uint count_train_images() const;
	uint count_test_images() const;

	QVector<QByteArray> &train();
	QVector<QByteArray> &test();

	QVector<uchar> &lb_train();
	QVector<uchar> &lb_test();

	void load();

private:
	QVector< QByteArray > m_mnist_train;
	QVector< uchar > m_mnist_labels_train;

	QVector< QByteArray > m_mnist_test;
	QVector< uchar > m_mnist_labels_test;

	uint m_count_train_images;
	uint m_count_test_images;

	uint readMnist(const QString &fn, QVector< QByteArray >& mnist);
	uint readMnist(QFile &file, QVector< QByteArray >& mnist);

	uint readMnistLabels(const QString &fn, QVector< uchar >& labels);
	uint readMnistLabels(QFile& file, QVector< uchar >& labels);

};

#endif // MNIST_READER_H
