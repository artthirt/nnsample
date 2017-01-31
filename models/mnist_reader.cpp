#include "mnist_reader.h"

#include <QFile>
#include <QDir>

const QString train_images_file("data/train-images.idx3-ubyte");
const QString train_labels_file("data/train-labels.idx1-ubyte");
const QString test_images_file("data/t10k-images.idx3-ubyte");
const QString test_labels_file("data/t10k-labels.idx1-ubyte");

/////////////////////////

inline uint invert(uint val)
{
	return (val >> 24) | (((val >> 16) & 0xFF) << 8) |
			(((val >> 8) & 0xFF) << 16) | (val << 24);
}

/////////////////////////

mnist_reader::mnist_reader()
{
	m_count_train_images = 0;
	m_count_test_images = 0;

	load();
}

uint mnist_reader::count_train_images() const
{
	return m_count_train_images;
}

uint mnist_reader::count_test_images() const
{
	return m_count_test_images;
}

QVector<QByteArray> &mnist_reader::train()
{
	return m_mnist_train;
}

QVector<QByteArray> &mnist_reader::test()
{
	return m_mnist_test;
}

QVector<uchar> &mnist_reader::lb_train()
{
	return m_mnist_labels_train;
}

QVector<uchar> &mnist_reader::lb_test()
{
	return m_mnist_labels_test;
}

void mnist_reader::init_train()
{
	if(!m_mnist_train.size())
		return;

	using namespace ct;

	const int out_cols = 10;

	m_X = Matf::zeros(m_mnist_train.size(), m_mnist_train[0].size());
	m_y = Matf::zeros(lb_train().size(), out_cols);

	for(int i = 0; i < m_mnist_train.size(); i++){
		int yi = m_mnist_labels_train[i];
		m_y.at(i, yi) = 1.;

		QByteArray &data = m_mnist_train[i];
		for(int j = 0; j < data.size(); j++){
			m_X.at(i, j) = ((uint)data[j] > 0)? 1. : 0.;
		}
	}

}

ct::Matf &mnist_reader::X()
{
	return m_X;
}

ct::Matf &mnist_reader::y()
{
	return m_y;
}

void mnist_reader::load()
{
	if(QFile::exists(train_images_file)){
		m_count_train_images = readMnist(train_images_file, m_mnist_train);
	}
	if(QFile::exists(test_images_file)){
		m_count_test_images = readMnist(test_images_file, m_mnist_test);
	}
	if(QFile::exists(train_labels_file)){
		readMnistLabels(train_labels_file, m_mnist_labels_train);
	}
	if(QFile::exists(test_labels_file)){
		readMnistLabels(test_labels_file, m_mnist_labels_test);
	}
}

uint mnist_reader::readMnist(const QString& fn, QVector< QByteArray >& mnist)
{
	QFile f(fn);
	if(f.open(QIODevice::ReadOnly)){
		uint cnt = readMnist(f, mnist);
		f.close();
		return cnt;
	}
	return 0;
}

uint mnist_reader::readMnist(QFile &file, QVector<QByteArray> &mnist)
{
	if(!file.isOpen())
		return 0;

	const int magic_number = 0x00000803;
	uint count_images, rows, cols, check_number;

	file.read((char*)&check_number, sizeof(uint));

	check_number = invert(check_number);

	if(check_number != magic_number)
		return 0;

	file.read((char*)&count_images, sizeof(uint));
	file.read((char*)&rows, sizeof(uint));
	file.read((char*)&cols, sizeof(uint));

	count_images = invert(count_images);
	rows = invert(rows);
	cols = invert(cols);

	int size = WidthIM * HeightIM;
	QByteArray data;

	mnist.clear();

	for(uint i = 0; i < count_images && file.pos() < file.size(); i++){
		data = file.read(size);
		mnist.push_back(data);
	}
	return count_images;
}

uint mnist_reader::readMnistLabels(const QString& fn, QVector<uchar> &labels)
{
	QFile f(fn);
	if(f.open(QIODevice::ReadOnly)){
		uint cnt = readMnistLabels(f, labels);
		f.close();
		return cnt;
	}
	return 0;
}

uint mnist_reader::readMnistLabels(QFile &file, QVector< uchar >& labels)
{
	if(!file.isOpen())
		return 0;

	const int magic_number = 0x00000801;
	uint count_images, check_number;

	file.read((char*)&check_number, sizeof(uint));

	check_number = invert(check_number);

	if(check_number != magic_number)
		return 0;

	file.read((char*)&count_images, sizeof(uint));

	count_images = invert(count_images);

	QByteArray data;

	labels.clear();

	for(uint i = 0; i < count_images && file.pos() < file.size(); i++){
		data = file.read(count_images);
	}
	for(int i = 0; i < data.size(); i++){
		labels.push_back(data[i]);
	}
	return count_images;
}
