#include "widgetmnist.h"
#include "ui_widgetmnist.h"

#include <QPainter>
#include <QPaintEvent>
#include <QImage>
#include <QVariant>
#include <QMap>

#include "simple_xml.hpp"

const QString config_mnist("config.xml.mnist");

const int wim		= 28;
const int him		= 28;

WidgetMNIST::WidgetMNIST(QWidget *parent) :
	QWidget(parent),
	ui(new Ui::WidgetMNIST)
{
	ui->setupUi(this);

	m_index = 0;
	m_count_images = 0;

	load_params();
}

WidgetMNIST::~WidgetMNIST()
{
	save_params();

	delete ui;
}

void WidgetMNIST::loadLabels(const QString &labels)
{
	if(!QFile::exists(labels))
		return;

	m_labelsFile = labels;

	readMnistLabels();
	update();
}

void WidgetMNIST::load(const QString &fileName)
{
	if(!QFile::exists(fileName))
		return;

	m_fileName = fileName;

	m_count_images = readMnist();
	update();
}

void WidgetMNIST::next()
{
	if(m_index < m_count_images)
		m_index++;
	update();
}

void WidgetMNIST::toBegin()
{
	m_index = 0;
	update();
}

uint WidgetMNIST::readMnist()
{
	QFile f(m_fileName);
	if(f.open(QIODevice::ReadOnly)){
		uint cnt = readMnist(f);
		f.close();
		return cnt;
	}
	return 0;
}

inline uint invert(uint val)
{
	return (val >> 24) | (((val >> 16) & 0xFF) << 8) |
			(((val >> 8) & 0xFF) << 16) | (val << 24);
}

uint WidgetMNIST::readMnist(QFile &file)
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

	int size = wim * him;
	QByteArray data;

	m_mnist.clear();

	for(uint i = 0; i < count_images && file.pos() < file.size(); i++){
		data = file.read(size);
		m_mnist.push_back(data);
	}
	return count_images;
}

uint WidgetMNIST::readMnistLabels()
{
	QFile f(m_labelsFile);
	if(f.open(QIODevice::ReadOnly)){
		uint cnt = readMnistLabels(f);
		f.close();
		return cnt;
	}
	return 0;
}

uint WidgetMNIST::readMnistLabels(QFile &file)
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

	m_mnist_labels.clear();

	for(uint i = 0; i < count_images && file.pos() < file.size(); i++){
		data = file.read(count_images);
	}
	for(int i = 0; i < data.size(); i++){
		m_mnist_labels.push_back(data[i]);
	}
	return count_images;
}

void WidgetMNIST::load_params()
{
	QMap<QString, QVariant> params;

	if(!SimpleXML::load_param(config_mnist, params))
		return;

	m_fileName = params["file_image"].toString();

	if(!params["file_labels"].isNull()){
		m_labelsFile = params["file_labels"].toString();
		readMnistLabels();
	}

	m_count_images = readMnist();
}

void WidgetMNIST::save_params()
{
	QMap<QString, QVariant> params;

	params["file_image"] = m_fileName;
	params["file_labels"] = m_labelsFile;

	SimpleXML::save_param(config_mnist, params);
}

void WidgetMNIST::paintEvent(QPaintEvent *event)
{
	QPainter painter(this);

	painter.setBrush(Qt::NoBrush);

	int x = 0, y = 0;
	for(int i = m_index; i < m_mnist.size() && y * him + him < height(); i++){
		QImage im((uchar*)m_mnist[i].data(), wim, him, QImage::Format_Grayscale8);

		painter.setPen(Qt::red);
		painter.drawImage(x * wim, y * him, im);
		painter.drawRect(x * wim, y * him, wim, him);

		if(!m_mnist_labels.empty() && m_mnist_labels.size() == m_mnist.size()){
			painter.setPen(Qt::green);
			QString text = QString::number((uint)m_mnist_labels[i]);
			painter.drawText(x * wim + 3, y * him + 12, text);
		}

		x++;

		if(i > 0 && ((x + 1) * wim > width())){
			x = 0;
			y++;
		}
	}
}
