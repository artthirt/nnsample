#include "widgetmnist.h"
#include "ui_widgetmnist.h"

#include <QPainter>
#include <QPaintEvent>
#include <QImage>
#include <QVariant>
#include <QMap>

const int wim		= 28;
const int him		= 28;

////////////////////////

WidgetMNIST::WidgetMNIST(QWidget *parent) :
	QWidget(parent),
	ui(new Ui::WidgetMNIST)
{
	ui->setupUi(this);

	m_index = 0;
	m_mnist = 0;
	m_mode = TRAIN;
}

WidgetMNIST::~WidgetMNIST()
{
	delete ui;
}

void WidgetMNIST::setTestMode()
{
	m_mode = TEST;
	update();
}

void WidgetMNIST::setTrainMode()
{
	m_mode = TRAIN;
	update();
}

int WidgetMNIST::mode() const
{
	return m_mode;
}

void WidgetMNIST::setMnist(mnist_reader *mnist)
{
	m_mnist = mnist;
}

uint WidgetMNIST::index() const
{
	return m_index;
}

void WidgetMNIST::updatePredictfromIndex(uint index, const QVector<uchar> &predict)
{
	if(!m_mnist || !m_mnist->train().size() || !m_mnist->test().size())
		return;

	QVector< uchar >& prediction = m_mode == TRAIN? m_prediction_train : m_prediction_test;

	if(!prediction.size()){
		prediction.resize(m_mode == TRAIN? m_mnist->count_train_images() : m_mnist->count_test_images());
	}
	for(int i = 0; i < predict.size(); i++){
		prediction[index + i] = predict[i];
	}
	update();
}

void WidgetMNIST::next()
{
	if(!m_mnist)
		return;
	if(m_index < m_mnist->count_train_images())
		m_index++;
	update();
}

void WidgetMNIST::toBegin()
{
	m_index = 0;
	update();
}

void WidgetMNIST::paintEvent(QPaintEvent *event)
{
	QPainter painter(this);

	if(!m_mnist)
		return;

	painter.setBrush(Qt::NoBrush);

	QVector< QByteArray >& data = m_mode == TRAIN? m_mnist->train() : m_mnist->test();
	QVector< uchar > &lb_data = m_mode == TRAIN? m_mnist->lb_train() : m_mnist->lb_test();
	QVector< uchar >& prediction = m_mode == TRAIN? m_prediction_train : m_prediction_test;

	int x = 0, y = 0;
	for(int i = m_index; i < data.size() && y * him + him < height(); i++){
		QImage im((uchar*)data[i].data(), wim, him, QImage::Format_Grayscale8);

		painter.setPen(Qt::red);
		painter.drawImage(x * wim, y * him, im);
		painter.drawRect(x * wim, y * him, wim, him);

		if(!lb_data.empty() && lb_data.size() == data.size()){
			painter.setPen(Qt::green);
			QString text = QString::number((uint)lb_data[i]);
			painter.drawText(x * wim + 3, y * him + 12, text);
		}
		if(prediction.size()){
			painter.setPen(QColor(30, 255, 100));
			QString text = QString::number((uint)prediction[i]);
			painter.drawText(x * wim + 17, y * him + 12, text);

			if(lb_data.size()){
				if(lb_data[i] != prediction[i]){
					QPen pen;
					pen.setColor(Qt::yellow);
					pen.setWidth(2);
					painter.setBrush(Qt::NoBrush);
					painter.setPen(pen);
					painter.drawRect(x * wim + 2, y * him + 2, wim - 4, him - 4);
				}
			}
		}

		x++;

		if(i > 0 && ((x + 1) * wim > width())){
			x = 0;
			y++;
		}
	}
}
