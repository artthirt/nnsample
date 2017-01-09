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
}

WidgetMNIST::~WidgetMNIST()
{
	delete ui;
}

void WidgetMNIST::setMnist(mnist_reader *mnist)
{
	m_mnist = mnist;
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

	int x = 0, y = 0;
	for(int i = m_index; i < m_mnist->train().size() && y * him + him < height(); i++){
		QImage im((uchar*)m_mnist->train()[i].data(), wim, him, QImage::Format_Grayscale8);

		painter.setPen(Qt::red);
		painter.drawImage(x * wim, y * him, im);
		painter.drawRect(x * wim, y * him, wim, him);

		if(!m_mnist->lb_train().empty() && m_mnist->lb_train().size() == m_mnist->train().size()){
			painter.setPen(Qt::green);
			QString text = QString::number((uint)m_mnist->lb_train()[i]);
			painter.drawText(x * wim + 3, y * him + 12, text);
		}

		x++;

		if(i > 0 && ((x + 1) * wim > width())){
			x = 0;
			y++;
		}
	}
}
