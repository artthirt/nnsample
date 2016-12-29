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

	void loadLabels(const QString& labels);
	void load(const QString& fileName);
	void next();
	void toBegin();

private:
	Ui::WidgetMNIST *ui;

	QVector< QByteArray > m_mnist;
	QVector< uchar > m_mnist_labels;

	QString m_fileName;
	QString m_labelsFile;

	uint m_index;
	uint m_count_images;

	uint readMnist();
	uint readMnist(QFile &file);

	uint readMnistLabels();
	uint readMnistLabels(QFile& file);

	void load_params();
	void save_params();

	// QWidget interface
protected:
	void paintEvent(QPaintEvent *event);
};

#endif // WIDGETMNIST_H
