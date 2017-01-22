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
	/**
	 * @brief setTestMode
	 */
	void setTestMode();
	/**
	 * @brief setTrainMode
	 */
	void setTrainMode();
	/**
	 * @brief mode
	 * @return TEST or TRAIN
	 */
	int mode() const;
	/**
	 * @brief setMnist
	 * set ref to reader mnist data
	 * @param mnist
	 */
	void setMnist(mnist_reader* mnist);
	/**
	 * @brief index
	 * @return index of beginning of the representation
	 */
	uint index() const;
	/**
	 * @brief updatePredictfromIndex
	 * update predict values from index
	 * @param index
	 * @param predict - array of predicted values
	 */
	void updatePredictfromIndex(uint index, const QVector<uchar>& predict);
	/**
	 * @brief next
	 */
	void next();
	/**
	 * @brief toBegin
	 */
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
