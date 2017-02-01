#ifndef DRAWCNVWEIGHT_H
#define DRAWCNVWEIGHT_H

#include <QWidget>

#include "custom_types.h"

namespace Ui {
class DrawCnvWeight;
}

class DrawCnvWeight : public QWidget
{
	Q_OBJECT

public:
	explicit DrawCnvWeight(QWidget *parent = 0);
	~DrawCnvWeight();

	void set_weight(const std::vector< std::vector < ct::Matf > > &W);
	void set_prev_weight(const std::vector< std::vector < ct::Matf > > &W);

private:
	Ui::DrawCnvWeight *ui;

	// QWidget interface
protected:
	virtual void paintEvent(QPaintEvent *event);

private:
	std::vector< std::vector < ct::Matf > > m_W, m_prevW, m_firstW;

	QSize draw_weight(QPainter& painter, int offset, const std::vector< std::vector < ct::Matf > > &Weights, bool is_prev);
};

#endif // DRAWCNVWEIGHT_H
