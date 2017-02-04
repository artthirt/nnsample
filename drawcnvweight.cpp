#include "drawcnvweight.h"
#include "ui_drawcnvweight.h"

#include <QPaintEvent>
#include <QPainter>

DrawCnvWeight::DrawCnvWeight(QWidget *parent) :
	QWidget(parent),
	ui(new Ui::DrawCnvWeight)
{
	ui->setupUi(this);
}

DrawCnvWeight::~DrawCnvWeight()
{
	delete ui;
}

void copy_weights(const std::vector<std::vector<ct::Matf> > &W,
				  std::vector<std::vector<ct::Matf> > &Wout)
{
	Wout.clear();
	Wout.resize(W.size());
	for(size_t i = 0; i < W.size(); i++){
		Wout[i].resize(W[i].size());
		for(size_t j = 0; j < W[i].size(); j++){
			W[i][j].copyTo(Wout[i][j]);
		}
	}
}

void DrawCnvWeight::set_weight(const std::vector<std::vector<ct::Matf> > &W)
{
	if(W.empty())
		return;

	copy_weights(W, m_W);

	if(m_prevW.empty())
		copy_weights(m_W, m_prevW);

	if(m_firstW.empty()){
		copy_weights(m_W, m_firstW);
	}

	update();
}

void DrawCnvWeight::set_prev_weight(const std::vector<std::vector<ct::Matf> > &W)
{
	copy_weights(W, m_prevW);
}

void draw_W(QPainter& painter, const ct::Matf& W, const ct::Matf& prevW, int _x, int _y, int w, float _max, float _min, bool is_prev)
{
	float m1 = _max;
	float m2 = _min;
	ct::Matf _W = W - m2;
	_W *= (1./(m1 - m2));
	_W *= 255.;

	float *dW1 = W.ptr();
	float *dW2 = prevW.ptr();

	QPen pen;
	pen.setWidth(2);

	float *dW = _W.ptr();
	for(int y = 0; y < _W.rows; ++y){
		for(int x = 0; x < _W.cols; ++x){
			QRect rt(QPoint(_x + x * w, _y + y * w), QSize(w, w));
			uchar c = dW[y * _W.cols + x];
			painter.setBrush(QColor(c, c, c));
			painter.drawRect(rt);

			float v1 = dW1[y * W.cols + x];
			float v2 = dW2[y * W.cols + x];

			if(is_prev && std::abs(v2 - v1) > 1e-9){
				float val = 100 + 2000 * std::abs((v2 - v1)/(m1 - m2));
				val = std::min(255.f, val);
				pen.setColor(QColor(val, 0, 0));
				painter.setPen(pen);
				painter.setBrush(Qt::NoBrush);
				rt.marginsRemoved(QMargins(1, 1, 1, 1));
				painter.drawRect(rt);
			}
		}
	}
}

void DrawCnvWeight::paintEvent(QPaintEvent *event)
{
	Q_UNUSED(event);

	QPainter painter(this);

	painter.fillRect(rect(), Qt::black);

	/*QSize s =*/
	draw_weight(painter, 0, m_W, true);
//	draw_weight(painter, s.height() + 20, m_firstW, false);
}

QSize DrawCnvWeight::draw_weight(QPainter &painter, int offset, const std::vector<std::vector<ct::Matf> > &Weights, bool is_prev)
{
	if(m_W.empty())
		return QSize(0, 0);

	int wd_blk = 20;

	int x = 0, y = offset, w = 0, h = 0;

	for(size_t i = 0; i < Weights.size(); ++i){
		const std::vector< ct::Matf > &Ws = Weights[i];
		x = 0;

		float _max = -99999999.f, _min = 999999999.f;

		for(size_t j = 0; j < Ws.size(); ++j){
			float m1 = Ws[j].max();
			float m2 = Ws[j].min();

			_max = std::max(m1, _max);
			_min = std::min(m2, _min);
		}

		for(size_t j = 0; j < Ws.size(); ++j){
			ct::Matf W = Ws[j];

			w = wd_blk * W.cols;
			h = wd_blk * W.rows;

			if(x >= rect().width() - (w + 2)){
				x = 0;
				y += h + 2;
			}

			painter.setPen(Qt::NoPen);
			draw_W(painter, W, m_prevW[i][j], x, y, wd_blk, _max, _min, is_prev);
			painter.setPen(Qt::blue);
			painter.setBrush(Qt::NoBrush);
			painter.drawRect(QRect(QPoint(x, y), QSize(w, h)));

			x += w + 2;
		}
		y += h + 2;
	}
	return QSize(x, y - offset);
}
