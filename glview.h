#ifndef GLVIEW_H
#define GLVIEW_H

#include <QWidget>
#include <QGLWidget>
#include <QPointF>
#include <QTimer>

#include "custom_types.h"

typedef ct::Vec3d Line[2];

namespace Ui {
class GLView;
}

class GLView : public QGLWidget
{
	Q_OBJECT

public:
	explicit GLView(QWidget *parent = 0);
	~GLView();

	void set_update();

	void add_graphic(const std::vector< ct::Vec3d >& pts, const ct::Vec3d& color);
	void add_graphicLines(const std::vector< ct::Vec3d >& p1,
						  const std::vector< ct::Vec3d >& p2, const ct::Vec3d& color);

	std::vector<ct::Vec3d> &pts(size_t index);
	ct::Vec3d &color(size_t index);
	size_t count() const;

	std::vector<ct::Vec3d> &pts1Line(size_t index);
	std::vector<ct::Vec3d> &pts2Line(size_t index);
	ct::Vec3d &colorLine(size_t index);
	size_t countLine() const;

private:
	Ui::GLView *ui;
	bool m_init;
	bool m_update;

	struct graphic{
		graphic(){

		}
		graphic( const std::vector< ct::Vec3d > & pts, const ct::Vec3d& color){
			this->points = pts;
			this->color = color;
		}
		graphic( const std::vector< ct::Vec3d > & p1,  const std::vector< ct::Vec3d > & p2, const ct::Vec3d& color){
			this->points = p1;
			this->points2 = p2;
			this->color = color;
		}

		std::vector < ct::Vec3d > points;
		std::vector < ct::Vec3d > points2;
		ct::Vec3d color;
	};

	std::vector< graphic > m_graphics;
	std::vector< graphic > m_graphicsLines;

	bool m_show_graphics;

	double m_prev_e_track;
	double m_prev_u;

	bool m_tracking;
	double m_tracking_angle;

	bool m_bind_rotation;

	QTimer m_timer;
	QTimer m_timer_model;

	QPointF m_mouse_pt;
	QPointF m_delta_pt;
	QPointF m_wheel_pt;
	double m_delta_z;
	double m_current_z;

	bool m_left_down;
	bool m_wheel_down;

	ct::Vec3d m_angles;

	ct::Vec3d m_color_space;

	void init();
	void draw_net();
	void draw_graphics();
	void draw_graphicsLine();

	void load_xml();
	void save_xml();

	void calculate_track();

	// QGLWidget interface
public slots:
	virtual void updateGL();
	void onTimeout();
	void onTimeoutModel();

signals:
	void push_logs(const QString& val);

protected:
	virtual void resizeGL(int w, int h);
	virtual void paintGL();
	virtual void glDraw();

	// QWidget interface
protected:
	virtual void mousePressEvent(QMouseEvent *event);
	virtual void mouseReleaseEvent(QMouseEvent *event);
	virtual void mouseMoveEvent(QMouseEvent *event);
};

#endif // GLVIEW_H
