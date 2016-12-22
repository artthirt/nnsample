#include "glview.h"
#include "ui_glview.h"

#ifdef _MSC_VER
#include <Windows.h>
#endif

#include <GL/gl.h>
#include <GL/glu.h>

#include <QMouseEvent>
#include <QDebug>

#include <simple_xml.hpp>

const QString xml_config("config.glview.xml");

typedef float GLVertex3f[3];

////////////////////////////////////

void draw_line(const ct::Vec3d& v, const ct::Vec3d & col = ct::Vec3d::ones(), double scale = 1.)
{
	glLineWidth(3);

	glPushMatrix();

	glScaled(scale, scale, scale);

	glColor3dv(col.ptr());
	glBegin(GL_LINES);
	glVertex3dv(ct::Vec3d::zeros().ptr());
	glVertex3dv(v.ptr());
	glEnd();

	glPopMatrix();

	glLineWidth(1);
}

void draw_cylinder(double R, double H, int cnt = 10, const ct::Vec4d& col = ct::Vec4d::ones())
{
	double z0 = 0;
	double z1 = H;

	glColor4dv(col.val);
	//glColor4d(0.8, 0.4, 0.1, 0.5);

	glBegin(GL_TRIANGLE_STRIP);
	for(int i = 0; i <= cnt; i++){
		double xi = sin(2. * i / cnt * M_PI);
		double yi = cos(2. * i / cnt * M_PI);
		double x0 = R * xi;
		double y0 = R * yi;

//		x1 = R * sin(2. * (i + 1) / cnt * M_PI);
//		y1 = R * cos(2. * (i + 1) / cnt * M_PI);

		glNormal3d(yi, xi, 0);
		glVertex3d(x0, y0, z0);
		glVertex3d(x0, y0, z1);
	}
	glEnd();
}

void draw_circle(const ct::Vec3d &pt, double R, const ct::Vec4d &col = ct::Vec4d::zeros())
{
	const int cnt = 32;
	ct::Vec3d z1 = pt, p1;

	glColor4dv(col.val);
	glBegin(GL_TRIANGLE_STRIP);
	for(int i = 0; i <= cnt; i++){
		glVertex3dv(z1.val);

		double id = 1. * i / cnt * 2 * M_PI;
		double x = R * sin(id);
		double y = R * cos(id);

		p1 = pt + ct::Vec3d(x, y, 0);
		glVertex3dv(p1.val);
	}
	glEnd();
}

////////////////////////////////////

GLView::GLView(QWidget *parent) :
	QGLWidget(parent),
	ui(new Ui::GLView)
  , m_init(false)
  , m_update(false)
  , m_delta_z(0)
  , m_current_z(0)
  , m_color_space(ct::Vec3d::ones())
  , m_tracking(false)
  , m_tracking_angle(0)
  , m_show_route(false)
  , m_prev_e_track(0)
  , m_prev_u(0)
  , m_timer_goal(0)
  , m_is_draw_track(false)
  , m_show_graphics(false)
  , m_bind_rotation(true)
{
	ui->setupUi(this);

	connect(&m_timer, SIGNAL(timeout()), this, SLOT(onTimeout()));
	m_timer.start(30);

	connect(&m_timer_model, SIGNAL(timeout()), this, SLOT(onTimeoutModel()));
	m_timer_model.start(100);

	load_xml();

	setMouseTracking(true);

#ifdef _MSC_VER
	QGLFormat newFormat = format();
	newFormat.setSampleBuffers(true);
	newFormat.setSamples(8);
	setFormat(newFormat);
#endif
}

GLView::~GLView()
{
	delete ui;

	save_xml();
}

void GLView::set_update()
{
	m_update = true;
}

void GLView::add_graphic(const std::vector<ct::Vec3d> &pts, const ct::Vec3d &color)
{
	m_graphics.push_back(graphic(pts, color));
}

std::vector<ct::Vec3d> &GLView::pts(size_t index)
{
	return m_graphics[index].points;
}

ct::Vec3d &GLView::color(size_t index)
{
	return m_graphics[index].color;
}

size_t GLView::count() const
{
	return m_graphics.size();
}

void GLView::init()
{
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glEnable(GL_POINT_SMOOTH);

	glFrontFace(GL_FRONT);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void GLView::draw_net()
{
	const int count = 100;

	static GLVertex3f ptLineX[count][2] = {0}, ptLineY[count][2] = {0};
	static bool isInit = false;

	if(!isInit){
		isInit = true;

		const double width = 100;
		for(int i = 0; i < count; i++){
			double it = (double)i/count;
			double x = -width/2 + width * it;

			ptLineX[i][0][0] = x;
			ptLineX[i][0][1] = -width/2;
			ptLineX[i][1][0] = x;
			ptLineX[i][1][1] = width/2;

			ptLineY[i][0][1] = x;
			ptLineY[i][0][0] = -width/2;
			ptLineY[i][1][1] = x;
			ptLineY[i][1][0] = width/2;
		}

	}

	glEnableClientState(GL_VERTEX_ARRAY);

	glColor3f(0, 0, 0);
	glVertexPointer(3, GL_FLOAT, 0, ptLineX);
	glDrawArrays(GL_LINES, 0, count * 2);

	glVertexPointer(3, GL_FLOAT, 0, ptLineY);
	glDrawArrays(GL_LINES, 0, count * 2);

	glDisableClientState(GL_VERTEX_ARRAY);
}

void GLView::draw_graphics()
{
	glEnable(GL_BLEND);
	glPointSize(3);
	for(size_t i = 0; i < m_graphics.size(); i++){
		graphic& g = m_graphics[i];

		glColor3dv(g.color.val);
		glBegin(GL_POINTS);
		for(size_t j = 0; j < g.points.size(); j++){
			glVertex3dv(g.points[j].val);
		}
		glEnd();
	}
	glDisable(GL_BLEND);
}

void GLView::load_xml()
{
	QMap< QString, QVariant > params;

	if(!SimpleXML::load_param(xml_config, params))
		return;

}

void GLView::save_xml()
{
	QMap< QString, QVariant > params;

	SimpleXML::save_param(xml_config, params);
}

void GLView::calculate_track()
{
	if(!m_bind_rotation)
		return;
//		ct::Vec3f ps = m_model.pos();
	ct::Vec3d dm = ct::Vec3d(1, 0, 0);

	const double kp = 0.1;
	const double kd = 0.3;

	dm[2] = 0;
	dm /= dm.norm();
	double angle = M_PI/2 - atan2(dm[1], dm[0]);
	double e = angle - ct::angle2rad(m_tracking_angle);
	e = atan2(sin(e), cos(e));

	double de = e - m_prev_e_track;
	m_prev_e_track = e;
	de = atan2(sin(de), cos(de));

	double u = kp * e + kd * de;
	u = ct::rad2angle(u);

	double avg_u = 0.5 * (m_prev_u + u);
	m_prev_u = avg_u;

	m_tracking_angle += avg_u;

//		gluLookAt(ps[0], ps[1], ps[2], ps[0] + dm[0], ps[1] + dm[1], ps[2] + dm[2], 0, 0, 1);
}


void GLView::updateGL()
{
}

void GLView::onTimeout()
{
	if(m_update){
		m_update = false;
		update();
	}
}

void GLView::onTimeoutModel()
{
}

void GLView::resizeGL(int w, int h)
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60., (double)w/h, 0.1, 500);
	glMatrixMode(GL_MODELVIEW);

	update();
}

void GLView::paintGL()
{
}

void GLView::glDraw()
{
	if(!m_init){
		m_init = true;
		init();
	}

	makeCurrent();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(m_color_space[0], m_color_space[1], m_color_space[2], 1);
	glLoadIdentity();

	const float poslight[] = {10, -100, 100, 1};
	const float alight[] = {0.0f, 0.0f, 0.0f, 1};
	const float slight[] = {0.0f, 0.0f, 0.0f, 1};
	const float dlight[] = {0.3f, 0.3f, 0.3f, 1};

	glPushMatrix();/////////////////

	glTranslatef(0, 0, -5);

	if(!m_tracking){
		glTranslatef(0, 0, -(m_current_z + m_delta_z));

		glRotatef(m_delta_pt.x(), 0, 1, 0);
		glRotatef(m_delta_pt.y(), 1, 0, 0);
	}else{
	}


	if(m_tracking){
		calculate_track();

		glTranslatef(0, 0, - (m_current_z + m_delta_z));
		glRotatef(m_delta_pt.y(), 1, 0, 0);
		glRotatef(m_delta_pt.x(), 0, 0, 1);
		glRotatef(m_tracking_angle, 0, 0, 1);
		glTranslatef(0, 0, 0);
	}

	glLightfv(GL_LIGHT0, GL_POSITION, poslight);
	glLightfv(GL_LIGHT0, GL_AMBIENT, alight);
	glLightfv(GL_LIGHT0, GL_SPECULAR, slight);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, dlight);
	glLightf(GL_LIGHT0, GL_SPOT_EXPONENT, 10);

	draw_net();
	draw_graphics();

	glPopMatrix(); //////////////////

	swapBuffers();
}


void GLView::mousePressEvent(QMouseEvent *event)
{
	m_mouse_pt = event->pos();

	m_left_down = event->buttons().testFlag(Qt::LeftButton);
	m_wheel_down = event->buttons().testFlag(Qt::MiddleButton);

	m_wheel_pt = event->pos();

	m_delta_z = 0;
}

void GLView::mouseReleaseEvent(QMouseEvent *event)
{
	m_mouse_pt = event->pos();

	m_left_down = false;
	m_wheel_down = false;

	m_current_z += m_delta_z;
	m_delta_z = 0;
}

void GLView::mouseMoveEvent(QMouseEvent *event)
{
	if(event->buttons().testFlag(Qt::LeftButton)){
		QPointF pt = event->pos() - m_mouse_pt;
		m_delta_pt += pt;
		m_mouse_pt = event->pos();

		set_update();
	}
	if(event->buttons().testFlag(Qt::MiddleButton)){
		QPointF pt = event->pos() - m_wheel_pt;

		m_delta_z = pt.y()/5.;

		set_update();
	}
}
