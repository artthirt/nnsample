#ifndef QT_WORK_MAT_H
#define QT_WORK_MAT_H

#include <QObject>

#include "custom_types.h"

namespace qt_work_mat{

void q_save_mat(const ct::Matf &mat, const QString &filename);
void q_load_mat(const QString &filename, ct::Matf& mat);

}

#endif // QT_WORK_MAT_H
