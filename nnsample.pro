#-------------------------------------------------
#
# Project created by QtCreator 2016-12-22T09:27:40
#
#-------------------------------------------------

QT       += core gui opengl xml

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = nnsample
TEMPLATE = app

FORMS    += mainwindow.ui \
    glview.ui \
    widgetmnist.ui

win32-msvc*{
    QMAKE_CXXFLAGS += /openmp
}else{
    LIBS += -lgomp
    QMAKE_CXXFLAGS += -fopenmp
}

win32{
    LIBS += -lGLU32 -lopengl32
}else{
    LIBS += -lGLU
}

CONFIG(debug, debug|release){
    DST = "debug"
}else{
    DST = "release"
}

UI_DIR = tmp/$$DST/ui
OBJECTS_DIR = tmp/$$DST/obj
RCC_DIR = tmp/$$DST/rcc
MOC_DIR = tmp/$$DST/moc

include(nsample.pri)
include(ct/ct.pri)
