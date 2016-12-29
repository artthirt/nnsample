#-------------------------------------------------
#
# Project created by QtCreator 2016-12-22T09:27:40
#
#-------------------------------------------------

QT       += core gui opengl xml

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = nnsample
TEMPLATE = app

SOURCES += main.cpp\
        mainwindow.cpp \
    glview.cpp \
    simple_xml.cpp \
    nnmodel.cpp \
    widgetmnist.cpp \
    custom_types.cpp

HEADERS  += mainwindow.h \
    custom_types.h \
    glview.h \
    simple_xml.hpp \
    nnmodel.h \
    widgetmnist.h

FORMS    += mainwindow.ui \
    glview.ui \
    widgetmnist.ui

win32{
    LIBS += -lopengl32 -lglu32
    QMAKE_CXXFLAGS += /openmp
}else{
    LIBS += -lGLU -lgomp
    QMAKE_CXXFLAGS += -fopenmp
}
