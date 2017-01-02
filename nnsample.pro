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
    custom_types.cpp \
    shared_memory.cpp

HEADERS  += mainwindow.h \
    custom_types.h \
    glview.h \
    simple_xml.hpp \
    nnmodel.h \
    widgetmnist.h \
    shared_memory.h

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
