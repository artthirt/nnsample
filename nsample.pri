INCLUDEPATH += $$PWD/models

SOURCES += $$PWD/main.cpp\
        $$PWD/mainwindow.cpp \
    $$PWD/glview.cpp \
    $$PWD/simple_xml.cpp \
    $$PWD/models/nnmodel.cpp \
    $$PWD/widgetmnist.cpp \
    $$PWD/models/mnist_reader.cpp \
    $$PWD/models/mnist_train.cpp \
    $$PWD/models/mnist_conv.cpp

HEADERS  += $$PWD/mainwindow.h \
    $$PWD/glview.h \
    $$PWD/simple_xml.hpp \
    $$PWD/models/nnmodel.h \
    $$PWD/widgetmnist.h \
    $$PWD/models/mnist_reader.h \
    $$PWD/models/mnist_train.h \
    $$PWD/models/mnist_conv.h \
    $$PWD/models/mnist_utils.h

HEADERS += \
    $$PWD/drawcnvweight.h

SOURCES += \
    $$PWD/drawcnvweight.cpp

FORMS    += $$PWD/mainwindow.ui \
    $$PWD/glview.ui \
    $$PWD/widgetmnist.ui \
    $$PWD/drawcnvweight.ui
