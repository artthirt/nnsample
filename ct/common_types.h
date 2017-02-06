#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

namespace ct{

struct Size{
	Size(){
		width = height = 0;
	}
	Size(int w, int h){
		width = w;
		height = h;
	}
	int area() const{
		return width * height;
	}

	int width;
	int height;
};

}

#endif // COMMON_TYPES_H