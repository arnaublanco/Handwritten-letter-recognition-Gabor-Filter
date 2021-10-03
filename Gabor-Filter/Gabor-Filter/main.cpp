#include <iostream>
#include "opencv2/opencv.hpp"
using namespace cv;

int main(int argc, char** argv) {
	VideoCapture cap;
	if (!cap.open(0))
		return 0;
	Mat frame;
	while (true) {
		cap.read(frame);
		imshow("Gabor Filter App", frame);
		waitKey(1);
	}
	return 0;
}