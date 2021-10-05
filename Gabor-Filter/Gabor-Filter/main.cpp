#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/ml.hpp"
using namespace cv;

int main(int argc, char** argv) {
	VideoCapture cap;
	if (!cap.open(0))
		return 0;
	Mat frame;
	while (true) {
		cap.read(frame);
		cv::cvtColor(frame, frame,COLOR_BGR2GRAY);
		frame = frame > 128;
		imshow("Gabor Filter App", frame);
		waitKey(1);
	}
	return 0;
}