#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "opencv2/calib3d.hpp"

using namespace std;
using namespace cv;

/*
@function computeReprojectionErrors
*/

static double computeReprojectionErrors(
	const vector<vector<Point3f> >& objectPoints,
	const vector<vector<Point2f> >& imagePoints,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	vector<float>& perViewErrors)
{
	size_t totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());
	vector<Point2f> imagePoints2;
	for (size_t i = 0; i < objectPoints.size(); ++i)
	{
		projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
		err = norm(imagePoints[i], imagePoints2, NORM_L2);
		size_t n = objectPoints[i].size();
		perViewErrors[i] = (float)sqrt(err*err / n);
		totalErr += err * err;
		totalPoints += n;
	}
	return sqrt(totalErr / totalPoints);
}

/*
@function unidstort
*/

static Mat unidstort(Mat img, Mat intrinsic, Mat distortionCoef) {
	// 4. Use the OpenCV's functions to compute undistorted image
	// 4.2 Compute the undistortion and rectification transformation map
	Mat map_x, map_y;
	initUndistortRectifyMap(intrinsic, distortionCoef, Mat(), intrinsic, Size(img.cols, img.rows), 
		CV_32FC1, map_x, map_y);
	Mat undistorted_image_out1; // output undistorted image
	remap(/*in*/img, /*out*/undistorted_image_out1, map_x, map_y, INTER_LINEAR, BORDER_TRANSPARENT);
	return undistorted_image_out1;
}

int main() {

	//upload images in a "data" vector
	vector<String> fn;
	vector<Mat> data;
	glob("checkerboard_images/*png", fn, true);
	for (size_t k = 0; k < fn.size(); ++k)
	{
		Mat im = imread(fn[k]);
		if (im.empty()) continue; //only proceed if sucsessful
		data.push_back(im);
	}

	Size patternsize(6, 5);

	int numSquares = 5 * 6; //i'll use it later
	/*
	List of 3D point expressed in float: physical position of the corners in 3D space
	List of 2D corners expressed in float: location of the corners on the image (of course it's in 2D).
	*/
	vector<vector<Point3f>> points_3d;
	vector<vector<Point2f>> points_2d;
	vector<Point2f> corners; //this will be filled by the detected corners

	/*
	Ideally, I should measure every distance from the lens for each picture of the cheeserboard taken;
	Mathematically, I just create a vector of 3D point called obj: thanks to the following loop I create
	a list of coordinates (0, 0, 0), (0, 1, 0), (0, 2, 0), ... (1, 4, 0) and so on. Each corresponds to a particular
	vertex.
	*/
	vector<Point3f> obj;
	for (int j = 0; j < numSquares; j++)
		obj.push_back(Point3f(j / 6, j%6, 0.0f));

	Mat gray; //I need a 1 channel image for the method findChesserboardCorners

	//with the following loop I get all the corners for each image
	for (size_t k = 0; k < data.size(); ++k) {
		cvtColor(data[k], gray, COLOR_BGR2GRAY);//source image
		
		bool patternfound = findChessboardCorners(gray, patternsize, corners,
			CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
			+ CALIB_CB_FAST_CHECK);

		if (patternfound) {
			cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
				TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
		}

		drawChessboardCorners(data[k], patternsize, Mat(corners), patternfound);

		points_2d.push_back(corners);
		points_3d.push_back(obj);
	}
	
	//Matrixes of the unkown
	Mat intrinsic = Mat(3, 3, CV_32F);
	Mat distortion_values;
	vector<Mat> rvecs;
	vector<Mat> tvecs;

	//Element (0, 0) and (1, 1) are the focal lenght along X and Y axis
	intrinsic.ptr<float>(0)[0] = 1;
	intrinsic.ptr<float>(1)[1] = 1;

	double error = calibrateCamera(points_3d, points_2d, data[0].size(), intrinsic, distortion_values, rvecs, tvecs);

	vector<float> perViewErrors;
	double error_ =	computeReprojectionErrors(points_3d, points_2d, rvecs, tvecs, intrinsic, distortion_values,
		perViewErrors); 

	//computing all the errors among all the input images: after that I'll take the biggest and the smallest error using 2 for loops

	float max = 0;
	int indexmax = 0;
	int indexmin = 0;

	for (int i = 0; i < perViewErrors.size(); i++) {
		if (perViewErrors[i] > max) {
			max = perViewErrors[i];
			indexmax = i;
		}
	}
	float min = max;
	for (int i = 0; i < perViewErrors.size(); i++) {
		if (perViewErrors[i] < min) {
			min = perViewErrors[i];
			indexmin = i;
		}
	}

	string fullmin = fn[indexmin];
	string fullmax = fn[indexmax];

	string image_min = fullmin.substr(fullmin.size() - 14, 14);
	string image_max = fullmax.substr(fullmax.size() - 14, 14);

	cout << "The picture that best performs calibration is " << image_min << endl;
	cout << "The picture that worst performs calibration is " << image_max << endl;

	cout << "Inrinsic matrix is " << intrinsic << "." << endl;
	cout << "Distortion values are" << distortion_values << "." << endl;
	cout << "Reprojection error on the calibrate method is " << error << "." << endl;
	cout << "Reprojection error on the computeReprojectionError method is " << error_ << "." << endl;
	cout << "To get these 2 informations I've used the reprojecting error on the single images." << endl;

	//getting informations about the intrinsic matrix
	//fx and fy are the x and y focal lengths
	double fx = intrinsic.at<double>(0, 0);
	double fy = intrinsic.at<double>(1, 1);

	//cy and cy are the x and y coordinates of the optical center in the image plane
	double cx = intrinsic.at<double>(0, 2);
	double cy = intrinsic.at<double>(1, 2);

	cout << "About the intrinsic matrix:" << endl;
	cout << "Focal lenght among X and Y axis is: " << fx << ", " << fy << "." << endl;
	cout << "Coordinates of the optical center in the image plane among X and Y axis is " << cx << ", " << cy << "." << endl;

	double k1 = distortion_values.at<double>(0, 0);
	double k2 = distortion_values.at<double>(0, 1);
	double p1 = distortion_values.at<double>(0, 2);
	double p2 = distortion_values.at<double>(0, 3);
	double k3 = distortion_values.at<double>(0, 4);

	cout << "About the distortion matrix:" << endl;
	//k1, k2 are the polinomyal coefficients for the radial distortion component, p1 and p2 are the parameters of the tangential
	//distortion component and the 5th is the 3rd coefficient of the radial distortion component
	cout << "k1, k2 are the polinomyal coefficients for the radial distortion component and their value are " << k1 << ", " << k2 << "." << endl;
	cout << "p1 and p2 are the parameters of the tangential distortion component and they are worth " << p1 << " and " << p2 << endl;
	cout << "The 5th is the 3rd coefficient of the radial distortion component and its value is " << k3 << endl;

	//undistorting the image thank to the undistort method defined at the begginning of the code, before the main() method.
	Mat image = imread("test_image.png");
	if (!image.data) {
		cout << "Cannot read image." << endl;
		return -1;
	}
	Mat undistorted = unidstort(image, intrinsic, distortion_values);

	resize(image, image, Size(image.cols / 4, image.rows / 4));
	resize(undistorted, undistorted, Size(undistorted.cols / 4, undistorted.rows / 4));

	Mat output[] = { image, undistorted };

	Mat out;
	hconcat(output, 2, out);

	imshow("win1", out);

	waitKey(0);
	return 0;
}