#include "stdio.h"
#include "iostream"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <sys/time.h>
#include <string.h>
#include <sys/stat.h>

#include "capture_v4l.h"

#define ANGLE_A 74.5
#define ANGLE_B 71.5
#define PROJECT_PATH "/home/t3min4l/workspace/error-detection/"
using namespace cv;
using namespace std;
std::ofstream logfile;

const int BIN_THRESHOLD_LOW = 160;
const int CANNY_THRESHOLD_LOW = 300;
const int IMG_SIZE = 512;
const int HOUGH_THRESHOLD_UNI = 40;
const int HOUGH_THRESHOLD_BODY = 50;
const int HOUGH_MIN_LINE_LENGTH_UNI = 20;
const int HOUGH_MIN_LINE_LENGTH_BODY = 30;
const int HOUGH_MAX_LINE_GAP_UNI = 3; 
const int HOUGH_MAX_LINE_GAP_BODY = 10;

// tinh toan khoang cach euclid
double euclidDist(Vec4i &line){
	return sqrt(static_cast<double>((line[0] - line[2]) * (line[0] - line[2]) + (line[1] - line[3])*(line[1]-line[3])));
}


// tinh goc theo radian tu (pi, -pi)
double calAngle(Vec4i &line){
	return atan2(line[3] - line[1], line[2] - line[0]);
}


// so sanh goc
bool compareAngle(Vec4i &l1, Vec4i &l2){
	double angle1 = calAngle(l1);
	double angle2 = calAngle(l2);
	return (angle1> angle2);
}


// so sanh khoang cach euclid
bool compareDist(Vec4i &l1, Vec4i &l2){
	double dist1 = euclidDist(l1);
	double dist2 = euclidDist(l2);
	return (dist1 > dist2);
}


// hien thi cua so anh co waitkey
void display_image_wk(Mat img, String img_name){
	if(!img.empty()){
		namedWindow("Display window", WINDOW_AUTOSIZE);
		imshow("Image" + img_name + ":", img);
		waitKey(5000);
		destroyAllWindows();
	}
	else{
		cout << "Image is not available to display" << endl;
	}
	
}


//hien thi anh real time
void display_image_no_wk(Mat img, String img_name, String window_name){
    if(!img.empty()){
        namedWindow(window_name, WINDOW_AUTOSIZE);
        imshow(img_name, img);
    }
    else{
        cout << "Image is not avaiable to display" << endl;
    }
}

//make save directories to save images
String make_dir(String filename, bool ok){
    String substr1 = "ok_";
    if(!ok) substr1 = "ng_";
    String substr2 = ".jpg";
    size_t idx1 = filename.find(substr1);
    String output_dirname = filename.substr(idx1);
    output_dirname = output_dirname.substr(0, output_dirname.size() - substr2.size());
    String path = "/home/t3min4l/workspace/error-detection/images/saved_images/" + output_dirname;
    cout << path << endl;
    int result = mkdir(path.c_str(), S_IRWXU|S_IRWXG|S_IROTH|S_IXOTH);
    if(result != 0){
        cout << "Can not create make save directory!" << endl;
        return "Failed!";
    }
    else{
        return output_dirname;
    }
}

// crop, gray and gaussian blur, open-morph image.
std::tuple<Mat, Mat, Mat, Mat> preprocess_img(Mat img){
    Mat img_crop, img_gray, img_binary;

    Rect roi = Rect(520, 220, IMG_SIZE, IMG_SIZE);
    img_crop = img(roi);

    cvtColor(img_crop, img_gray, COLOR_RGB2GRAY);
    GaussianBlur(img_gray, img_gray, Size(11,11), 1.5);
    threshold(img_gray, img_binary, BIN_THRESHOLD_LOW, 255, THRESH_BINARY);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3,3), Point(1,1));
    erode(img_binary, img_binary, kernel);
    dilate(img_binary, img_binary, kernel);

    return make_tuple(img_crop, img_gray, img_binary, kernel);
}

// detect components as 2 shells and the body
std::tuple<Mat, Mat, Mat, Mat> detect_components(Mat img_morph_bin){
    int top, bottom, left, right;
    int borderType;

    top = (int) 0.5*img_morph_bin.rows;
    bottom = (int) 0.5*img_morph_bin.rows;
    left = (int)0.5*img_morph_bin.cols;
    right = (int) 0.5*img_morph_bin.cols;

    Mat labelImage;
    Mat stats, centroids;
    int nLabels = connectedComponentsWithStats(img_morph_bin, labelImage, stats, centroids, 8, CV_32S);
    int max1 = 0, max2 = 0, max3 = 0;
    int label1 = 0, label2 = 0, label3 = 0;
    for (int label = 1; label < nLabels; label ++){
        int area = stats.at<int>(label, CC_STAT_AREA);
        if (area > max1)
        {
            max3 = max2;
            label3 = label2;
            max2 = max1;
            label2 = label1;
            max1 = area;
            label1 = label;
        }
        else if (area > max2)
        {
            max3 = max2;
            label3 = label2;
            max2 = area;
            label2 = label;
        }
        else if (area > max3)
        {
            max3 = area;
            label3 = label;
        }
    }

    vector<Vec3b> colors(nLabels);     //color for each component
    colors[0] = Vec3b(0, 0, 0);        //background component
    colors[label1] = Vec3b(0, 0, 255); //Red
    colors[label2] = Vec3b(0, 255, 0); //Green
    colors[label3] = Vec3b(255, 0, 0); //Blue

    Mat components(img_morph_bin.size(), CV_8UC3, Scalar(0,0,0));
    Mat body(img_morph_bin.size(), CV_8UC3, Scalar(0, 0, 0));
    Mat shell1(img_morph_bin.size(), CV_8UC3, Scalar(0, 0, 0));
    Mat shell2(img_morph_bin.size(), CV_8UC3, Scalar(0, 0, 0));

    for(int r = 0; r < components.rows; ++r){
        for (int c = 0; c < components.cols; ++c)
        {
            int label = labelImage.at<int>(r, c);
            Vec3b &pixel = components.at<Vec3b>(r, c);
            pixel = colors[label];
        }
    }

    int rshell1 = 0, rshell2 = 0;
    for (int r = 0; r < labelImage.rows; r++)
    {
        for (int c = 0; c < labelImage.cols; c++)
        {
            int label = labelImage.at<int>(r, c);
            if (label == label1)
            {
                Vec3b &pixel = body.at<Vec3b>(r, c);
                pixel = colors[label1];
            }
            if (label == label2)
            {
                Vec3b &pixel = shell1.at<Vec3b>(r, c);
                pixel = colors[label2];
                rshell1 = r;
            }
            if (label == label3)
            {
                Vec3b &pixel = shell2.at<Vec3b>(r, c);
                pixel = colors[label3];
                rshell2 = r;
            }
        }
    }
    if (rshell1 > rshell2)
    {
        Mat temp = shell1;
        shell1 = shell2; //upper shell
        shell2 = temp;   //below shell
    }

    return std::make_tuple(components, body, shell1, shell2);
}

// caculate body angle
double body_angle_calculate(Mat body_canny, Mat result_img){
    vector<Vec4i> body_lines;
    HoughLinesP(body_canny, body_lines, 1, CV_PI/ 360, HOUGH_THRESHOLD_BODY, HOUGH_MIN_LINE_LENGTH_BODY, HOUGH_MAX_LINE_GAP_BODY);

    double body_angle = 0;
    double angle_longest_line;
    Point p1, p2;
    if (body_lines.size() > 0){
        sort(body_lines.begin(), body_lines.end(), compareDist);
        Vec4i longest_line = body_lines[0];
        angle_longest_line = calAngle(longest_line);
        p1 = Point(longest_line[0], longest_line[1]);
        p2 = Point(longest_line[2], longest_line[3]);
        line(result_img, p1, p1, Scalar(0, 0, 255), 3, LINE_AA);
    }

    angle_longest_line = ceilf(angle_longest_line*180/CV_PI * 10)/10;

    return angle_longest_line;
}

// calculate shell angle
double shell_angle_calculate(Mat body_canny, Mat shell, Mat kernel, double body_angle, Mat result_img){
    vector<Vec4i> body_lines;
    Mat tmp_shell;
    double shell_angle = 0;
    int count = 0;
    cvtColor(shell, tmp_shell, COLOR_BGR2GRAY);
    dilate(tmp_shell, tmp_shell, kernel);

    HoughLinesP(body_canny, body_lines, 1, CV_PI/360, HOUGH_THRESHOLD_UNI, HOUGH_MIN_LINE_LENGTH_UNI, HOUGH_MAX_LINE_GAP_UNI);
    if(body_lines.size() > 0){
        Vec4i b_line;
        double angle_b_line = 0, edge_length = 0;
        for(int i = 0; i < body_lines.size(); i++){
            b_line = body_lines[i];
            Vec3b color = (rand() % 255, rand() % 255, rand() % 255);
            line(result_img, Point(b_line[0], b_line[1]), Point(b_line[2], b_line[3]), color, 1, LINE_AA);
            angle_b_line = calAngle(b_line);
            angle_b_line = angle_b_line * 180 / CV_PI;
            
            if (abs(angle_b_line) > 10) {
                
                if ((tmp_shell.at<uchar>(b_line[1], b_line[0]) > 0) || (tmp_shell.at<uchar>(b_line[3], b_line[2]) > 0) && (count < 2)) {
                    
                    if (edge_length == 0) {
                        edge_length = euclidDist(b_line);
                        shell_angle += angle_b_line;
                        count++;
                    }
                    else if (edge_length * 0.65 < euclidDist(b_line)){
                        shell_angle += angle_b_line;
                        count++;
                    }                            
                }
            }
        }
        
        if(count > 0){
            shell_angle = shell_angle/count;
        }

        shell_angle = abs(shell_angle - body_angle);

        return shell_angle;
    }

}

// print result into image
std::tuple<bool, bool> present_result(Mat display, double shellangle1, double shellangle2, double bodyangle){
    char str[128];
    sprintf(str, "shell angle 1 = %.1f", shellangle1);
    putText(display, str, Point(50, 50), FONT_HERSHEY_COMPLEX_SMALL, 1.0, Scalar(0, 255, 255), 1, LINE_AA);
    sprintf(str, "shell angle 2 = %.1f", shellangle2);
    putText(display, str, Point(50, 450), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 255, 255), 1, LINE_AA);
    sprintf(str, "body angle = %.1f", bodyangle);
    putText(display, str, Point(50, 250), FONT_HERSHEY_COMPLEX_SMALL, 1.0, Scalar(0, 0, 255), 1, LINE_AA);

    bool okng1 = true;
    bool okng2 = true;

    
    if (shellangle1 >= ANGLE_B && shellangle1 <= ANGLE_A) {
        putText(display, "OK", Point(350, 50), FONT_HERSHEY_COMPLEX_SMALL, 1.0, Scalar(0, 255, 0), 1, LINE_AA);
        okng1 = true;
    }
    else {
        putText(display, "NG", Point(350, 50), FONT_HERSHEY_COMPLEX_SMALL, 1.0, Scalar(0, 0, 255), 1, LINE_AA);
        okng1 = false;
    }
    if(shellangle2 >= ANGLE_B && shellangle2 <= ANGLE_A){
        putText(display, "OK", Point(350, 450), FONT_HERSHEY_COMPLEX_SMALL, 1.0, Scalar(0, 255, 0), 1, LINE_AA);
        okng2 = true;
    }
    else{
        putText(display, "NG", Point(350, 450), FONT_HERSHEY_COMPLEX_SMALL, 1.0, Scalar(0, 0, 255), 1, LINE_AA);
        okng2 = false;
    }
    return std::make_tuple(okng1, okng2);
    
}

// logging result to csv file
void logging_result(std::ofstream logfile, double bodyangle, double shellangle1, double shellangle2, bool okng1, bool okng2)
{
    logfile << bodyangle << ",";
    logfile << shellangle1 << ",";
    logfile << shellangle2 << ",";
    if(okng1 && okng2)
        logfile << "OK" << endl;
    else logfile << "NG" << endl;
}



main(int argc, char const *argv[])
{   
    int fd;
    if (argc > 2 || argc < 2){
        cout << "Invalid command line!" << endl;
        exit(1);
    }
    char option;
    cout << argv << endl;
    if (std::string(argv[1]) == "usecamera"){
        option = 'r';
    }
    if(std::string(argv[1]) == "loadimages"){
        option = 'l';
    }
    // else option = 'q';

    cout << option << endl;
    Mat imgbuf, img;
    size_t filesnum = 0;
    vector<String> filenames;


    if (option == 'l') {

        char logfilenames[90];
        struct timeval tv;
        gettimeofday(&tv, NULL);
        time_t curtime = tv.tv_sec;
        strftime(logfilenames, sizeof(logfilenames), "./logfile_%d%H%M%S.csv", localtime(&curtime));
        cout << "Logfile name:" << logfilenames << endl;

        logfile.open(logfilenames, std::ios_base::app);
        logfile << "filename"
                << ",body_angle"
                << ",shell_1_angle"
                << ",shell_2_angle"
                << ",ok/ng" << endl;

        int idx = 0;
        glob("/home/t3min4l/workspace/error-detection/images/test_images/ok_*.jpg", filenames, false);
        while(idx < filenames.size()){
            cout << idx << endl;
            img = imread(filenames[idx]);
            String save_dir = make_dir(filenames[idx], true);
            if (save_dir == "Failed!"){
                exit(1);
            }
            logfile << save_dir + ".jpg" << ",";

            // crop, gray and morph imgs
            auto imgs_preprocessed = preprocess_img(img);
            Mat img_crop, img_gray, img_binary, kernel;
            img_crop = get<0>(imgs_preprocessed);
            img_gray = get<1>(imgs_preprocessed);
            img_binary = get<2>(imgs_preprocessed);
            kernel = get<3>(imgs_preprocessed);

            display_image_wk(img_crop, "img_crop");
            // imwrite("./images/saved_images/" + save_dir + "/crop.jpg", img_crop);
            
            // extract components in imgs
            auto components_and_elements = detect_components(img_binary);
            Mat components, body, shell1, shell2;
            components = get<0>(components_and_elements);
            body = get<1>(components_and_elements);
            shell1 = get<2>(components_and_elements);
            shell2 = get<3>(components_and_elements);
            display_image_wk(components, "components");

            // imwrite("./images/saved_images" + save_dir + "/components.jpg", components);
            
            // calculate body and shells angles
            Mat body_canny;
            Mat result_img = img_crop.clone();
            int lowthres = CANNY_THRESHOLD_LOW, highthres = lowthres * 2.5;
            Canny(img_gray, body_canny, lowthres, highthres, 5);
            Point p1_longest_line, p2_longest_line;
            double body_angle;
            body_angle = body_angle_calculate(body_canny, result_img);

            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(body_canny, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

            /// Draw contours
            Mat drawing = Mat::zeros(body_canny.size(), CV_8UC3);
            for (int i = 0; i < contours.size(); i++)
            {
                Scalar color = Scalar(255, 255, 255);
                drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, Point());
            }
            display_image_wk(drawing, "contour");

            double shell1_angle = 0, shell2_angle = 0;
            shell1_angle = shell_angle_calculate(body_canny, shell1, kernel, body_angle, result_img);
            shell2_angle = shell_angle_calculate(body_canny, shell2, kernel, body_angle, result_img);
            // imwrite("./images/saved_images" + save_dir + "/contours.jpg", contours);
            

            // loggin and present result into image
            bool okng1, okng2;
            auto ok_ng_results = present_result(result_img, shell1_angle, shell2_angle, body_angle);
            okng1 = get<0>(ok_ng_results);
            okng2 = get<1>(ok_ng_results);


            logfile << body_angle << ",";
            logfile << shell1_angle << ",";
            logfile << shell2_angle << ",";
            if (okng1 && okng2)
                logfile << "OK" << endl;
            else
                logfile << "NG" << endl;
            display_image_wk(result_img, "Result");
            // imwrite("./images/saved_images" + save_dir + "/result.jpg", result_img);
            idx++;
        }
    }
    if(option == 'r') {

        char logfilenames[90];
        struct timeval tv;
        gettimeofday(&tv, NULL);
        time_t curtime = tv.tv_sec;
        strftime(logfilenames, sizeof(logfilenames), "./logfile_%d%H%M%S.csv", localtime(&curtime));
        cout << "Logfile name:" << logfilenames << endl;

        logfile.open(logfilenames, std::ios_base::app);
        logfile << "filename"
                << ",body_angle"
                << ",shell_1_angle"
                << ",shell_2_angle"
                << ",ok/ng" << endl;

        fd = open("/dev/video2", O_RDWR);
        if (fd == -1){
            perror("Opening video device");
            return -1;
        }
        print_caps(fd);
        init_mmap(fd);
        start_capture(fd);
        
        // Mat img, img_buf
        int count = 0;
        while(1){
            capture_image(fd);
            imgbuf = Mat(FRAME_HEIGHT, FRAME_WIDTH, CV_8U, (void*)buffer);
            img = imdecode(imgbuf, IMREAD_COLOR);
            display_image_no_wk(img, "img", "img");
            char filename[80];
            struct timeval tv;
            gettimeofday(&tv, NULL);
            time_t curtime;
            curtime = tv.tv_usec;
            strftime(filename, sizeof(filename), "OK_00_%H%m%s%n.jpg", localtime(&curtime));
            logfile << filename << ".jpg"
                    << ",";

            // preprocess images
            auto imgs_preprocessed = preprocess_img(img);
            Mat img_crop, img_gray, img_binary, kernel;
            img_crop = get<0>(imgs_preprocessed);
            img_gray = get<1>(imgs_preprocessed);
            img_binary = get<2>(imgs_preprocessed);
            kernel = get<3>(imgs_preprocessed);

            display_image_no_wk(img_crop, "img crop", "img crop");

            // extract components
            auto components_and_elements = detect_components(img_binary);
            Mat components, body, shell1, shell2;
            components = get<0>(components_and_elements);
            body = get<1>(components_and_elements);
            shell1 = get<2>(components_and_elements);
            shell2 = get<3>(components_and_elements);
            display_image_no_wk(components, "components", "components");

            // calculate body and shells angles
            Mat body_canny;
            Mat result_img = img_crop.clone();
            int lowthres = CANNY_THRESHOLD_LOW, highthres = lowthres * 2.5;
            Canny(img_gray, body_canny, lowthres, highthres, 5);
            Point p1_longest_line, p2_longest_line;
            double body_angle;
            body_angle = body_angle_calculate(body_canny, result_img);

            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(body_canny, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

            /// Draw contours
            Mat drawing = Mat::zeros(body_canny.size(), CV_8UC3);
            for (int i = 0; i < contours.size(); i++)
            {
                Scalar color = Scalar(255, 255, 255);
                drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, Point());
            }
            display_image_no_wk(drawing, "contour", "contour");

            double shell1_angle = 0, shell2_angle = 0;
            shell1_angle = shell_angle_calculate(body_canny, shell1, kernel, body_angle, result_img);
            shell2_angle = shell_angle_calculate(body_canny, shell2, kernel, body_angle, result_img);
            
            // logging and present result into image
            auto ok_ng_results = present_result(result_img, shell1_angle, shell2_angle, body_angle);
            bool okng1, okng2;
            okng1 = get<0>(ok_ng_results);
            okng2 = get<1>(ok_ng_results);
            display_image_no_wk(result_img, "result", "result");
            
            logfile << body_angle << ",";
            logfile << shell1_angle << ",";
            logfile << shell2_angle << ",";
            if (okng1 && okng2)
                logfile << "OK" << endl;
            else
                logfile << "NG" << endl;

            
            switch (waitKey(10))
            {
                case 'e':
                    
                    break;
                case 'c':
                    count ++;
                    imwrite(filename, img);
                    printf("\nImage %d saved!\n", count);
                default:
                    break;
            }
        }
        
    }
    else {
        printf("Wrong arguments!\nrealtime: for using camera\nloadimages: for using images on storage\n");
        exit(1);
    }

        return 0;
}
