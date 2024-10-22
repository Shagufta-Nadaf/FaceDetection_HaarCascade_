#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace dnn;
using namespace std;

const int YU_NET_INPUT_SIZE = 820;

int main() {
    // Load the YuNet model
    string modelPath = "/home/itstarkenn/Downloads/face_detection_yunet_2023mar.onnx";
   
    // Create face detector
    Ptr<FaceDetectorYN> faceDetector;
    try {
        faceDetector = FaceDetectorYN::create(modelPath, "", Size(YU_NET_INPUT_SIZE, YU_NET_INPUT_SIZE), 0.9, 0.3, 5000);
    } catch (const cv::Exception& e) {
        cerr << "Error loading model: " << e.what() << endl;
        return -1;
    }

    // Open webcam
    VideoCapture capture(0);
    if (!capture.isOpened()) {
        cerr << "Error: Could not open webcam." << endl;
        return -1;
    }

    Mat frame;
    while (true) {
        capture >> frame; // Capture a new frame
        if (frame.empty()) break; // Exit if no frame is captured

        // Resize frame to match expected input size for face detection
        Mat resizedFrame;
        resize(frame, resizedFrame, Size(YU_NET_INPUT_SIZE, YU_NET_INPUT_SIZE));

        // Detect faces
        Mat faces;
        try {
            faceDetector->detect(resizedFrame, faces);
        } catch (const cv::Exception& e) {
            cerr << "Error during face detection: " << e.what() << endl;
            continue; // Skip this iteration if an error occurs
        }

        // Process and draw detected faces
        for (int i = 0; i < faces.rows; i++) {
            float x1 = faces.at<float>(i, 0);
            float y1 = faces.at<float>(i, 1);
            float w = faces.at<float>(i, 2);
            float h = faces.at<float>(i, 3);
            float confidence = faces.at<float>(i, 4);

            // Scale coordinates back to original frame size
            int originalX1 = static_cast<int>(x1 * (frame.cols / static_cast<float>(YU_NET_INPUT_SIZE)));
            int originalY1 = static_cast<int>(y1 * (frame.rows / static_cast<float>(YU_NET_INPUT_SIZE)));
            int originalX2 = static_cast<int>((x1 + w) * (frame.cols / static_cast<float>(YU_NET_INPUT_SIZE)));
            int originalY2 = static_cast<int>((y1 + h) * (frame.rows / static_cast<float>(YU_NET_INPUT_SIZE)));

            // Draw bounding box and confidence score
            rectangle(frame, Point(originalX1, originalY1), Point(originalX2, originalY2), Scalar(0, 255, 0), 2);
            putText(frame, format("%.2f", confidence), Point(originalX1, originalY1 - 5),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);

            // Keypoint detection using ORB on the detected face region
            Mat faceROI = frame(Rect(Point(originalX1, originalY1), Point(originalX2, originalY2)));

            // Ensure the ROI is valid before processing
            if (faceROI.empty()) {
                cerr << "Empty face ROI!" << endl;
                continue; // Skip this iteration if ROI is invalid
            }
           
            Ptr<ORB> orb = ORB::create();
            std::vector<KeyPoint> keypoints;
            Mat descriptors;

            orb->detectAndCompute(faceROI, noArray(), keypoints, descriptors);

            // Limit the number of keypoints to display to a maximum of 5
            int maxKeypointsToShow = min(5, static_cast<int>(keypoints.size()));

            for (int j = 0; j < maxKeypointsToShow; j++) {
                const auto& kp = keypoints[j];
                int kpX = static_cast<int>(kp.pt.x + originalX1);
                int kpY = static_cast<int>(kp.pt.y + originalY1);

                // Draw a filled circle for each keypoint
                circle(frame, Point(kpX, kpY), 3, Scalar(255, 0 ,255), -1);
            }
        }

        imshow("Face Detection with Keypoints", frame); // Show the output frame
       
        if (waitKey(30) >= 0) break; // Exit on key press
    }

    capture.release(); // Release the video capture object
    destroyAllWindows(); // Close all OpenCV windows
    return 0; // Exit the program successfully
}

