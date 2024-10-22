
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
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

    // Load an image from file
    Mat frame = imread("/home/itstarkenn/Downloads/srk.jpeg"); // Update with your image path
    if (frame.empty()) {
        cerr << "Error: Could not open or find the image." << endl;
        return -1;
    }

    // Resize frame to match expected input size for face detection
    Mat resizedFrame;
    resize(frame, resizedFrame, Size(YU_NET_INPUT_SIZE, YU_NET_INPUT_SIZE));

    // Detect faces
    Mat faces;
    try {
        faceDetector->detect(resizedFrame, faces);
    } catch (const cv::Exception& e) {
        cerr << "Error during face detection: " << e.what() << endl;
        return -1; // Exit if an error occurs
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
        
        // Draw bounding box and confidence score
        rectangle(frame, Point(originalX1, originalY1), Point(originalX1 + w * (frame.cols / static_cast<float>(YU_NET_INPUT_SIZE)), 
                originalY1 + h * (frame.rows / static_cast<float>(YU_NET_INPUT_SIZE))), Scalar(0, 255, 0), 2);
        
        putText(frame, format("Confidence: %.2f", confidence), Point(originalX1 + 5, originalY1 - 5),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);

        // Calculate keypoint positions relative to bounding box dimensions
        vector<Point> keypoints;

        // Define keypoint positions relative to bounding box dimensions
        keypoints.push_back(Point(originalX1 + w * 0.3f * (frame.cols / static_cast<float>(YU_NET_INPUT_SIZE)), originalY1 + h * 0.35f * (frame.rows / static_cast<float>(YU_NET_INPUT_SIZE)))); // Left Eye Corner
        keypoints.push_back(Point(originalX1 + w * 0.7f * (frame.cols / static_cast<float>(YU_NET_INPUT_SIZE)), originalY1 + h * 0.35f * (frame.rows / static_cast<float>(YU_NET_INPUT_SIZE)))); // Right Eye Corner
        keypoints.push_back(Point(originalX1 + w * 0.5f * (frame.cols / static_cast<float>(YU_NET_INPUT_SIZE)), originalY1 + h * 0.5f * (frame.rows / static_cast<float>(YU_NET_INPUT_SIZE)))); // Nose Tip
        keypoints.push_back(Point(originalX1 + w * 0.35f * (frame.cols / static_cast<float>(YU_NET_INPUT_SIZE)), originalY1 + h * 0.75f * (frame.rows / static_cast<float>(YU_NET_INPUT_SIZE)))); // Left Mouth Corner
        keypoints.push_back(Point(originalX1 + w * 0.65f * (frame.cols / static_cast<float>(YU_NET_INPUT_SIZE)), originalY1 + h * 0.75f * (frame.rows / static_cast<float>(YU_NET_INPUT_SIZE)))); // Right Mouth Corner


        // Draw selected keypoints on the frame with enhanced visibility
        for (size_t j = 0; j < keypoints.size(); j++) {
            circle(frame, keypoints[j], 3, Scalar(255, j*50 ,255), -1); // Use different colors for each point for better visibility
            
            putText(frame,
                    "", 
                    Point(keypoints[j].x + 10, keypoints[j].y), FONT_HERSHEY_SIMPLEX,
                    0.4,
                    Scalar(255,255,255),
                    1); // Add labels next to points if needed
        }
    }

    imshow("Face Detection with Keypoints", frame); // Show the output frame
    
    waitKey(0); // Wait indefinitely until a key is pressed

    return 0; // Exit the program successfully
}
