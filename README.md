**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector. 
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/sample_images.jpg
[image2]: ./output_images/car_hogfeatures.jpg
[image3]: ./output_images/noncar_hogfeatures.jpg
[image4]: ./output_images/sliding_windows.jpg
[image5]: ./output_images/final1.jpg
[image6]: ./output_images/final2.jpg
[image7]: ./output_images/final3.jpg
[image8]: ./output_images/final4.jpg
[image9]: ./output_images/final5.jpg
[image10]: ./output_images/final6.jpg
[image11]: ./output_images/training1.jpg
[image12]: ./output_images/training2.jpg
[image13]: ./output_images/training3.jpg
[image14]: ./output_images/training4.jpg
[image15]: ./output_images/training5.jpg
[image16]: ./output_images/training6.jpg
[video1]: ./project_video_result.mp4


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a README that includes all the rubric points

You're reading it!

The IPython notebook is `vehicle-detection.ipynb`

Training dataset is from Udacity [car](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-car](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)  

### Histogram of Oriented Gradients (HOG)

#### 1. Extracted HOG features from the training images.

The code for this step is contained in the function `get_hog_features` and `extract_hog_features` in the IPython notebook

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) in function `hog_feature_test` at the very end of the python notebook 

I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

CAR IMAGE

![alt text][image2]

NON-CAR IMAGE

![alt text][image3]

#### 2. Settled on the final choice of HOG parameters and training.

I train the classifier in function `train_pipeline`. It uses `LinearSVC` with `StandardScaler`

I tried various combinations of parameters and below is the training result:

```python
Using colorspace: RGB with these details: 9 8 2 ALL
109.43 Seconds to extract hog feature
Feature vector length: 5292
20.51 Seconds to train SVC...
Test Accuracy of SVC =  0.971

Using colorspace: HSV with these details: 9 8 2 ALL
104.0 Seconds to extract hog feature
Feature vector length: 5292
19.52 Seconds to train SVC...
Test Accuracy of SVC =  0.9673

Using colorspace: HLS with these details: 9 8 2 ALL
116.89 Seconds to extract hog feature
Feature vector length: 5292
21.64 Seconds to train SVC...
Test Accuracy of SVC =  0.9696

Using colorspace: LUV with these details: 9 8 2 ALL
152.41 Seconds to extract hog feature
Feature vector length: 5292
19.91 Seconds to train SVC...
Test Accuracy of SVC =  0.9707

Using colorspace: YUV with these details: 9 8 2 ALL
105.05 Seconds to extract hog feature
Feature vector length: 5292
19.91 Seconds to train SVC...
Test Accuracy of SVC =  0.9727

Using colorspace: YCrCb with these details: 9 8 2 ALL
109.41 Seconds to extract hog feature
Feature vector length: 5292
21.17 Seconds to train SVC...
Test Accuracy of SVC =  0.9682
```

### Spatial and Historgram features
I extract spatial feature in function `bin_spatial`

I extract histogram feature in function `color_hist`

Then I combine both features in `extract_spatial_hist_feature` and train them. Below is the result:

```python
Using cspace: RGB with spatial binning of: 16 and 32 histogram bins
9.46 Seconds to extract color feature...
Feature vector length: 864
11.76 Seconds to train SVC...
Test Accuracy of SVC =  0.9476

Using cspace: HLS with spatial binning of: 16 and 32 histogram bins
9.99 Seconds to extract color feature...
Feature vector length: 864
7.51 Seconds to train SVC...
Test Accuracy of SVC =  0.9445

Using cspace: HSV with spatial binning of: 16 and 32 histogram bins
9.22 Seconds to extract color feature...
Feature vector length: 864
4.57 Seconds to train SVC...
Test Accuracy of SVC =  0.953

Using cspace: LUV with spatial binning of: 16 and 32 histogram bins
10.7 Seconds to extract color feature...
Feature vector length: 864
5.57 Seconds to train SVC...
Test Accuracy of SVC =  0.9513

Using cspace: YUV with spatial binning of: 16 and 32 histogram bins
8.75 Seconds to extract color feature...
Feature vector length: 864
6.0 Seconds to train SVC...
Test Accuracy of SVC =  0.9524

Using cspace: YCrCb with spatial binning of: 16 and 32 histogram bins
9.14 Seconds to extract color feature...
Feature vector length: 864
5.04 Seconds to train SVC...
Test Accuracy of SVC =  0.9445
```

### Final hyperparameters:
Based on my training result above, these are my final hyperparameters:

- color_space = **'YUV'**
- orient = **9**
- pix_per_cell = **8**
- cell_per_block = **2**
- hog_channel = **'ALL'**
- spatial_size = **16**
- hist_bins = **32**
- y_start_stop = **[400, 660]** # Min and max in y to search in slide_window()
- x_start_stop = **[250, 1280]** # Min and max in x to search in slide_window()

### Sliding Window Search

#### 1. Implemented a sliding window search

I decided to detect vehicles in windows at 6 different scales in the bottom half of the image and cut left 250px
- Total: 341 windows

![alt text][image4]

#### 2. Test images to demonstrate how the pipeline is working

My pipeline finally consists of 6 different window scale within each extracts all features: bin_spatial + color_hist + hog_features

```python
windows0 = slide_window(image, x_start_stop=x_start_stop, y_start_stop=[400,500], 
                    xy_window=(52, 52), xy_overlap=(0.5, 0.5))
windows1 = slide_window(image, x_start_stop=x_start_stop, y_start_stop=[400,500], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5))
windows2 = slide_window(image, x_start_stop=x_start_stop, y_start_stop=[400,600], 
                    xy_window=(80, 80), xy_overlap=(0.5, 0.5))
windows3 = slide_window(image, x_start_stop=x_start_stop, y_start_stop=[400,600], 
                    xy_window=(96, 96), xy_overlap=(0.4, 0.4))
windows4 = slide_window(image, x_start_stop=x_start_stop, y_start_stop=[400,600], 
                    xy_window=(110, 110), xy_overlap=(0.4, 0.4))
windows5 = slide_window(image, x_start_stop=x_start_stop, y_start_stop=[400,660], 
                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))
windows= windows0 +  windows1  + windows2 + windows3 + windows4 + windows5                    
```

I follow the guidance from lecture `Hog Sub-sampling Window Search` to make the pipeline more efficent. The code is in function `detect_cars`

function `car_search_pipeline` is where the pipeline gets executed

```python
def car_search_pipeline(image, heatmap_visualize = True):
    draw_image = np.copy(image)
    image = image.astype(np.float32)/255

    scales = [[0.8, 400, 500], [1, 400, 500], [1.25, 400, 600], [1.5, 400, 600], [1.8, 400, 600], [2, 400, 660]]
    hot_windows = []
    
    for scale in scales:
        bboxes = detect_cars(image, scale[1], scale[2], 250, 1280, scale[0], X_scaler = X_scaler, svc = classifier, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
        if len(bboxes) > 0:
            for box in bboxes:
                hot_windows.append(box)

    draw_img, heatmap = apply_heatmap(draw_image,hot_windows,5)
    
    if heatmap_visualize:
        return draw_img, heatmap
    else:
        return draw_img
```
When I run this function for each individual image, I receive below result

![alt text][image11]

![alt text][image12]

![alt text][image13]

![alt text][image14]

![alt text][image15]

![alt text][image16]

### Video Implementation

#### 1. Final video output
Here's a [link to my video result](./project_video_result.mp4)

#### 2. Filter for false positives and some method for combining overlapping bounding boxes.

With the above result from running the pipeline, there's a need to **filter false positives**

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded `heatmap_threshold = 5` that map to identify vehicle positions.  

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image5]

![alt text][image6]

![alt text][image7]

![alt text][image8]

![alt text][image9]

![alt text][image10]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- Clearly the video is not perfect. The bounding box sometimes not tight enough to the vehicle, sometimes it's smaller than the real vehicle, sometimes it loses track of the car
- False positives still appear in the final video, this is a sign that I should fine tune the sliding window search algorithm
- The bounding box is lagging from frame to frame, an improvement is needed to normalize / smooth the bounding box's positions & size
- Apply the technique from previous `Advanced Lane Finding` project, once a vehicle is detected, it's likely appears in the surrouding area in the next frame. This will increase the pipeline performance
- Accept the challenge (plug lane-finding pipeline) in next revision
