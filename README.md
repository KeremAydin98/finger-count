# finger-count

Firstly, I have generated 5000 photos for each number(1,2,3,4,5) by taking every frame from videos in order to use as a data. The code which does this is called:

photo_generator.py

Some examples of photos:

![image_902](https://user-images.githubusercontent.com/77073029/143734216-e4b1df01-371d-449e-87dc-2d057cfbece5.png)

![image_911](https://user-images.githubusercontent.com/77073029/143734227-e0b6c8e2-a609-4088-9d26-81ef64bccd8e.png)

Then I have successfully trained a CNN model to detect how many fingers I am holding up. 

The classification report can be seen below:

                  precision  recall   f1-score   support

           0       0.98      0.87      0.92       906
           1       0.82      0.93      0.87       906
           2       0.88      0.93      0.91       906
           3       0.92      0.80      0.86       906
           4       0.93      0.98      0.96       906

    accuracy                            0.90      4530
    macro avg       0.91      0.90      0.90      4530
    weighted avg    0.91      0.90      0.90      4530

![image](https://user-images.githubusercontent.com/77073029/143734253-10e32304-c8e1-459f-9441-77e1e35a70ea.png)

And then I compared it with Convex Hull algorithm. Convex Hull algorithm was much more adaptive to different variations of finger holding, however the accuracy of deep learning was much higher in similar variations.

![image](https://user-images.githubusercontent.com/77073029/143734316-b43f8a36-2c49-4e87-b870-d21cad9fa629.png)

![image](https://user-images.githubusercontent.com/77073029/143734353-6b43e856-053f-4cec-a9dc-847fb9c1f71b.png)




