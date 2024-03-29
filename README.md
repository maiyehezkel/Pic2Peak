# Pic2Peak

Artificial intelligence system called "Pic2Peak". This system knows how to turn a two-dimensional landscape image into a three-dimensional environment. Using CNN model of TensorFlow, Python, and Blender.

![image](https://user-images.githubusercontent.com/93497035/180607993-0e301b94-9cae-4154-9cd6-c0d8a53423fc.png)
![image](https://user-images.githubusercontent.com/93497035/180607995-015f159b-c241-4788-a419-115db04e953e.png)
![image](https://user-images.githubusercontent.com/93497035/180608000-1599e287-57e8-4ad2-896f-b0661c4cba6a.png)

* For more information you can download "Pic2Peak Project Book" from this repository, and read it (written in Hebrew).

# How to use Pic2Peak:
1) Open "Canvas.py" and run it. This code do Image detection on picture from your computer, And generete for you at the end CSV file with all the data that the model found.
2) After the program finishes running, we will save the CSV file.
* I upload the result file to this repository, You can open "DataTable.csv" and see the output CSV file.
  
2) For this part you will need Blender on your computer - https://www.blender.org/. In Blender open a "Text Editor" window and copy "Pic2PeakForBlender.py" to it. make sure that the path to the CSV file is correct (line 23). then you can run it from "Run Script" button in the window, and the environment will be built.

