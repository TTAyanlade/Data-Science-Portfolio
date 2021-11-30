This is a python application that checks if a specific person
is present in a list of group images. It returns the set of
images where the person is present.


 
To Use this: 

........Copy the images for training into the 
-trainingimages- folder

........Then Copy the images to be scanned 
into the -Comparison- folder.


........Launch CR.exe (Wait while it Initializes, untill 
GUI pops up)

........Select 'Choose the Training images'

........Select 'Preprocess the Training Images'

........Select 'Model'

........Select 'Execute Comparison Process'-Main Process-


Note: The Progress bar is dependent on the number of 
images in the -Comparison- folder

........Finally, Similar images would be moved to the
-Similar- folder in the Main Folder 


Note: The Application uses a freeze; while a process is
running, the button stays depressed till it is done.
This ensures you do not interact with the app while 
processing. Although you can use another application.

Note: To improve model, additional dummy images should be
added to the category2 folder -----To do this, add random images
to the trainingimages folder, preprocess them then move the 
cropped images to the category2 folder----- 