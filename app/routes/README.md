# Routes 

We've added several routes here in this module. 
What's important here is the functionality

## Start
```
/api/start/
```

To start the job we pass in a video: video file, and image:image file. This returns a job. 

Simple. It also uploads a video and image and starts a job.

## Preprocess
```
/api/pre-process/
```

To pre-process the frames from a video (extracting the frames and storing on firebase) you 

Simple. It also uploads a video and image and starts a job.



## Upload 

**Deprecated**


API Upload Path: 
```
/api/upload/
```

##### Upload Videos:
Videos get uploaded to firebase in the firebase folder for videos/ 
The way to do this is by body using the key `video` uploading the target video there.

