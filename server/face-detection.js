/**
 * this code is part of some javascript experiments with face recognition and face detection,
 * built on top of openCV
 * https://opencv.org/
 */
const path = require('path');
const cv = require('opencv4nodejs'); // openCV

let image = null; // will contain the instance of the loaded image
const imagePath = path.join(__dirname, 'images', 'face-detection.jpg');
const { HAAR_FRONTALFACE_ALT } = cv; // face detection XML model

/**
 * according to openCV documentation:
 * CascadeClassifier class is used to detect objects in a video stream.
 * The first parameter could be a detection model XML file.
 * https://docs.opencv.org/2.4/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html
 */
const classifier = new cv.CascadeClassifier(HAAR_FRONTALFACE_ALT);

// let's track performances
console.time('Face detection');

// load the image
cv.imreadAsync(imagePath)
  .then(result => {
    image = result;
    // let's detect faces
    return classifier.detectMultiScaleAsync(image);
  })
  .then(({ objects }) => {
    if (!objects || objects.length === 0) throw new Error('No objects found');

    // create a rectangle for each found face
    objects.forEach((rect) => {
      const point1 = new cv.Point(rect.x, rect.y);
      const point2 = new cv.Point(rect.x + rect.width, rect.y + rect.height);
      const vector = new cv.Vec(rect.x, rect.y, .5);
      const thickness = 2;

      image.drawRectangle(point1, point2, vector, cv.LINE_4, thickness);
    });

    // detection end,
    console.timeEnd('Face detection');

    return { objects };
  })
  .then(() => cv.imshowWait('Detected faces', image))
  .catch(err => console.error(err));