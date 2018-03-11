/**
 * this code is part of some javascript experiments with face recognition and face detection,
 * built on top of openCV
 * https://opencv.org/
 */
const path = require('path');
const cv = require('opencv4nodejs'); // openCV

const size = 120;
// CascadeClassifier class is used to detect faces in a video stream or image.
const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT);

/**
 * get the first face from a give image
 * @param image
 * @returns {*}
 */
const getFaceImage = (image) => {
  const detectedFaces = classifier.detectMultiScale(image).objects;
  if (!detectedFaces.length) throw new Error('no faces found');

  // prediction should be done on greyscale images of the same sizes
  return image.getRegion(detectedFaces[0]).bgrToGray().resize(size, size);
};

console.time('Face recognition');

// the image in which we gonna search Eleven (yes, from Stranger Things)
const predictionImg = cv.imread(path.join(__dirname, 'images', 'face-recognition.png'));

// other images of Eleven's face
const imagesPaths = [
  path.join(__dirname, 'images', 'face-0.png'),
  path.join(__dirname, 'images', 'face-1.png'),
  path.join(__dirname, 'images', 'face-2.png'),
  path.join(__dirname, 'images', 'face-3.png'),
];
// array of images
const images = imagesPaths.map((path) => cv.imread(path)).map(getFaceImage);

/**
 * EigenFaceRecognizer class is used to detect faces
 * https://docs.opencv.org/3.3.1/dd/d7c/classcv_1_1face_1_1EigenFaceRecognizer.html
 * @type {cv.EigenFaceRecognizer}
 */
const eigenRecognizer = new cv.EigenFaceRecognizer();

// let's train our face recognizer to recognize eleven:
// the 1st parameter is the array of Eleven's face images
// the second is an array of int identifiers for the picture we are training
eigenRecognizer.train(images, [11, 11, 11, 11]);

// let's find faces into predictionImg
classifier.detectMultiScaleAsync(predictionImg)
  .then(({ objects }) => {
    if (!objects || objects.length === 0) throw new Error('No objects found');

    let found = null;

    // for each found face let's predict if is Eleven
    objects.forEach((rect) => {
      const faceImage = (predictionImg.getRegion(rect)).bgrToGray().resize(size, size);
      const result = eigenRecognizer.predict(faceImage);

      // the more prediction confidence is near 0, the more efficient prediction is
      if (!found || found.prediction.confidence > result.confidence) {
        found = {
          prediction: result,
          rect,
        }
      }
    });

    console.timeEnd('Face recognition');

    return found;
  })
  .then((found) => {
    if (!found) throw new Error('No face found');
    const { rect } = found;

    // let's draw a rectangle on found face
    const point1 = new cv.Point(rect.x, rect.y);
    const point2 = new cv.Point(rect.x + rect.width, rect.y + rect.height);
    const vector = new cv.Vec(rect.x, rect.y, .5);
    const thickness = 2;

    predictionImg.drawRectangle(point1, point2, vector, cv.LINE_4, thickness);

    cv.imshowWait('Eleven', predictionImg);
  })
  .catch((err) => console.error(err));