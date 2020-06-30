from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
from keras import applications
from keras.models import Sequential
from keras.layers import Dropout,Flatten, Dense
import operator
import pandas as pd
import cv2
from tqdm import tqdm
from utils import video_to_image


class VGG16(object):
    def __init__(self):
        # build the VGG16 network  
        self.base_model = applications.VGG16(include_top=False, weights='imagenet')  
        self.__loaded_model()
    
    def __loaded_model(self, shape=(7, 7, 512), top_model_weights_path="models/vgg16_top_model_weights_updated.h5"):
        # build top model  
        self.model = self.__create_top_model("softmax", shape)
        self.model.load_weights(top_model_weights_path) 
    
    
    def __create_top_model(self, final_activation, input_shape, num_classes = 10):
        model = Sequential()  
        model.add(Flatten(input_shape=input_shape))  
        model.add(Dense(256, activation='relu'))  
        model.add(Dropout(0.5))  
        model.add(Dense(num_classes, activation=final_activation)) # sigmoid to train, softmax for prediction
        return model

    def __get_lables(self):
        class_labels = ['safe_driving', 'texting_right', 'talking_on_phone_right', 'texting_left', 'talking_on_phone_left',
                    'operating_radio', 'drinking', 'reaching_behind', 'doing_hair_makeup', 'talking_to_passanger']
        return class_labels            

    def _predict(self, image_path):
        target_size=(224,224)

        # prepare image for classification using keras utility functions
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image) # convert from PIL Image to NumPy array
        image /= 255
        # the dimensions of image should now be (150, 150, 3)

        # to be able to pass it through the network and use batches, we want it with shape (1, 224, 224, 3)
        image = np.expand_dims(image, axis=0)
        # print(image.shape)

        bottleneck_prediction = self.base_model.predict(image) 
        #print(bottleneck_prediction.shape[1:])
        
        # use the bottleneck prediction on the top model to get the final classification  
        class_predicted = self.model.predict_classes(bottleneck_prediction) 

        probs = self.model.predict(bottleneck_prediction) 
        return probs

    def test(self, test_df, folder_path):
        subimission_df = []
        subimission_df =pd.DataFrame(columns=["img", "c0", "c1" "c2", "c3", "c4",
                                            "c5", "c6", "c7", "c8", "c9"])
        for index, row in test_df.iterrows():
            image_name = row["img"]
            image_path = os.path.join(folder_path, image_name)
            probs = self._predict(image_path)[0]
            result = {}
            result["img"] = row["img"]
            count = 0
            for prob in probs:
                result["c{}".format(count)] = prob
            subimission_df = subimission_df.append(result, ignore_index=True)    

        return subimission_df
        

    def prediction(self, img_path):
        class_labels = self.__get_lables()

        probs = self._predict(img_path)
        #print(probs[0])
        decoded_predictions = dict(zip(class_labels, probs[0]))
        decoded_predictions = sorted(decoded_predictions.items(), key=operator.itemgetter(1), reverse=True)

        count = 1
        top_prediction, score = decoded_predictions[:1][0]
        return top_prediction, score

    def __getDuplicatesWithCount(self, listOfElems):
        ''' Get frequency count of duplicate elements in the given list '''
        dictOfElems = dict()
        # Iterate over each element in list
        for elem in listOfElems:
            # If element exists in dict then increment its value else add it in dict
            if elem in dictOfElems:
                dictOfElems[elem] += 1
            else:
                dictOfElems[elem] = 1    
     
        # Filter key-value pairs in dictionary. Keep pairs whose value is greater than 1 i.e. only duplicate elements from list.
        dictOfElems = { key:value for key, value in dictOfElems.items() if value > 1}
        # Returns a dict of duplicate elements and thier frequency count
        return dictOfElems

    def predict_video(self, image_paths):
        all_prediction = []
        with tqdm(total=len(image_paths)) as pbar:    
            for image_path in image_paths:
                pbar.update(1)
                prediction, _ = self.prediction(image_path)
                all_prediction.append(prediction)
        total_prediction = len(all_prediction)
        prediction_counts = self.__getDuplicatesWithCount(all_prediction)
        final_result = []
        for key, value in prediction_counts.items():
            score = round(float(value/total_prediction), 4)
            #print("{} Has predicted: count {} and occurnace: {}".format(key, value, score))
            result = {}
            result["label"] = key
            result["count"] = value
            result["occurred_ratio"] = score
            result["occurred_percentage"] = score*100
            final_result.append(result)
        final_result = sorted(final_result, key = lambda i: i['score'], reverse=True) 
        return final_result    

if __name__ == '__main__':
    VGG16_obj = VGG16()
    video_path = "videos/driver_distractor.mp4"
    image_paths = video_to_image(video_path)
    result = VGG16_obj.predict_video(image_paths)
    print(result)