import os
import math
import time
import numpy as np
from keras.engine.training import GeneratorEnqueuer
#from model_factory import Model_Factory
from tools.save_images import save_img3
from tools.yolo_utils import *
from keras.preprocessing import image
import keras.backend as K
"""
Interface for normal (one net) models and adversarial models. Objects of
classes derived from Model are returned by method make() of the Model_Factory
class.
"""
class Model():
    def train(self, train_gen, valid_gen, cb):
        pass

    def predict(self, test_gen, tag='pred'):
        pass

    def test(self, test_gen):
        pass


"""
Wraper of regular models like FCN, SegNet etc consisting of a one Keras model.
But not GANs, which are made of two networks and have a different training
strategy.
In this class we implement the train(), test() and predict() methods common to
all of them.
"""
# TODO: Better call it Regular_Model ?
class One_Net_Model(Model):
    def __init__(self, model, cf, optimizer):
        self.cf = cf
        self.optimizer = optimizer
        self.model = model

    # Train the model
    def train(self, train_gen, valid_gen, cb):
        if (self.cf.train_model):
            print('\n > Training the model...')
            hist = self.model.fit_generator(generator=train_gen,
                                            steps_per_epoch=self.cf.dataset.n_images_train//self.cf.batch_size_train,
                                            epochs=self.cf.n_epochs,
                                            verbose=1,
                                            callbacks=cb,
                                            validation_data=valid_gen,
                                            validation_steps = self.cf.dataset.n_images_valid//self.cf.batch_size_valid,
                                            class_weight=None,
                                            max_queue_size=10,
                                            workers=1,
                                            use_multiprocessing=False)
            print('   Training finished.')

            return hist
        else:
            return None

    # Predict the model
    def predict(self, test_gen, tag='pred'):
        if self.cf.pred_model:
            # TODO model predict method for other tasks
            print('\n > Predicting the model...')

            if self.cf.problem_type == 'segmentation':
                # Load best trained model
                self.model.load_weights(self.cf.weights_file)

                # Create output directory
                if not os.path.exists(os.path.join(self.cf.savepath,'Predictions')):
                    os.makedirs(os.path.join(self.cf.savepath,'Predictions'))

                # Create a data generator
                enqueuer = GeneratorEnqueuer(test_gen,wait_time=0.05)
                enqueuer.start(workers=1,max_queue_size=1)
                output_generator = enqueuer.get()

                # Process the dataset
                start_time = time.time()
                for i in range(int(math.ceil(self.cf.dataset.n_images_test/float(self.cf.batch_size_test)))):

                    # Get data for this minibatch
                    data = next(output_generator)
                    x_true = data[0]
                    y_true = data[1].astype('int32')

                    # Get prediction for this minibatch
                    y_pred = self.model.predict(x_true)

                    # Reshape y_true and compute the y_pred argmax
                    if K.image_dim_ordering() == 'th':
                        print('Predict method not implemented for th dim ordering.')
                        return
                    else:
                        # Find the most probable class of each pixel
                        y_pred = np.argmax(y_pred, axis=2)
                        y_true = np.argmax(y_true, axis=2)

                        # Reshape from (?,172800) to (?, 360, 480)
                        # print('Acc with shapes y_pred={} and y_true{}: {}'.format(y_pred.shape,y_true.shape,np.mean(np.ravel(y_pred==y_true))))
                        y_true = np.reshape(y_true, (x_true.shape[0], x_true.shape[1], x_true.shape[2]))
                        y_pred = np.reshape(y_pred, (x_true.shape[0], x_true.shape[1], x_true.shape[2]))
                        # print('Acc with shapes y_pred={} and y_true{}: {}'.format(y_pred.shape,y_true.shape,np.mean(np.ravel(y_pred==y_true))))

                    # Save output images
                    save_img3(x_true, y_true, y_pred, os.path.join(self.cf.savepath,'Predictions'), 0,
                              self.cf.dataset.color_map, self.cf.dataset.classes, tag + str(i), self.cf.dataset.void_class)

                # Stop data generator
                if enqueuer is not None:
                    enqueuer.stop()
            else:
                print('Predict method not implemented.')
                return

            total_time = time.time() - start_time
            fps = float(self.cf.dataset.n_images_test) / total_time
            s_p_f = total_time / float(self.cf.dataset.n_images_test)
            print ('   Predicting time: {}. FPS: {}. Seconds per Frame: {}'.format(total_time, fps, s_p_f))

    # Test the model
    def test(self, test_gen):
        if self.cf.test_model:
            print('\n > Testing the model...')

            # Load best trained model
            print('\n > Loading model from '+self.cf.weights_test_file)
            self.model.load_weights(self.cf.weights_test_file)

            #if not self.cf.problem_type == 'detection':
            if True:
                # Evaluate model
                start_time_global = time.time()
                test_metrics = self.model.evaluate_generator(test_gen,
                                                             steps=self.cf.dataset.n_images_test//self.cf.batch_size_test,
                                                             max_queue_size=10,
                                                             workers=1)

                total_time_global = time.time() - start_time_global
                fps = float(self.cf.dataset.n_images_test) / total_time_global
                s_p_f = total_time_global / float(self.cf.dataset.n_images_test)
                print ('   Testing time: {}. FPS: {}. Seconds per Frame: {}'.format(total_time_global, fps, s_p_f))
                metrics_dict = dict(zip(self.model.metrics_names, test_metrics))
                print ('   Test metrics: ')
                for k in metrics_dict.keys():
                    print ('      {}: {}'.format(k, metrics_dict[k]))

            if self.cf.problem_type == 'detection':
                # Dataset and the model used
                dataset_name = self.cf.dataset_name 
                #model_name = self.cf.model_name 
                # Net output post-processing needs two parameters:
                detection_threshold = 0.3 # Min probablity for a prediction to be considered
                nms_threshold       = 0.2 # Non Maximum Suppression threshold
                # IMPORTANT: the values of these two params will affect the final performance of the netwrok
                #            you are allowed to find their optimal values in the validation/train sets
                
                if dataset_name == 'TT100K_detection':
                    classes = ['i2','i4','i5','il100','il60','il80','io','ip','p10','p11','p12','p19','p23','p26','p27','p3','p5','p6','pg','ph4','ph4.5','ph5','pl100','pl120','pl20','pl30','pl40','pl5','pl50','pl60','pl70','pl80','pm20','pm30','pm55','pn','pne','po','pr40','w13','w32','w55','w57','w59','wo']
                elif dataset_name == 'Udacity':
                    classes = ['Car','Pedestrian','Truck']
                else:
                    print "Error: Dataset not found!"
                    quit()
                priors = [[0.9,1.2], [1.05,1.35], [2.15,2.55], [3.25,3.75], [5.35,5.1]]
                
                input_shape = (self.cf.dataset.n_channels,
                        self.cf.target_size_test[0],
                        self.cf.target_size_test[1])
                

                
                test_dir = test_gen.directory
                imfiles = [os.path.join(test_dir,f) for f in os.listdir(test_dir) 
                                    if os.path.isfile(os.path.join(test_dir,f)) 
                                    and f.endswith('jpg')]
                inputs = []
                img_paths = []
                chunk_size = 128 # we are going to process all image files in chunks
                
                ok = 0.
                total_true = 0.
                total_pred = 0.
                start_time = time.time()
                for i, img_path in enumerate(imfiles):
                    img = image.load_img(img_path, target_size=(input_shape[1], input_shape[2]))
                    img = image.img_to_array(img)
                    img = img / 255.
                    inputs.append(img.copy())
                    img_paths.append(img_path)
                    
                    if len(img_paths)%chunk_size == 0 or i+1 == len(imfiles):
                        print(str(i)+'/'+str(len(imfiles)))
                        inputs = np.array(inputs)
                        net_out = self.model.predict(inputs, batch_size = self.cf.batch_size_test, verbose = 0)
                        # Find correct detections (per image)
                        for i, img_path in enumerate(img_paths):
                            boxes_pred = yolo_postprocess_net_out(net_out[i], priors, classes, detection_threshold, nms_threshold)
                            boxes_true = []
                            label_path = img_path.replace('jpg','txt')
                            gt = np.loadtxt(label_path)
                            if len(gt.shape) == 1:
                                gt = gt[np.newaxis,]
                            for j in range(gt.shape[0]):
                                bx = BoundBox(len(classes))
                                bx.probs[int(gt[j,0])] = 1.
                                bx.x, bx.y, bx.w, bx.h = gt[j,1:].tolist()
                                boxes_true.append(bx)
                            
                            total_true += len(boxes_true)
                            true_matched = np.zeros(len(boxes_true))
                            for b in boxes_pred:
                                if b.probs[np.argmax(b.probs)] < detection_threshold:
                                    continue
                                total_pred += 1.
                                for t,a in enumerate(boxes_true):
                                    if true_matched[t]:
                                        continue
                                    if box_iou(a, b) > 0.5 and np.argmax(a.probs) == np.argmax(b.probs):
                                        true_matched[t] = 1
                                        ok += 1.
                                        break
                            # You can visualize/save per image results with this:
                            #im = cv2.imread(img_path)
                            #im = yolo_draw_detections(boxes_pred, im, priors, classes, detection_threshold, nms_threshold)
                            #cv2.imshow('', im)
                            #cv2.waitKey(0)
                        inputs = []
                        img_paths = []
                    
                print 'total_true:',total_true,' total_pred:',total_pred,' ok:',ok
                p = 0. if total_pred == 0 else (ok/total_pred)
                r = ok/total_true
                print('Precission = ' + str(p))
                print('Recall     = ' + str(r))
                f = 0. if (p + r) == 0 else (2*p*r/(p + r))
                print('F-score    = '+str(f))
                print ('{} images predicted in {:.5f} seconds. {:.5f} fps').format(len(imfiles),
                       time.time() - start_time,
                        (len(imfiles)/(time.time() - start_time)))
    

            """
            if self.cf.problem_type == 'segmentation':
                # Compute Jaccard per class
                metrics_dict = dict(zip(self.model.metrics_names, test_metrics))
                I = np.zeros(self.cf.dataset.n_classes)
                U = np.zeros(self.cf.dataset.n_classes)
                jacc_percl = np.zeros(self.cf.dataset.n_classes)
                for i in range(self.cf.dataset.n_classes):
                    I[i] = metrics_dict['I'+str(i)]
                    U[i] = metrics_dict['U'+str(i)]
                    jacc_percl[i] = I[i] / U[i]
                    print ('   {:2d} ({:^15}): Jacc: {:6.2f}'.format(i,
                                                                     self.cf.dataset.classes[i],
                                                                     jacc_percl[i]*100))
                # Compute jaccard mean
                jacc_mean = np.nanmean(jacc_percl)
                print ('   Jaccard mean: {}'.format(jacc_mean))
            """
