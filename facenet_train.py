import os,cv2,json,time
import numpy as np
import tensorflow
import math
from inception_resnet_v2 import inference_VGG as VGG16
from inception_resnet_v2 import inference as inception_resnet_v2
# from inception_resnet_v2 import VGG16
# from validate_on_lfw import get_paths, read_pairs, evaluate
from sklearn import metrics#install scikit-learn
# from mask_wear_class import detect_mouth
import matplotlib.pyplot as plt

#----tensorflow version check
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.compat.v1.gfile as gfile
print("Tensorflow version: ",tf.__version__)


class NormDense(tf.keras.layers.Layer):

    def __init__(self, feature_num, classes=1000, output_name=''):
        super(NormDense, self).__init__()
        self.classes = classes
        self.w = self.add_weight(name='norm_dense_w', shape=(feature_num, self.classes),
                                 initializer='random_normal', trainable=True)
        self.output_name = output_name
        print("W shape = ", self.w.shape)

    # def build(self, input_shape):
    #     self.w = self.add_weight(name='norm_dense_w', shape=(input_shape[-1], self.classes),
    #                              initializer='random_normal', trainable=True)

    def call(self, inputs, **kwargs):
        norm_w = tf.nn.l2_normalize(self.w, axis=0)
        x = tf.matmul(inputs, norm_w, name=self.output_name)

        return x

class Facenet():
    def __init__(self,para_dict):
        #----var parsing
        train_img_dir = para_dict['train_img_dir']#CASIA
        test_img_dir = para_dict['test_img_dir']#LFW
        label_dict = para_dict['label_dict']
        get_image_type = para_dict['get_image_type']

        mask_img_dir = r".\mask_img"

        #----get label names to number dictionary
        # if label_dict is None:
        #     label_dict = self.__get_label_dict(train_img_dir)
        #     print(label_dict)
        #
        # class_num = len(label_dict.keys())

        # ----read training set paths and labels
        if isinstance(train_img_dir, list):
            for idx, img_dir in enumerate(train_img_dir):
                if idx == 0:
                    # ----get label names to number dictionary
                    label_dict = self.__get_label_dict(img_dir)
                    train_paths, train_labels = self.__get_paths_labels(img_dir, label_dict,type=get_image_type)
                elif idx > 0:
                    temp_paths, temp_labels = self.__get_paths_labels(img_dir, label_dict,type=get_image_type)
                    if get_image_type == 'extend':
                        train_paths = np.concatenate([train_paths, temp_paths], axis=0)
                        train_labels = np.concatenate([train_labels, temp_labels], axis=0)
                    elif get_image_type == 'append':
                        train_paths.extend(temp_paths)
                        train_labels.extend(temp_labels)

            # ----list to numpy array
            train_paths = np.array(train_paths)
            train_labels = np.array(train_labels)

        else:
            # ----get label names to number dictionary
            label_dict = self.__get_label_dict(train_img_dir)
            train_paths, train_labels = self.__get_paths_labels(train_img_dir, label_dict,type=get_image_type)
        class_num = len(label_dict.keys())
        print("train path shape:{}, train label shape:{}".format(train_paths.shape, train_labels.shape))
        print("class number:", class_num)

        #----read test set paths and labels
        # if test_img_dir is not None:
        #     test_paths, test_labels = self.__get_paths_labels(test_img_dir,label_dict)
        #     print("test path shape:{}, test label shape:{}".format(test_paths.shape, test_labels.shape))

        #----read mask png images
        mask_files = [file.path for file in os.scandir(mask_img_dir) if file.name.split(".")[-1] == 'png']
        len_mask = len(mask_files)
        if len_mask == 0:
            print("Error: no face mask PNG images in  ", mask_img_dir)
        else:
            print("mask image quantity:",len_mask)

        #----log update
        content = dict()
        content = self.log_update(content,para_dict)

        #----local var to global
        self.train_img_dir = train_img_dir
        self.test_img_dir = test_img_dir
        self.label_dict = label_dict
        self.train_paths = train_paths
        self.train_labels = train_labels
        self.class_num = class_num
        self.content = content
        self.get_image_type = get_image_type
        self.mask_files = mask_files
        # if test_img_dir is not None:
        #     self.test_img_dir = test_img_dir
        #     self.test_paths = test_paths
        #     self.test_labels = test_labels

    def model_init(self,para_dict):
        #----var parsing
        model_shape = para_dict['model_shape']#[N,H,W,C]
        infer_method = para_dict['infer_method']
        loss_method = para_dict['loss_method']
        opti_method = para_dict['opti_method']
        learning_rate = para_dict['learning_rate']
        save_dir = para_dict['save_dir']
        embed_length = para_dict['embed_length']

        #----tf_placeholder declaration
        tf_input = tf.placeholder(shape=model_shape,dtype=tf.float32,name='input')
        tf_keep_prob = tf.placeholder(dtype=tf.float32,name="keep_prob")
        tf_label_batch = tf.placeholder(shape=[None],dtype=tf.int32,name="label_batch")
        tf_phase_train = tf.placeholder(dtype=tf.bool,name="phase_train")

        #---inference selection
        if infer_method == "simple_resnet":
            prelogits = self.simple_resnet(tf_input,tf_keep_prob,self.class_num)
            embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        # elif infer_method == "inception_resnet_v1":
        #     prelogits, _ = inception_resnet_v1(tf_input, tf_keep_prob, phase_train=tf_phase_train,
        #       bottleneck_layer_size=embed_length, weight_decay=0.0, reuse=None)
        #     prelogits = tf.identity(prelogits,name='prelogits')
        #     embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        elif infer_method == "inception_resnet_v2":
            prelogits = inception_resnet_v2(tf_input, tf_keep_prob, phase_train=tf_phase_train,
              bottleneck_layer_size=embed_length, weight_decay=0.0, reuse=None)
            embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        elif infer_method == 'VGG16':
            prelogits = VGG16(tf_input, weight_decay=0.0, reuse=None)
            embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')


        #---loss selection
        if loss_method == "cross_entropy":
            output = tf.layers.dense(inputs=prelogits, units=self.class_num, activation=None, name="output")
            prediction = tf.nn.softmax(output,name="prediction")
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_label_batch,logits=output),
                                  name="loss")
        elif loss_method == "arc_loss":
            m1 = 1.0  # logits_margin1: 1.0  # m1: sphereface should >= 1
            m2 = 0.5  # logits_margin2: 0.2  # m2: cosineface should >= 0
            m3 = 0.0  # logits_margin3: 0.3  # m3: arcface    should >= 0
            s = 64.0  # logits_scale: 64.0

            norm_dense = NormDense(embed_length, self.class_num, output_name='prelogit')
            prelogit = norm_dense(embeddings)

            logit_cos = self.arcloss(embeddings, prelogit, tf_label_batch, m1, m2, m3, s)
            prediction = tf.nn.softmax(logit_cos, name='prediction')
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf_label_batch, logits=logit_cos), name='loss')

        #----optimizer selection
        if opti_method == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        elif opti_method == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

        # ----face detection init
        # self.detector = dlib.get_frontal_face_detector()
        # self.predictor = dlib.shape_predictor('src/models/shape_predictor_68_face_landmarks.dat')


        #----create the dir to save model weights(CKPT, PB)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        out_dir_prefix = os.path.join(save_dir,"model")
        saver = tf.train.Saver(max_to_keep=5)


        #----appoint PB node names
        pb_save_path = os.path.join(save_dir,"pb_model.pb")
        pb_save_list = ['prelogits','prediction',"embeddings"]


        #----create the log(JSON)
        count = 0
        for i in range(100):
            log_path = os.path.join(save_dir,"train_result_" + str(count) + ".json")
            if not os.path.exists(log_path):
                break
            count += 1
        print("log_path: ",log_path)
        self.content = self.log_update(self.content,para_dict)

        #----local var to global
        self.tf_input = tf_input
        self.tf_keep_prob = tf_keep_prob
        self.tf_label_batch = tf_label_batch
        self.tf_phase_train = tf_phase_train
        self.embeddings = embeddings
        self.optimizer = optimizer
        self.prediction = prediction
        self.out_dir_prefix = out_dir_prefix
        self.saver = saver
        self.pb_save_path = pb_save_path
        self.pb_save_list = pb_save_list
        self.log_path = log_path
        self.save_dir = save_dir
        self.model_shape = model_shape
        self.loss = loss

    def train(self,para_dict):
        #----var parsing
        epochs = para_dict['epochs']
        GPU_ratio = para_dict['GPU_ratio']
        batch_size = para_dict['batch_size']
        ratio=para_dict['ratio']
        select_num = para_dict['select_num']
        process_dict = para_dict['process_dict']
        aug_times = para_dict['aug_times']
        test_acc_threshold = 0.965

        #----local var
        train_loss_list = list()
        train_acc_list = list()
        test_loss_list = list()
        test_acc_list = list()
        epoch_time_list = list()
        img_quantity = 0
        aug_enable = False
        p_dict_1 = {'rdm_mask': False, 'rdm_crop': True, 'rdm_br': True, 'rdm_blur': True, 'rdm_flip': True,
                    'rdm_noise': False, 'rdm_angle': True}
        p_dict_2 = {'rdm_mask': True, 'rdm_crop': True, 'rdm_br': True, 'rdm_blur': True, 'rdm_flip': True,
                    'rdm_noise': False, 'rdm_angle': True}
        p_dict_3 = {'rdm_mask': True, 'rdm_crop': True, 'rdm_br': True, 'rdm_blur': True, 'rdm_flip': True,
                    'rdm_noise': False, 'rdm_angle': True}

        self.content = self.log_update(self.content, para_dict)

        #----ratio
        # if ratio <= 1.0:
        #     img_quantity = int(self.train_paths.shape[0] * ratio)
        # else:
        #     img_quantity = self.train_paths.shape[0]



        #----check if the augmentation(image processing) is enabled
        if isinstance(process_dict,dict):
            if True in process_dict.values():
                aug_enable = True
                batch_size = batch_size // aug_times  #the batch size must be integer!!


        #----calculate iterations of one epoch
        #train_ites = math.ceil(img_quantity / batch_size)
        # if self.test_img_dir is not None:
        #     test_ites = math.ceil(self.test_paths.shape[0] / batch_size)

        #----GPU setting
        config = tf.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True)
        if GPU_ratio is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio

        with tf.Session(config=config) as sess:
            #----tranfer learning check
            files = [file.path for file in os.scandir(self.save_dir) if file.name.split(".")[-1] == 'meta']
            if len(files) == 0:
                sess.run(tf.global_variables_initializer())
                print("no previous model param can be used!")
            else:
                check_name = files[-1].split("\\")[-1].split(".")[0]
                model_path = os.path.join(self.save_dir,check_name)
                self.saver.restore(sess,model_path)
                msg = "use previous model param:{}".format(model_path)
                print(msg)

            #----epoch training
            for epoch in range(epochs):
                #----record the start time
                d_t = time.time()

                train_loss = 0
                train_acc = 0
                test_loss = 0
                test_acc = 0

                #----training data process
                t_1 = time.time()
                if ratio is not None:
                    if ratio <= 1.0:
                        img_quantity = int(self.train_paths.shape[0] * ratio)
                    else:
                        img_quantity = self.train_paths.shape[0]
                    # ----shuffle
                    indice = np.random.permutation(self.train_paths.shape[0])
                    self.train_paths = self.train_paths[indice]
                    self.train_labels = self.train_labels[indice]
                    train_paths_ori = self.train_paths[:img_quantity]
                    train_labels_ori = self.train_labels[:img_quantity]
                elif select_num is not None:
                    select_num = int(select_num)
                    if select_num >= 1:
                        #----reset paths and lables
                        train_paths_ori = list()
                        train_labels_ori = list()
                        #----select images from each folder
                        for paths,labels in zip(self.train_paths,self.train_labels):
                            np.random.shuffle(paths)
                            num = np.minimum(len(paths), select_num)
                            train_paths_ori.extend(paths[:num])
                            train_labels_ori.extend(labels[:num])
                        #----list to np array
                        train_paths_ori = np.array(train_paths_ori)
                        train_labels_ori = np.array(train_labels_ori)
                        #----shuffle
                        indice = np.random.permutation(train_paths_ori.shape[0])
                        train_paths_ori = train_paths_ori[indice]
                        train_labels_ori = train_labels_ori[indice]

                        #----calculate the image quantity
                        img_quantity = train_paths_ori.shape[0]
                    else:
                        print("select number is under 1")
                        raise ValueError
                t_1 = time.time() - t_1
                print("\nTime of preparing the training data: ",t_1)

                # ----calculate iterations of one epoch
                train_ites = math.ceil(img_quantity / batch_size)

                # ----info display
                print("img_quantity:", img_quantity)
                if aug_enable is True:
                    print("aug_enable is True, the data quantity of one epoch is {} times".format(aug_times))


                # if aug_enable is True:
                #     train_paths_aug = train_paths_ori[::-1]
                #     train_labels_aug = train_labels_ori[::-1]

                #----do optimizers(training by iteration)
                for index in range(train_ites):
                    #----get image start and end numbers
                    num_start = index * batch_size
                    num_end = np.minimum(num_start + batch_size, train_paths_ori.shape[0])
                    paths = train_paths_ori[num_start:num_end]
                    labels = train_labels_ori[num_start:num_end]

                    # ----get 4-D data
                    if aug_enable is True:
                        for i in range(aug_times):
                            if i == 0:
                                temp = self.get_4D_data(paths, self.model_shape[1:], process_dict=None)
                                batch_data = temp
                                batch_labels = labels
                            elif i == 1:
                                temp = self.get_4D_data(paths, self.model_shape[1:], process_dict=p_dict_1)
                            elif i == 2:
                                temp = self.get_4D_data(paths, self.model_shape[1:], process_dict=p_dict_2)
                            elif i == 3:
                                temp = self.get_4D_data(paths, self.model_shape[1:], process_dict=p_dict_3)

                            if i > 0:
                                batch_data = np.concatenate([batch_data, temp], axis=0)
                                batch_labels = np.concatenate([batch_labels, labels], axis=0)

                        indice = np.random.permutation(batch_data.shape[0])
                        batch_data = batch_data[indice]
                        batch_labels = batch_labels[indice]
                    else:
                        batch_data = self.get_4D_data(train_paths_ori[num_start:num_end], self.model_shape[1:])
                        batch_labels = train_labels_ori[num_start:num_end]
                    #----images display(test code)
                    #set batch size = 4 and not shuffle
                    # print("batch_data shape:", batch_data.shape)
                    # for i in range(aug_times):
                    #     plt.subplot(1, aug_num, i + 1)
                    #     plt.imshow(batch_data[i])
                    #     plt.axis('off')
                    #
                    # plt.show()
                    
                    #----put all data to tf placeholders
                    feed_dict = {self.tf_input:batch_data,
                                 self.tf_label_batch:batch_labels,
                                 self.tf_keep_prob:0.8,
                                 self.tf_phase_train:True}
                    #----session run
                    sess.run(self.optimizer,feed_dict=feed_dict)
                    #----evaluation(training set)
                    feed_dict[self.tf_keep_prob] = 1.0
                    feed_dict[self.tf_phase_train] = False
                    loss_temp, predict_temp = sess.run([self.loss, self.prediction], feed_dict=feed_dict)

                    # ----calculate the loss and accuracy
                    train_loss += loss_temp
                    train_acc += self.evaluation(predict_temp, batch_labels)

                train_loss /= train_ites
                train_acc /= img_quantity
                if aug_enable is True:#divided by aug_times because the training data quantity is multiplied by aug_times
                    train_acc /= aug_times

                #-----testing set(LFW) evaluation
                # if self.test_img_dir is not None:
                #     test_acc = self.eval_on_lfw(sess, feed_dict, self.test_img_dir, batch_size=batch_size)

                #print("train_loss:{}, train_acc:{}".format(train_loss,train_acc))

                #----evaluation(test set)
                # if self.test_img_dir is not None:
                #     for index in range(test_ites):
                #         # ----get image start and end numbers
                #         num_start = index * batch_size
                #         num_end = np.minimum(num_start + batch_size, self.test_paths.shape[0])
                #
                #         batch_data = self.get_4D_data(self.test_paths[num_start:num_end], self.model_shape[1:])
                #
                #         # ----put all data to tf placeholders
                #         feed_dict = {self.tf_input: batch_data,
                #                      self.tf_label_batch: self.test_labels[num_start:num_end],
                #                      self.tf_keep_prob: 1.0}
                #
                #         # ----session run
                #         loss_temp, predict_temp = sess.run([self.loss, self.prediction], feed_dict=feed_dict)
                #
                #         # ----calculate the loss and accuracy
                #         test_loss += loss_temp
                #         test_acc += self.evaluation(predict_temp, self.test_labels[num_start:num_end])
                #
                #     test_loss /= test_ites
                #     test_acc /= self.test_paths.shape[0]
                #     #print("test_loss:{}, test_acc:{}".format(test_loss, test_acc))

                #----save ckpt, pb files
                model_save_path = self.saver.save(sess,self.out_dir_prefix,global_step=epoch)
                print("save model CKPT to ",model_save_path)

                # graph = tf.get_default_graph().as_graph_def()
                # output_graph_def = tf.graph_util.convert_variables_to_constants(sess,graph,self.pb_save_list)
                # with tf.gfile.GFile(self.pb_save_path,'wb')as f:
                #     f.write(output_graph_def.SerializeToString())
                print("save PB file to ",self.pb_save_path)

                if test_acc > test_acc_threshold:
                    pb_save_path = "pb_{}.pb".format(np.round(float(test_acc*100),1))
                    pb_save_path = os.path.join(self.save_dir,pb_save_path)
                    graph = tf.get_default_graph().as_graph_def()
                    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph, self.pb_save_list)
                    with tf.gfile.GFile(pb_save_path, 'wb')as f:
                        f.write(output_graph_def.SerializeToString())

                    test_acc_threshold = test_acc

                #----record the end time
                d_t = time.time() - d_t

                #----save results in the log file
                train_loss_list.append(float(train_loss))
                train_acc_list.append(float(train_acc))
                if self.test_img_dir is not None:
                    #test_loss_list.append(float(test_loss))
                    test_acc_list.append(float(test_acc))

                self.content["train_loss_list"] = train_loss_list
                self.content["train_acc_list"] = train_acc_list
                if self.test_img_dir is not None:
                    #self.content["test_loss_list"] = test_loss_list
                    self.content["test_acc_list"] = test_acc_list

                epoch_time_list.append(d_t)
                self.content['ave_epoch_time'] = float(np.average(epoch_time_list))

                with open(self.log_path, 'w') as f:
                    json.dump(self.content,f)

                print("save the log file in ",self.log_path)



                #----display training results
                print("\nEpoch: ",epoch)
                print("training loss:{}, accuracy:{}".format(train_loss,train_acc))
                if self.test_img_dir is not None:
                    print("test set accuracy:{}".format( test_acc))

                print("Epoch time consumption:",d_t)
                
            pb_save_path = "pb_{}.pb".format(np.round(float(train_acc*100),1))
            pb_save_path = os.path.join(self.save_dir,pb_save_path)
            graph = tf.get_default_graph().as_graph_def()
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph, self.pb_save_list)
            with tf.gfile.GFile(pb_save_path, 'wb')as f:
                f.write(output_graph_def.SerializeToString())
    #----loss functions
    def arcloss(self, x, normx_cos, labels, m1, m2, m3, s):
        norm_x = tf.norm(x, axis=1, keepdims=True)
        print("norm_x shape = ", norm_x.shape)
        cos_theta = normx_cos / norm_x
        theta = tf.acos(cos_theta)
        mask = tf.one_hot(labels, depth=normx_cos.shape[-1])
        zeros = tf.zeros_like(mask)
        cond = tf.where(tf.greater(theta * m1 + m3, math.pi), zeros, mask)
        cond = tf.cast(cond, dtype=tf.bool)
        m1_theta_plus_m3 = tf.where(cond, theta * m1 + m3, theta)
        cos_m1_theta_plus_m3 = tf.cos(m1_theta_plus_m3)
        prelogits = tf.where(cond, cos_m1_theta_plus_m3 - m2, cos_m1_theta_plus_m3) * s

        return prelogits

    #----functions
    def log_update(self,content,para_dict):
        for key, value in para_dict.items():
            content[key] = value

        return content

    def evaluation(self,predictions,labels):
        count = 0
        for i in range(predictions.shape[0]):
            if np.argmax(predictions[i]) == labels[i]:
                count += 1

        return count

    def get_4D_data(self,paths, img_shape, process_dict=None):
        # ----var
        re_array = []
        processing_enable = False
        x_range = 10
        y_range = 20
        flip_list = [1, 0]
        kernel_list = [1, 3, 5, 7]

        # ----create default np array
        batch_dim = [len(paths)]
        batch_dim.extend(img_shape)
        batch_data = np.zeros(batch_dim, dtype=np.float32)

        # ----check process_dict
        if isinstance(process_dict, dict):
            if len(process_dict) > 0:
                processing_enable = True  # image processing is enabled

        for idx,path in enumerate(paths):
            img = cv2.imread(path)
            if img is None:
                print("read failed:", path)
            else:
                # ----image processing
                if processing_enable is True:
                    if 'rdm_crop' in process_dict.keys():
                        if process_dict['rdm_crop'] is True:
                            #img = cv2.resize(img,(width_rdm_crop,height_rdm_crop))

                            # ----Find a random point
                            x_start = np.random.randint(x_range)
                            y_start = np.random.randint(y_range)

                            # ----From the random point, crop the image
                            img = img[y_start:, x_start:, :]
                    if 'rdm_br' in process_dict.keys():
                        if process_dict['rdm_br'] is True:
                            mean_br = np.mean(img)
                            if mean_br == float("Nan"):
                                print(f'mean_br{mean_br} of path: {path}')
                            br_factor = np.random.randint(mean_br * 0.7, mean_br * 1.3)
                            img = np.clip(img / mean_br * br_factor, 0,
                                          255)  # the multiplication makes the numeric type become floating
                            img = img.astype(np.uint8)  # transform the numeric type to unsigned integer 8(UINT8)
                    # if 'rdm_mask' in process_dict.keys():
                    #     if process_dict['rdm_mask'] is True:
                    #         x_min, x_max, y_min, y_max, size = detect_mouth(img, self.detector, self.predictor)
                    #         if size is not None:
                    #             # ----random selection of face mask
                    #             item_path = np.random.choice(self.mask_files)

                    #             # ----face mask process
                    #             item_img = cv2.imread(item_path, cv2.IMREAD_UNCHANGED)
                    #             # item_img = mask_paths[which]
                    #             #print(item_img.shape)
                    #             item_img = cv2.resize(item_img, size)
                    #             item_img_rgb = item_img[:, :, :3]
                    #             # item_img_rgb = item_img_rgb[:,:,::-1]#transform the color format to RGB

                    #             item_alpha_ch = item_img[:, :, 3]
                    #             _, item_mask = cv2.threshold(item_alpha_ch, 220, 255, cv2.THRESH_BINARY)
                    #             img_item = cv2.bitwise_and(item_img_rgb, item_img_rgb, mask=item_mask)

                    #             # ----mouth part process
                    #             roi = img[y_min:y_min + size[1], x_min:x_min + size[0]]
                    #             item_mask_inv = cv2.bitwise_not(item_mask)
                    #             roi = cv2.bitwise_and(roi, roi, mask=item_mask_inv)

                    #             # ----addition of mouth and face mask
                    #             dst = cv2.add(roi, img_item)
                    #             img[y_min: y_min + size[1], x_min:x_min + size[0]] = dst

                    if 'rdm_blur' in process_dict.keys():
                        if process_dict['rdm_blur'] is True:
                            kernel = tuple(np.random.choice(kernel_list, size=2))
                            #print("kernel:", kernel)
                            img = cv2.GaussianBlur(img, kernel, 0, 0)

                    if 'rdm_flip' in process_dict.keys():
                        if process_dict['rdm_flip'] is True:
                            flip_type = np.random.choice(flip_list)
                            if flip_type == 1:
                                img = cv2.flip(img, flip_type)
                    if 'rdm_noise' in process_dict.keys():
                        if process_dict['rdm_noise'] is True:
                            uniform_noise = np.empty((img.shape[0], img.shape[1]), dtype=np.uint8)
                            cv2.randu(uniform_noise, 0, 255)
                            ret, impulse_noise = cv2.threshold(uniform_noise, 240, 255, cv2.THRESH_BINARY_INV)
                            img = cv2.bitwise_and(img, img, mask=impulse_noise)
                    if 'rdm_angle' in process_dict.keys():
                        if process_dict['rdm_angle'] is True:
                            angle = np.random.randint(-15, 15)
                            img = cv2.resize(img, (img_shape[1], img_shape[0]))
                            #print(img.shape)
                            h, w = img.shape[:2]
                            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                            img = cv2.warpAffine(img, M, (h, w))
                # ----
                img = cv2.resize(img, (img_shape[1], img_shape[0]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_data[idx] = img

        #re_array = np.array(re_array)
        x = batch_data/255
        return batch_data / 255

    def get_4D_data_2(self,paths, img_shape, process_dict=None):
        # ----var
        random_flip = False
        random_brightness = False
        random_crop = False
        random_angle = False
        random_noise = False
        flip_list = [1, 0]

        # ----create default np array
        batch_dim = [len(paths)]
        batch_dim.extend(img_shape)
        batch_data = np.zeros(batch_dim, dtype=np.float32)

        # ----update var
        if isinstance(process_dict, dict):
            if 'random_flip' in process_dict.keys():
                random_flip = process_dict['random_flip']
            if 'random_brightness' in process_dict.keys():
                random_brightness = process_dict['random_brightness']
            if 'random_crop' in process_dict.keys():
                random_crop = process_dict['random_crop']
            if 'random_angle' in process_dict.keys():
                random_angle = process_dict['random_angle']
            if 'random_noise' in process_dict.keys():
                random_noise = process_dict['random_noise']

        for idx, path in enumerate(paths):
            img = cv2.imread(path)
            if img is None:
                print("read failed:", path)
            else:
                img = cv2.resize(img, (img_shape[1], img_shape[0]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # ----random brightness
                if random_brightness is True:
                    mean_br = np.mean(img)
                    br_factor = np.random.randint(mean_br * 0.7, mean_br * 1.3)
                    img = np.clip(img / mean_br * br_factor, 0, 255)
                    img = img.astype(np.uint8)

                # ----random crop
                if random_crop is True:
                    # ----resize the image 1.15 times
                    img = cv2.resize(img, None, fx=1.15, fy=1.15)

                    # ----Find a random point
                    y_range = img.shape[0] - img_shape[0]
                    x_range = img.shape[1] - img_shape[1]
                    x_start = np.random.randint(x_range)
                    y_start = np.random.randint(y_range)

                    # ----From the random point, crop the image
                    img = img[y_start:y_start + img_shape[0], x_start:x_start + img_shape[1], :]

                # ----random flip
                if random_flip is True:
                    flip_type = np.random.choice(flip_list)
                    if flip_type == 1:
                        img = cv2.flip(img, flip_type)

                # ----random angle
                if random_angle is True:
                    angle = np.random.randint(-60, 60)
                    height, width = img.shape[:2]
                    M = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
                    img = cv2.warpAffine(img, M, (width, height))

                # ----random noise
                if random_noise is True:
                    uniform_noise = np.empty((img.shape[0], img.shape[1]), dtype=np.uint8)
                    cv2.randu(uniform_noise, 0, 255)
                    ret, impulse_noise = cv2.threshold(uniform_noise, 240, 255, cv2.THRESH_BINARY_INV)
                    img = cv2.bitwise_and(img, img, mask=impulse_noise)

                batch_data[idx] = img

        batch_data /= 255
        return batch_data

    # def eval_on_lfw(self, sess, feed_dict, lfw_dir, batch_size=12):
    #     # ----Read the file containing the pairs used for testing
    #     time_eval = time.time()
    #     lfw_pairs_path = r'pairs.txt'
    #     pairs = read_pairs(os.path.expanduser(lfw_pairs_path))

    #     # ----Get the paths for the corresponding images
    #     paths, actual_issame = get_paths(os.path.expanduser(lfw_dir), pairs)

    #     # ----collect all embeddings
    #     iterations = math.ceil(len(paths) / batch_size)

    #     for i in range(iterations):
    #         n_start = i * batch_size
    #         n_end = np.minimum(i * batch_size + batch_size, len(paths))

    #         batch_data = self.get_4D_data(paths[n_start:n_end], self.model_shape[1:])

    #         feed_dict[self.tf_input] = batch_data
    #         sess_out = sess.run(self.embeddings, feed_dict=feed_dict)
    #         if i == 0:
    #             embeddings = sess_out
    #         else:
    #             embeddings = np.concatenate([embeddings, sess_out], axis=0)

    #     tpr, fpr, accuracy, val, val_std, far = evaluate(embeddings, actual_issame, nrof_folds=10,
    #                                                      distance_metric=0,
    #                                                      subtract_mean=False)

    #     print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    #     print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    #     auc = metrics.auc(fpr, tpr)
    #     print('Area Under Curve (AUC): %1.3f' % auc)

    #     time_eval = time.time() - time_eval
    #     print("Eval on LFW time:", time_eval)

    #     return np.mean(accuracy)

    def __get_label_dict(self,img_dir):
        label_dict = dict()
        count = 0
        for obj in os.scandir(img_dir):
            if obj.is_dir():
                label_dict[obj.name] = count
                count += 1
        if count == 0:
            print("No dir in the ",img_dir)
            return None
        else:
            return label_dict

    def __get_paths_labels(self,img_dir,label_dict,type='extend'):
        #----var
        img_format = {'png', 'jpg', 'bmp'}
        re_paths = list()
        re_labels = list()

        #----read dirs
        dirs = [obj.path for obj in os.scandir(img_dir) if obj.is_dir()]
        if len(dirs) == 0:
            print("No dirs in the ",img_dir)
        else:
            #-----read paths of each dir
            for dir_path in dirs:
                path_temp = [file.path for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format]
                if len(path_temp) == 0:
                    print("No images in the ",dir_path)
                else:
                    #----get the label number from class name
                    label_num = dir_path.split("\\")[-1]
                    label_num = label_dict[label_num]
                    #----create the label array
                    label_temp = np.ones(len(path_temp),dtype=np.int32) * label_num

                    #----collect paths and labels
                    if type == 'append':
                        re_paths.append(path_temp)
                        re_labels.append(label_temp)
                    elif type == 'extend':
                        re_paths.extend(path_temp)
                        re_labels.extend(label_temp)

            #----list to numpy array
            # re_paths = np.array(re_paths)
            # re_labels = np.array(re_labels)

            #----shuffle
            # indice = np.random.permutation(re_paths.shape[0])
            # re_paths = re_paths[indice]
            # re_labels = re_labels[indice]

        return re_paths, re_labels

    #----models
    def resnet_block(self,input_x, k_size=3,filters=32):
        net = tf.layers.conv2d(
            inputs=input_x,
            filters = filters,
            kernel_size=[k_size,k_size],
            kernel_regularizer=tf.keras.regularizers.l2(0.1),
            padding="same",
            activation=tf.nn.relu
        )
        net = tf.layers.conv2d(
            inputs=net,
            filters=filters,
            kernel_size=[k_size, k_size],
            kernel_regularizer=tf.keras.regularizers.l2(0.1),
            padding="same",
            activation=tf.nn.relu
        )

        net_1 = tf.layers.conv2d(
            inputs=input_x,
            filters=filters,
            kernel_size=[k_size, k_size],
            kernel_regularizer=tf.keras.regularizers.l2(0.1),
            padding="same",
            activation=tf.nn.relu
        )

        add = tf.add(net,net_1)

        add_result = tf.nn.relu(add)

        return add_result

    def simple_resnet(self,tf_input,tf_keep_prob,class_num):
        net = self.resnet_block(tf_input,k_size=3,filters=16)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2,2], strides=2)
        print("pool_1 shape:",net.shape)

        net = self.resnet_block(net, k_size=3, filters=32)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        print("pool_2 shape:", net.shape)

        net = self.resnet_block(net, k_size=3, filters=48)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        print("pool_3 shape:", net.shape)

        net = self.resnet_block(net, k_size=3, filters=64)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        print("pool_4 shape:", net.shape)

        #----flatten
        net = tf.layers.flatten(net)
        print("flatten shape:",net.shape)

        #----dropout
        net = tf.nn.dropout(net,keep_prob=tf_keep_prob)

        #----FC
        net = tf.layers.dense(inputs=net,units=128,activation=tf.nn.relu)
        print("FC shape:",net.shape)

        #----output
        # output = tf.layers.dense(inputs=net,units=class_num,activation=None)
        # print("output shape:",output.shape)

        return net


if __name__ == "__main__":
    # train_img_dir = r"D:\CASIA\CASIA-WebFace"
    train_img_dir = [r"train_dir"
                     ]
    test_img_dir = r"test_dir"
    label_dict = None
    embed_length = 128
    get_image_type = 'append'#append #extend

    para_dict = {"train_img_dir":train_img_dir,"test_img_dir":test_img_dir,"label_dict":label_dict,'get_image_type':get_image_type}

    cls = Facenet(para_dict)

    model_shape = [None,112,112,3]#at least[None,80,80,3] if you use inception_resnet_v1
    infer_method = "inception_resnet_v2" #inception_resnet_v1
    loss_method = "cross_entropy" #"arc_loss"#"cross_entropy"
    opti_method = "adam" #adagrad #adam
    learning_rate = 5e-4
    save_dir = r"save_dir"

    para_dict = {"model_shape":model_shape,"infer_method":infer_method,"loss_method":loss_method,
                 "opti_method":opti_method,'learning_rate':learning_rate,"save_dir":save_dir,'embed_length':embed_length}
    cls.model_init(para_dict)

    epochs = 100
    GPU_ratio = None#0.1 ~ 0.9
    batch_size = 48#depends on your GPU resource. Set <= 96 if 6GB GPU using inception_resnet_v1
    ratio = None
    select_num = 2


    random_flip = True
    random_brightness = True
    random_crop = True
    random_angle = True
    random_noise = True
    aug_times = 4

    process_dict = {"rdm_flip":random_flip,'rdm_br':random_brightness,'rdm_crop':random_crop,'rdm_angle':random_angle,
                    'rdm_noise':random_noise}
    if True in process_dict.values():
        pass
    else:
        process_dict = None
    para_dict = {'epochs':epochs, "GPU_ratio":GPU_ratio, "batch_size":batch_size,"ratio":ratio,'process_dict':process_dict,
                 'select_num':select_num,'aug_times':aug_times}

    cls.train(para_dict)
