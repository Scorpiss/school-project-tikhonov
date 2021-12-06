from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np


class NeuralNetwork:

    def __init__(self,
                 alpha, maxpool_1, hidden_1_layer, hidden_2_layer, drop_1, output_layer
                 ):

        self.lr = alpha
        self.mp_1 = (maxpool_1[0], maxpool_1[1])
        self.mp_1_s = maxpool_1[-1]

        input_layer = int((224 - self.mp_1[1])//self.mp_1_s + 1)**2*3
        
        self.w_in_1 = self.init_weights((hidden_1_layer, input_layer))
        self.w_1_2 = self.init_weights((hidden_2_layer, hidden_1_layer))
        self.w_2_out = self.init_weights((output_layer, hidden_2_layer))
        self.bl_1 = np.zeros((hidden_1_layer, 1))
        self.bl_2 = np.zeros((hidden_2_layer, 1))
        self.bl_3 = np.zeros((output_layer, 1))
        
        self.drp_1 = drop_1

        self.history = {
            "epochs": 0,
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
    def info(self):
        print(f"Learning rate: {self.lr}")
        print(f"MaxPooling: {self.mp_1} with stride {self.mp_1_s}")
        print(f"Layers:\n#1 {self.w_in_1.shape}\n#2 {self.w_1_2.shape}\n#3 {self.w_2_out.shape}")
        try:
            print(f"Dropout {self.drp_1}")
        except:
            pass
            
    def maxpool2d(self, input_matrix: np.array, shape_: tuple, stride: int = 1):
        m, n = shape_
        if m == n:
            n_c, x, y = input_matrix.shape
            x = int((x - m)//stride + 1)
            y = int((y - m)//stride + 1)
            
            output = np.zeros((n_c, y, x))
            for c in range(n_c):
                for k in range(y):
                    for l in range(x):
                        output[c, k, l] = np.max(input_matrix[c, k:k+m, l:l+m])
        else:
            output = None
        return output

    def init_weights(self, matrix_shape: tuple):
        # Инициализация нормализированных связей между слоями
        return np.random.normal(0.0, pow(matrix_shape[1], -0.5), (matrix_shape[0], matrix_shape[1]))
    
    def BCE(self, y, t):
        return - np.mean(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))

    def sigmoid(self, x):
        x = np.array(x) if type(x) == list else x
        return 1 / (1 + np.exp(-x))

    def deriv_sigmoid(self, x):
        return x * (1.0 - x)
    
    def dropout(self, inputs, prob):
        binary_val = np.random.rand(inputs.shape[0], inputs.shape[1]) < prob
        res = np.multiply(inputs, binary_val) / prob
        return res, binary_val

    def deriv_dropout(self, mask, w, prob): return (np.multiply(mask, w) / prob)
    
    def query(self, input_data, train = False):
        #############################################
        ########## Прямое распространение ###########
        #############################################
        
        # MaxPooling2D
        mpool_1 = self.maxpool2d(input_data, self.mp_1, self.mp_1_s)

        # 3D to Flatten
        inputs_flatten = mpool_1.reshape(mpool_1.shape[1] ** 2 * 3, 1)

        # Полносвязные слои
        I_O = self.sigmoid(self.w_in_1.dot(inputs_flatten) + self.bl_1)
        
        H_O = self.sigmoid(self.w_1_2.dot(I_O) + self.bl_2)
        if train:
            H_O, D_1 = self.dropout(H_O, self.drp_1)
        
        F_O = self.sigmoid(self.w_2_out.dot(H_O) + self.bl_3)
        
        return F_O if not train else [inputs_flatten, I_O, H_O, D_1, F_O]

    def train(self, input_data, target):
        
        # Прямое распространение
        inputs_flatten, I_O, H_O, D_1, F_O = \
                                self.query(input_data, train=True)

        #############################################
        ############## Расчёт  ошибок ###############
        #############################################

        # Ошибки полносвязных слоёв
        er_out = F_O - target
        dbl_3 = np.sum(er_out, axis=1).reshape(self.bl_3.shape)
        
        er_1_2 = self.w_2_out.T.dot(er_out)
        er_1_2 = self.deriv_dropout(D_1, er_1_2, self.drp_1) # Производная Dropout
        dbl_2 = np.sum(er_1_2, axis=1).reshape(self.bl_2.shape)

        er_in_1 = self.w_1_2.T.dot(er_1_2)
        dbl_1 = np.sum(er_in_1, axis=1).reshape(self.bl_1.shape)

        #############################################
        ############# Обновление  весов #############
        #############################################
        forward_data = [I_O, H_O, F_O]
        er_data = [
                    (er_out, dbl_3),
                    (er_1_2, dbl_2),
                    (er_in_1, dbl_1)
                  ]

        self.update_weights(er_data, forward_data, inputs_flatten)
        
        return F_O

    def update_weights(self, errors, data_f, input_data):
        self.w_2_out = self.w_2_out - (self.lr * np.dot(errors[0][0] * self.deriv_sigmoid(data_f[2]), data_f[1].T))
        self.bl_3 = self.bl_3 - (self.lr * errors[0][1])

        self.w_1_2 = self.w_1_2 - (self.lr * np.dot(errors[1][0] * self.deriv_sigmoid(data_f[1]), data_f[0].T))
        self.bl_2 = self.bl_2 - (self.lr * errors[1][1])
        
        self.w_in_1 = self.w_in_1 - (self.lr * np.dot(errors[2][0] * self.deriv_sigmoid(data_f[0]), input_data.T))
        self.bl_1 = self.bl_1 - (self.lr * errors[2][1])
        
        
    def fit(self, inputs_data, targets_data, epochs, val_per=0.2, upd=10):
        assert inputs_data.shape[0] == targets_data.shape[0], f"Different length ({inputs_data.shape[0]} != {targets_data.shape[0]})"
        sl_end = inputs_data.shape[0] - int(inputs_data.shape[0] * val_per)
        train_data_slice, train_targets_slice = inputs_data[:sl_end], targets_data[:sl_end]
        test_data_slice, test_targets_slice = inputs_data[sl_end:], targets_data[sl_end:]
        
        for ep in range(epochs):
            accur = [0, 0]
            loss = [0, 0]
            print(f"Epoch {ep+1}/{epochs}")
            self.history["epochs"] += 1
            with trange(len(inputs_data)) as bar:
                for indx in range(len(train_data_slice)):
                    bar.set_description(f"Training {indx+1}")
                    accur[0] += 1
                    loss[0] += 1
                    
                    pred = self.train(train_data_slice[indx], train_targets_slice[indx])
                    pred_ = 0.99 if pred >= 0.6 else 0.01
                    
                    accur[1] = accur[1] + 1 if train_targets_slice[indx] == pred_ else accur[1] + 0
                    loss[1] += self.BCE(train_targets_slice[indx], pred_)
                    if (indx+1) % upd == 0:
                        bar.set_postfix(accuracy=f"{accur[1]/accur[0]:1.5}", loss=f"{loss[1]/loss[0]:.5}")
                    bar.update()

                loss = float(f"{loss[1]/loss[0]:.5}")
                accur = float(f"{accur[1]/accur[0]:1.5}")
                accur_t = [0, 0]
                loss_t = [0, 0]
                for indx in range(len(test_data_slice)):
                    bar.set_description(f"Validation {indx+1}")
                    accur_t[0] += 1
                    loss_t[0] += 1
                    
                    pred = self.query(test_data_slice[indx])
                    pred_ = 0.99 if pred >= 0.6 else 0.01
                    
                    accur_t[1] = accur_t[1] + 1 if test_targets_slice[indx] == pred_ else accur_t[1] + 0
                    loss_t[1] += self.BCE(test_targets_slice[indx], pred_)
                    if (indx+1) % 2 == 0:
                        bar.set_postfix(accuracy=accur, loss=loss, 
                                        val_accuracy=f"{accur_t[1]/accur_t[0]:1.5}", val_loss=f"{loss_t[1]/loss_t[0]:.5}")
                    bar.update()
                bar.set_description(f"Epoch completed")
                bar.ncols = 0
            self.history["train_loss"].append(loss)
            self.history["train_acc"].append(accur)
            self.history["val_loss"].append(float(f"{loss_t[1]/loss_t[0]:.5}"))
            self.history["val_acc"].append(float(f"{accur_t[1]/accur_t[0]:1.5}"))
        
    def build_graph(self, save=False, loss=True, acc=True, train=True, test=True):
        EP = self.history["epochs"]
        plt.style.use("fivethirtyeight")
        plt.figure()
        if train:
            plt.plot(np.arange(0, EP), self.sigmoid(self.history["train_loss"]), label="train_loss") if loss else ""
        if test:
            plt.plot(np.arange(0, EP), self.sigmoid(self.history["val_loss"]), label="val_loss") if loss else ""
        if train:
            plt.plot(np.arange(0, EP), self.history["train_acc"], label="train_acc") if acc else ""
        if test:
            plt.plot(np.arange(0, EP), self.history["val_acc"], label="val_acc") if acc else ""
        
        plt.title("История обучения модели (Loss и Accuracy)")
        plt.xlabel(f"№ Эпохи (всего {EP})")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        if save:
            plt.savefig(save, dpi = 200, bbox_inches='tight')
    
    def validation(self, val_data, val_targets):
        accur = [0, 0]
        with trange(len(val_data)) as bar:
            bar.set_description("Validation process started")
            for indx in range(len(val_data)):
                bar.set_description("Validation process")
                accur[0] += 1
                pred = self.query(val_data[indx])
                pred_ = 0.99 if pred >= 0.6 else 0.01
                accur[1] += 1 if pred_ == val_targets[indx] else 0
                if (indx+1) % 20:
                    bar.set_postfix(accuracy=f"{accur[1]/accur[0]:.6}")
                bar.update()
            bar.set_description("Validation completed")
