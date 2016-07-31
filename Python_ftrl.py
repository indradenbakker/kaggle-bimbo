from csv import DictReader
from math import sqrt, log, expm1
from datetime import datetime

# TL; DR, the main training process starts on line: 250,
# you may want to start reading the code from there


##############################################################################
# parameters #################################################################
##############################################################################

# A, paths
train = 'train_extended_producto_c_f_s.csv'               # path to training file
test = 'test.csv'                 # path to testing file
submission = 'submission_nonval.csv'  # path of to be outputted submission file

# B, model
alpha = .01 #.02  # learning rate
beta = 0.1 # 1.   # smoothing parameter for adaptive learning rate
L1 =  0. # 0.     # L1 regularization, larger value means more regularized
L2 = 0.1 # 1.     # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 33 # 23             # number of weights to use
interaction = True     # whether to enable poly2 feature interactions

# Results standard settings: Epoch 0 finished, validation RMSLE: 0.496561, elapsed time: 0:43:24.419631
#1 beta to 0.8 => Epoch 0 finished, validation RMSLE: 0.496098, elapsed time: 0:42:51.744152
#2 keep 1 + L2 to 3 => Epoch 0 finished, validation RMSLE: 0.496143, elapsed time: 0:42:46.694739                                  
#3 keep 1 + L2 to 0.5 => Epoch 0 finished, validation RMSLE: 0.496075, elapsed time: 0:43:35.642031 
#4 keep 3 + L1 to 0.5 => Epoch 0 finished, validation RMSLE: 0.496632, elapsed time: 0:43:37.684836                               #5 keep 3 + beta to 0.2 => Epoch 0 finished, validation RMSLE: 0.495083, elapsed time: 0:43:17.261509                             

#6 keep 5 + D to 2 ** 25 => Epoch 0 finished, validation RMSLE: 0.474776, elapsed time: 0:43:10.890549
#Epoch 4 finished, validation RMSLE: 0.464027, elapsed time: 3:39:29.982226                                                      
#=> 0.47595
#diff: 0,011923

#7 keep 6 + D to 2 ** 26 + alpha to .01 => Epoch 0 finished, validation RMSLE: 0.481837, elapsed time: 0:44:55.345911             #Epoch 5 finished, validation RMSLE: 0.459646, elapsed time: 4:23:07.208881
#=> 0.47038
#diff: 0,010734

#8 keep 6 + D to 2 ** 27 => Epoch 0 finished, validation RMSLE: 0.479033, elapsed time: 0:44:17.743013                           

#9 keep 6 + D to 2 ** 29 => Epoch 0 finished, validation RMSLE: 0.476613, elapsed time: 0:45:13.654363                           #10 keep 9 + L2 to 0.6 => Epoch 0 finished, validation RMSLE: 0.476629, elapsed time: 0:46:59.412419                             

#11 keep 9 => 
#Epoch 0 finished, validation RMSLE: 0.476613, elapsed time: 0:44:58.674604
#Epoch 1 finished, validation RMSLE: 0.464618, elapsed time: 1:29:57.552301                                                        
#Epoch 2 finished, validation RMSLE: 0.459007, elapsed time: 2:14:20.410751                                                        
#Epoch 3 finished, validation RMSLE: 0.456717, elapsed time: 2:56:00.923282                                                       #Epoch 4 finished, validation RMSLE: 0.455638, elapsed time: 3:37:54.649256                                                       #Epoch 5 finished, validation RMSLE: 0.455182, elapsed time: 4:19:44.513727                                                       
#Epoch 6 finished, validation RMSLE: 0.455177, elapsed time: 5:01:45.133674                                                       #NOVAL
#Epoch 0 finished, validation RMSLE: 0.000000, elapsed time: 0:44:15.523639                                                       
#=> 0.44988

#12 keep 11 + L2 to 0.4 => Epoch 0 finished, validation RMSLE: 0.476598, elapsed time: 0:42:07.134475                             #13 keep 11 + L2 to 0.3 => Epoch 0 finished, validation RMSLE: 0.476583, elapsed time: 0:42:38.159764                                                         
#14 keep 13 + D to 2 ** 30 => Epoch 0 finished, validation RMSLE: 0.476181, elapsed time: 0:43:43.260729                         #15 keep 14 + L2 to 0.2 => Epoch 0 finished, validation RMSLE: 0.476166, elapsed time: 0:44:01.927254                                                         
#16 keep 15 + L2 to 0.1 =>  Epoch 0 finished, validation RMSLE: 0.476150, elapsed time: 0:43:09.261362                                                         
#train_extended_producto_c
#1 Keep 16 => Epoch 0 finished, validation RMSLE: 0.505922, elapsed time: 0:45:16.207940                                         #2 Keep 16 + L2 to 0.8 => Epoch 0 finished, validation RMSLE: 0.514963, elapsed time: 0:45:25.549229                             #3 Keep 16 + alpha to 0.02 => 
#Epoch 0 finished, validation RMSLE: 0.488286, elapsed time: 0:44:42.854919                                                      
#Epoch 1 finished, validation RMSLE: 0.461783, elapsed time: 1:29:08.371014                                                       #Epoch 2 finished, validation RMSLE: 0.458877, elapsed time: 2:13:32.646559
#Epoch 3 finished, validation RMSLE: 0.459383, elapsed time: 2:57:59.875738                                                       #.. kill  

#4 keep 16 alpha back to 0.01 =>
#Epoch 0 finished, validation RMSLE: 0.505922, elapsed time: 0:44:57.736427                                                        
#Epoch 1 finished, validation RMSLE: 0.471991, elapsed time: 1:28:59.966807                                                        
#Epoch 2 finished, validation RMSLE: 0.463790, elapsed time: 2:12:58.581609                                                        
#Epoch 3 finished, validation RMSLE: 0.460395, elapsed time: 2:57:02.987538                                                        
#Epoch 4 finished, validation RMSLE: 0.458786, elapsed time: 3:41:03.907627                                                        
#Epoch 5 finished, validation RMSLE: 0.458065, elapsed time: 4:25:04.318906                                                        
#Epoch 6 finished, validation RMSLE: 0.457838, elapsed time: 5:09:00.104553                                                        
#Epoch 7 finished, validation RMSLE: 0.457905, elapsed time: 5:53:00.317891                                                       
#Epoch 8 finished, validation RMSLE: 0.458151, elapsed time: 6:36:58.972384 

#5 Keep 4 + L1, L2, and beta back to default
#Epoch 0 finished, validation RMSLE: 0.517491, elapsed time: 0:45:10.890411                                                       

#6 keep 4 + Epoch 0 finished, validation RMSLE: 0.518850, elapsed time: 0:45:24.473441

#train_extended_producto_c_f
#1 keep 16 => Epoch 0 finished, validation RMSLE: 0.512770, elapsed time: 0:43:30.942052                                         

#train
#0.01, .2, .5, .1, 30 => Epoch 0 finished, validation RMSLE: 0.481887, elapsed time: 0:44:19.050580
#0.01, .2, .2, .1, 30 => Epoch 0 finished, validation RMSLE: 0.478337, elapsed time: 0:43:37.890891
#0.01, .2, .1, .1, 30 => Epoch 0 finished, validation RMSLE: 0.477177, elapsed time: 0:44:07.342824                               
#0.01, .2, 0., .1, 30 => Epoch 0 finished, validation RMSLE: 0.476150, elapsed time: 0:43:17.149523             
#train_extended_producto_c_f
#0.01, .2, 0., .1, 30 => Epoch 0 finished, validation RMSLE: 0.505922, elapsed time: 0:45:45.791042                               
#1 now added shortName, brand, and Town to OHE => Epoch 0 finished, validation RMSLE: 0.497911, elapsed time: 2:58:11.659738       
#2 Keep 1 + alpha to 0.02 => Epoch 0 finished, validation RMSLE: 0.559991, elapsed time: 2:58:48.856897                       

#1 now added brand to OHE => Epoch 0 finished, validation RMSLE: 0.490301, elapsed time: 1:10:50.854083 
#Epoch 1 finished, validation RMSLE: 0.462156, elapsed time: 2:21:15.072434                                                       
#Epoch 2 finished, validation RMSLE: 0.458011, elapsed time: 3:33:02.277052                                                       
#Epoch 3 finished, validation RMSLE: 0.457580, elapsed time: 4:44:50.080185  

#alpha 0.01
#2 Keep 1 + beta to 0.1 => Epoch 0 finished, validation RMSLE: 0.489745, elapsed time: 1:10:28.573481                          
#3 Keep 2 + beta to 0.05 => Epoch 0 finished, validation RMSLE: 0.489785, elapsed time: 1:09:12.574720                            
#4 Keep 2 + beta to 0.1 + Epoch 0 finished, validation RMSLE: 0.490873, elapsed time: 1:09:50.154106  

#only brand?
#Epoch 0 finished, validation RMSLE: 0.509272, elapsed time: 1:09:43.417509                                                
#Epoch 1 finished, validation RMSLE: 0.474005, elapsed time: 2:20:49.695232                                                    
#Epoch 2 finished, validation RMSLE: 0.465168, elapsed time: 3:31:57.556501                                                 
#Epoch 3 finished, validation RMSLE: 0.461345, elapsed time: 4:43:08.581703                                                
#Epoch 4 finished, validation RMSLE: 0.459367, elapsed time: 5:54:16.482602

#only town
#Epoch 0 finished, validation RMSLE: 0.491048, elapsed time: 1:08:58.377660                                                 
#Epoch 1 finished, validation RMSLE: 0.463545, elapsed time: 2:20:02.896870  

#Epoch 1 finished, validation RMSLE: 0.476234, elapsed time: 2:22:31.410323                                                       

#only town
#0.01, .2, 0., .1, 32 => Epoch 0 finished, validation RMSLE: 0.490839, elapsed time: 1:17:46.419111                          

# sorted!!!
#Epoch 0 finished, validation RMSLE: 0.467737, elapsed time: 1:18:48.040279                                                       
#Epoch 1 finished, validation RMSLE: 0.456856, elapsed time: 2:37:10.301975                                                       
#Epoch 2 finished, validation RMSLE: 0.455265, elapsed time: 3:55:33.288530                                                    
#Epoch 3 finished, validation RMSLE: 0.455579, elapsed time: 5:14:05.013319

# now with alpha 0.005
#Epoch 0 finished, validation RMSLE: 0.481340, elapsed time: 1:17:52.802189                                                       
#Epoch 1 finished, validation RMSLE: 0.466525, elapsed time: 2:36:03.867720                                                        
#Epoch 2 finished, validation RMSLE: 0.461224, elapsed time: 3:54:12.924191
#Epoch 3 finished, validation RMSLE: 0.458480, elapsed time: 5:12:24.986266
#Epoch 4 finished, validation RMSLE: 0.456941, elapsed time: 6:30:35.227014                                                 
#Epoch 5 finished, validation RMSLE: 0.456079, elapsed time: 7:48:34.767519                                                 
#Epoch 6 finished, validation RMSLE: 0.455634, elapsed time: 9:06:56.879167                                                 

# back to only brand with alpha 0.01 and best settings
#Epoch 0 finished, validation RMSLE: 0.465612, elapsed time: 1:18:40.550792                                                         
#Epoch 1 finished, validation RMSLE: 0.458688, elapsed time: 2:37:42.114046                                                        #Epoch 2 finished, validation RMSLE: 0.455941, elapsed time: 3:56:40.411355                                                   #Epoch 3 finished, validation RMSLE: 0.456649, elapsed time: 5:15:38.530392                                                        #Epoch 4 finished, validation RMSLE: 0.457561, elapsed time: 6:34:40.705740                                                   #Epoch 5 finished, validation RMSLE: 0.458825, elapsed time: 7:53:32.066216



#
#Epoch 0 finished, validation RMSLE: 0.475302, elapsed time: 0:48:08.173723                                                  
#Epoch 1 finished, validation RMSLE: 0.464098, elapsed time: 1:34:42.215594                                              

# D, training/validation
epoch = 15  # learn training data for N passes
holdout = 9  # use week holdout validation


##############################################################################
# class, function, generator definitions #####################################
##############################################################################

class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)
            for i in range(L):
                for j in range(i+1, L):
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D
                    for k in range(j+1, L):
                        yield abs(hash(str(x[i]) + '_' + str(x[j]) +
                                  '_' + str(x[k]))) % D
                        for l in range(k+1, L):
                            yield abs(hash(str(x[i]) + '_' + str(x[j]) +
                                      '_' + str(x[k]) + '_' + str(x[l]))) % D

    def predict(self, x):
        ''' Get demand estimation on x

            INPUT:
                x: features

            OUTPUT:
                demand
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if ((L1 > 0) & (sign * z[i] <= L1)):
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # Raw Output
        return wTx

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: demand prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


def data(path, D):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y: log(actual demand +1)
    '''

    for t, row in enumerate(DictReader(open(path))):
        ID = 0
        week = 0
        y = 0.
        if 'id' in row:
            ID = row['id']
            del row['id']
        if 'Demanda_uni_equil' in row:
            y = log(float(row['Demanda_uni_equil'])+1.)
            del row['Demanda_uni_equil']
        if 'Semana' in row:
            week = int(row['Semana'])
            del row['Semana']
        # build x
        x = []
        for key in row:
            if(key == 'Canal_ID' or key == 'Ruta_SAK' or
               key == 'Cliente_ID' or key == 'Producto_ID' or
               key == 'Agencia_ID'
               ):
                value = row[key]
                # one-hot encode everything with hash trick
                index = abs(hash(key + '_' + value)) % D
                x.append(index)
            elif(key == 'wtpcs' or key == 'lag_3'):
                 x.append(row[key])

        yield t, week, ID, x, y


##############################################################################
# start training #############################################################
##############################################################################
if __name__ == "__main__":
    print('Use PYPY!!!!')
    # print('Remove the next line!!!!')
    # exit(0)
    start = datetime.now()

    # initialize ourselves a learner
    learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)

    # start training
    for e in range(epoch):
        loss = 0.
        count = 0
        for t, week, ID, x, y in data(train, D):  # data is a generator
            #   t: just a instance counter
            #   week: you know what this is
            #   ID: id provided in original data
            #   x: features
            #   y: log(actual demand + 1)
            # step 1, get prediction from learner
            p = learner.predict(x)
            #if((t % 1000000) == 0):
            #    print(t)

            if ((holdout != 0) and (week >= holdout)):
                # step 2-1, calculate validation loss
                #           we do not train with the validation data so our
                #           validation loss is an accurate estimation
                #
                # holdout: train instances from day 1 to day N -1
                #            validate with instances from day N and after
                #
                loss += (max(0, p)-y)**2
                count += 1
            else:
                # step 2-2, update learner with demand information
                learner.update(x, p, y)

        count = max(count, 1)
        print('Epoch %d finished, validation RMSLE: %f, elapsed time: %s' %
              (e, sqrt(loss/count), str(datetime.now() - start)))

    #########################################################################
    # start testing, and build Kaggle's submission file #####################
    #########################################################################
    
    #with open(submission, 'w') as outfile:
    #    outfile.write('id,Demanda_uni_equil\n')
    #    for t, date, ID, x, y in data(test, D):
    #        p = learner.predict(x)
    #        outfile.write('%s,%.3f\n' % (ID,
    #                                     expm1(max(0, p))))
    #        if((t % 1000000) == 0):
    #            print(t)
    print('Finished')
