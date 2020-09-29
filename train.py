from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
import tools

# -------------------------- Prepare Train Set --------------------------
# Making images into array
def train_imgToArray():
    rootdirTrain = '.\dataset\\train'
    dataTrain, labelsTrain = tools.imgToArray(rootdirTrain)
    return dataTrain, labelsTrain


# Convert dataTrain and labelsTrain to numpy arrays
def train_convertToNumpy(dataTrain, labelsTrain):
    cowsTrain, labelsTrain = tools.convertToNumpy(dataTrain, labelsTrain)
    return cowsTrain,labelsTrain


# Create data - X and Y
def train_createData(cowsTrain, labelsTrain):
    x_train, y_train = tools.createData(cowsTrain, labelsTrain)
    return x_train, y_train


# -------------------------- Making Keras model --------------------------
def kerasModel():
    model=Sequential()
    model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(500,activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(46,activation="softmax"))
    model.summary()

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model


# -------------------------- Train the model --------------------------
def training(model, x_train, y_train):
    model.fit(x_train,y_train,batch_size=50,epochs=5,verbose=1)
    return model
