from prepData import X_train,  y_train
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, LeakyReLU

def CNNmodel(input_shape:tuple):
    model =  Sequential()
    #Feature Extraction
    model.add(Conv2D(64,(4,4), input_shape=input_shape, activation='relu'))
    model.add(Conv2D(64,(4,4), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(64,(4,4), activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(128,(3,3), activation='relu'))
    model.add(Conv2D(128,(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64,(4,4), activation='relu'))
    model.add(LeakyReLU())
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(MaxPool2D((2,2)))

    #Linear Classifer

    model.add(Flatten())
    model.add(Dense(150,activation='relu'))
    model.add(Dense(120,activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

model = CNNmodel(input_shape=(50, 50, 3))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())


model.fit(X_train, y_train, epochs=40)
    