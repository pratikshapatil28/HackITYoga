# # import os
# # import numpy as np
# # from tensorflow.keras.utils import to_categorical
# # from keras.layers import Input, Dense, LSTM, Embedding, Conv1D, GlobalMaxPooling1D, concatenate
# # from keras.models import Model

# # is_init = False
# # size = -1

# # label = []
# # dictionary = {}
# # c = 0

# # X = None  # Initialize X outside the loop
# # y_list = []  # Use a list to collect labels

# # for i in os.listdir():
# #     if i.split(".")[-1] == "npy" and not (i.split(".")[0] == "labels"):
# #         if not is_init:
# #             is_init = True
# #             X = np.load(i)
# #             size = X.shape[0]
# #             y_list.extend([i.split('.')[0]] * size)
# #         else:
# #             X = np.concatenate((X, np.load(i)))
# #             y_list.extend([i.split('.')[0]] * size)

# #         label.append(i.split('.')[0])
# #         dictionary[i.split('.')[0]] = c
# #         c = c + 1

# # # Check if y_list is not empty before converting to NumPy array
# # if y_list:
# #     y = np.array(y_list).reshape(-1, 1)

# #     # Now, y is defined outside the loop
# #     for i in range(y.shape[0]):
# #         y[i, 0] = dictionary[y[i, 0]]

# #     y = np.array(y, dtype="int32")
# #     y = to_categorical(y)

# #     # Reshape X to add a third dimension (timesteps) for the LSTM layer
# #     X = X.reshape((X.shape[0], 1, X.shape[1]))

# #     X_new = X.copy()
# #     y_new = y.copy()

# #     counter = 0

# #     cnt = np.arange(X.shape[0])
# #     np.random.shuffle(cnt)

# #     # Use min(X_new.shape[0], len(cnt)) as the loop limit
# #     for i in cnt[:min(X_new.shape[0], len(cnt))]:
# #         X_new[counter] = X[i]
# #         y_new[counter] = y[i]
# #         counter = counter + 1

# #     # Model architecture
# #     ip = Input(shape=(X.shape[1], X.shape[2]))

# #     # LSTM layer
# #     lstm_out = LSTM(64, activation='tanh')(ip)

# #     # Dense layers
# #     dense_out = Dense(128, activation='tanh')(lstm_out)
# #     dense_out = Dense(64, activation='tanh')(dense_out)

# #     # CNN layer with adjusted parameters
# #     cnn_out = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(ip)
# #     cnn_out = GlobalMaxPooling1D()(cnn_out)

# #     # Concatenate the outputs of LSTM and CNN
# #     merged = concatenate([lstm_out, cnn_out])

# #     # Output layer
# #     op = Dense(y.shape[1], activation='softmax')(merged)

# #     # Model definition
# #     model = Model(inputs=ip, outputs=op)

# #     # Compilation
# #     model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# #     # Training
# #     model.fit(X_new, y_new, epochs=80)

# #     # Save the model and labels
# #     model.save("model.h5")
# #     np.save("labels.npy", np.array(label))
# # else:
# #     print("No valid files found to process.")

# import os
# import numpy as np
# from tensorflow.keras.utils import to_categorical
# from keras.layers import Input, Dense, LSTM, Embedding, Conv1D, GlobalMaxPooling1D, concatenate
# from keras.models import Model

# is_init = False
# size = -1

# label = []
# dictionary = {}
# c = 0

# X = None  # Initialize X outside the loop
# y_list = []  # Use a list to collect labels

# for i in os.listdir():
#     if i.split(".")[-1] == "npy" and not (i.split(".")[0] == "labels"):
#         if not is_init:
#             is_init = True
#             X = np.load(i)
#             size = X.shape[0]
#             y_list.extend([i.split('.')[0]] * size)
#         else:
#             X = np.concatenate((X, np.load(i)))
#             y_list.extend([i.split('.')[0]] * size)

#         label.append(i.split('.')[0])
#         dictionary[i.split('.')[0]] = c
#         c = c + 1

# # Check if y_list is not empty before converting to NumPy array
# if y_list:
#     y = np.array(y_list).reshape(-1, 1)

#     # Now, y is defined outside the loop
#     for i in range(y.shape[0]):
#         y[i, 0] = dictionary[y[i, 0]]

#     y = np.array(y, dtype="int32")
#     y = to_categorical(y)

#     # Reshape X to add a third dimension (timesteps) for the LSTM layer
#     X = X.reshape((X.shape[0], 1, X.shape[1]))

#     X_new = X.copy()
#     y_new = y.copy()

#     counter = 0

#     cnt = np.arange(X.shape[0])
#     np.random.shuffle(cnt)

#     # Use min(X_new.shape[0], len(cnt)) as the loop limit
#     for i in cnt[:min(X_new.shape[0], len(cnt))]:
#         X_new[counter] = X[i]
#         y_new[counter] = y[i]
#         counter = counter + 1

#     # Model architecture
#     ip = Input(shape=(X.shape[1], X.shape[2]))

#     # LSTM layer
#     lstm_out = LSTM(64, activation='tanh')(ip)

#     # Dense layers
#     dense_out = Dense(128, activation='tanh')(lstm_out)
#     dense_out = Dense(64, activation='tanh')(dense_out)

#     # CNN layer with adjusted parameters
#     cnn_out = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(ip)
#     cnn_out = GlobalMaxPooling1D()(cnn_out)

#     # Concatenate the outputs of LSTM and CNN
#     merged = concatenate([lstm_out, cnn_out])

#     # Output layer
#     op = Dense(y.shape[1], activation='softmax')(merged)

#     # Model definition
#     model = Model(inputs=ip, outputs=op)

#     # Compilation
#     model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

#     # Training
#     model.fit(X_new, y_new, epochs=80)

#     # Save the model and labels
#     model.save("model.h5")
#     np.save("labels.npy", np.array(label))
# else:
#     print("No valid files found to process.")

import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense, LSTM, Embedding, Conv1D, GlobalMaxPooling1D, concatenate
from keras.models import Model

is_init = False
size = -1

label = []
dictionary = {}
c = 0

X = None  # Initialize X outside the loop
y_list = []  # Use a list to collect labels

for i in os.listdir():
    if i.split(".")[-1] == "npy" and not (i.split(".")[0] == "labels"):
        if not is_init:
            is_init = True
            X = np.load(i)
            size = X.shape[0]
            y_list.extend([i.split('.')[0]] * size)
        else:
            temp_X = np.load(i)
            size += temp_X.shape[0]
            X = np.concatenate((X, temp_X))
            y_list.extend([i.split('.')[0]] * temp_X.shape[0])

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c
        c += 1

# Check if y_list is not empty before converting to NumPy array
if y_list:
    y = np.array(y_list).reshape(-1, 1)

    # Now, y is defined outside the loop
    for i in range(y.shape[0]):
        y[i, 0] = dictionary[y[i, 0]]

    y = np.array(y, dtype="int32")
    y = to_categorical(y)

    # Reshape X to add a third dimension (timesteps) for the LSTM layer
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    X_new = X.copy()
    y_new = y.copy()

    counter = 0

    cnt = np.arange(X.shape[0])
    np.random.shuffle(cnt)

    # Use min(X_new.shape[0], len(cnt)) as the loop limit
    for i in cnt[:min(X_new.shape[0], len(cnt))]:
        X_new[counter] = X[i]
        y_new[counter] = y[i]
        counter += 1

    # Model architecture
    ip = Input(shape=(X.shape[1], X.shape[2]))

    # LSTM layer
    lstm_out = LSTM(64, activation='tanh')(ip)

    # Dense layers
    dense_out = Dense(128, activation='tanh')(lstm_out)
    dense_out = Dense(64, activation='tanh')(dense_out)

    # CNN layer with adjusted parameters
    cnn_out = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(ip)
    cnn_out = GlobalMaxPooling1D()(cnn_out)

    # Concatenate the outputs of LSTM and CNN
    merged = concatenate([lstm_out, cnn_out])

    # Output layer
    op = Dense(y.shape[1], activation='softmax')(merged)

    # Model definition
    model = Model(inputs=ip, outputs=op)

    # Compilation
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    # Training
    model.fit(X_new, y_new, epochs=80)

    # Save the model and labels
    model.save("model.h5")
    np.save("labels.npy", np.array(label))
else:
    print("No valid files found to process.")
