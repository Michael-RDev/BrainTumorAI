from model import model
import numpy as np
import matplotlib.pyplot as plt 


def predictImgs(x_thing: list, y_thing: list):
    predictions = model.predict(x_thing)
    predicted_labels = [np.argmax(y_labels) for y_labels in predictions]
    true_labels = [np.argmax(y_labels) for y_labels in y_thing]
    correct_pred = sum(1 for pred, true_label in zip(predicted_labels, true_labels) if pred == true_label)
    accuracy = correct_pred / len(true_labels)
    print(accuracy)

    fig, axes = plt.subplots(3, 8, figsize=(12,5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_thing[i], cmap='gray')
        ax.set_title(f"Pred: {predicted_labels[i]}, True: {true_labels[i]}")

    plt.show()
    return accuracy


#0 - No Cancer, 1 - Cancer