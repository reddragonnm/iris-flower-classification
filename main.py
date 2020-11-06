share_model = False # gives a public link that can be shared

# importing modules
import gradio as gr # gradio is used to open the model in the web
from keras.models import load_model
import numpy as np

# loading final keras model
model = load_model("models/keras_model.h5")

# a function for convenience
def predict(*args):
    assert len(args) == 4
    pred = model.predict(np.array(args).reshape((1, 4,)))[0].argmax()
    return list(pred)

# predicting the confidence  level for each classs
def predict_classes(*args):
    assert len(args) == 4
    pred = model.predict(np.array(args).reshape((1, 4,)))[0]
    return list(pred / np.sum(pred))

# final function for gradio to use
def gradio_predict(*args):
    assert len(args) == 4
    args = [float(i) for i in args]

    pred = predict_classes(*args)
    pred = [str(round(i)) for i in pred]

    flower_names = [
        "Iris setosa",
        "Iris versicolor",
        "Iris virginica"
    ]

    return dict(zip(flower_names, pred))

if __name__ == '__main__':
    # making the gradio interface
    labels = ["Sepal width", "Sepal length", "Petal width", "Petal length"]
    interface = gr.Interface(fn=gradio_predict, inputs=[gr.inputs.Number(label=labels[i]) for i in range(4)], outputs="label")
    interface.launch(share=share_model)
