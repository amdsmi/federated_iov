import tensorflow as tf
from pymoo.core.problem import ElementwiseProblem
import json
from tensorflow import keras
import src.config.config as cfg
from src.AI.dataset import DataGenerator
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

with open('model1.json') as f:
    data1 = json.load(f)

with open('model2.json') as f:
    data2 = json.load(f)

with open('model3.json') as f:
    data3 = json.load(f)

model1 = keras.models.model_from_json(data1['model'])
model2 = keras.models.model_from_json(data2['model'])
model3 = keras.models.model_from_json(data3['model'])

weights1 = [np.array(w) for w in data1['weights']]
weights2 = [np.array(w) for w in data2['weights']]
weights3 = [np.array(w) for w in data3['weights']]

model1.set_weights(weights1)
model2.set_weights(weights2)
model3.set_weights(weights3)

model1.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model2.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model3.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

val_dataset = DataGenerator(
    images_path=cfg.dataset_path,
    label_csv=cfg.val_labels,
    dim=cfg.image_size,
    pad_resize=cfg.pad_resize,
    batch_size=cfg.batch_size)

loss_acc = []
X = []


# Step 3: Initialize Pymoo Problem
class ModelCombinationProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=3,
                         n_obj=2,
                         n_ieq_constr=2,
                         xl=np.array([0, 0, 0]),
                         xu=np.array([1, 1, 1]))

    def _evaluate(self, x, out, *args, **kwargs):
        # Normalize the weights
        print(x)
        weights = x / np.sum(x)
        X.append(list(weights))

        combined_model = tf.keras.models.clone_model(model1)
        weights_model1 = [weights[0] * i for i in model1.get_weights()]
        weights_model2 = [weights[1] * i for i in model2.get_weights()]
        weights_model3 = [weights[2] * i for i in model3.get_weights()]

        # Combine the weights using the normalized weights
        combined_weights = []
        for i in range(len(weights_model1)):
            combined_weights.append(weights_model1[i] + weights_model2[i] + weights_model3[i])

            # Set the weights of the combined model
        combined_model.set_weights(combined_weights)
        combined_model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Evaluate the accuracy of the combined model on a validation set
        loss, acc = combined_model.evaluate(val_dataset)
        loss_acc.append([loss, acc])

        out["F"] = [loss, -acc]


# Step 4: Define Optimization Algorithm
problem = ModelCombinationProblem()
algorithm = NSGA2(
    pop_size=2,
    n_offsprings=2,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 2)

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbos=True)


X = np.array(X)
F = np.array(loss_acc)

best_solution = X[np.argmax(F[:, 1])]

weights_model1 = [best_solution[0] * i for i in model1.get_weights()]
weights_model2 = [best_solution[1] * i for i in model2.get_weights()]
weights_model3 = [best_solution[2] * i for i in model3.get_weights()]
combined_model = tf.keras.models.clone_model(model1)
combined_weights = []
for i in range(len(weights_model1)):
    combined_weights.append(weights_model1[i] + weights_model2[i] + weights_model3[i])
#

accuracy = combined_model.evaluate(val_dataset)[1]
combined_model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

combined_model.set_weights(combined_weights)

print("Best Accuracy:", accuracy)
