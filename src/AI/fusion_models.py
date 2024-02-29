from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.termination import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.mutation.pm import PM
from src.AI.dataset import DataGenerator
from pymoo.optimize import minimize
import src.config.config as cfg
from tensorflow import keras
import tensorflow as tf
import numpy as np

np.random.seed(cfg.seed)
tf.random.set_seed(cfg.seed)


def genetic_fusion_model(models_list):
    data1 = models_list[0]
    data2 = models_list[1]

    model1 = keras.models.model_from_json(data1['model'])
    model2 = keras.models.model_from_json(data2['model'])

    weights1 = [np.array(w) for w in data1['weights']]
    weights2 = [np.array(w) for w in data2['weights']]

    model1.set_weights(weights1)
    model2.set_weights(weights2)

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
    val_dataset = DataGenerator(
        images_path=cfg.dataset_path,
        label_csv=cfg.val_labels,
        dim=cfg.image_size,
        pad_resize=cfg.pad_resize,
        batch_size=cfg.batch_size)

    loss_acc = []
    solutions = []

    class ModelCombinationProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=3,
                             n_obj=2,
                             n_ieq_constr=2,
                             xl=np.array([0, 0]),
                             xu=np.array([1, 1]))

        def _evaluate(self, x, out, *args, **kwargs):
            # Normalize the weights
            print(x)
            weights = x / np.sum(x)
            solutions.append(list(weights))

            combined_model = tf.keras.models.clone_model(model1)
            weights_model1 = [weights[0] * i for i in model1.get_weights()]
            weights_model2 = [weights[1] * i for i in model2.get_weights()]

            # Combine the weights using the normalized weights
            combined_weights = []
            for i in range(len(weights_model1)):
                combined_weights.append(weights_model1[i] + weights_model2[i])

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

    solutionis = np.array(solutions)
    loss_acc = np.array(loss_acc)

    best_solution = solutions[np.argmax(loss_acc[:, 1])]

    weights_model1 = [best_solution[0] * i for i in model1.get_weights()]
    weights_model2 = [best_solution[1] * i for i in model2.get_weights()]

    combined_model = tf.keras.models.clone_model(model1)

    combined_weights = []
    for i in range(len(weights_model1)):
        combined_weights.append(weights_model1[i] + weights_model2[i] )

    combined_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    combined_model.set_weights(combined_weights)

    loss, accuracy = combined_model.evaluate(val_dataset)

    data = {'model': combined_model.to_json(),
            'weights': [np.array(w).tolist() for w in combined_weights],
            'metrics': {'loss': loss, 'acc': accuracy}}

    return data


def fusion_best_models(models_list):
    best_model = models_list[0]

    for idx in range(1, len(models_list)):
        model = models_list[idx]
        if model['metrics']['acc'] > best_model['metrics']['acc']:
            best_model = model

    return best_model
