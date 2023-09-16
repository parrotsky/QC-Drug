import optuna
from optuna_dashboard import run_server
# import pandas as pd
import numpy as np
from functools import partial
import os
import argparse
import copy, pickle

## quantum
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper,ParityMapper,QubitConverter
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit_aer.primitives import Estimator
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
import numpy as np
import pylab
import qiskit.providers
from qiskit import Aer,pulse, QuantumCircuit
from qiskit.utils import QuantumInstance, algorithm_globals
import time

from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver



def objective(trial, vqe_solver, mapper, qmolecule):
    vqe_solver.initial_point = [trial.suggest_float(f'{i}', -4, 4) for i in range(92)]
    # vqe_solver.initial_point = [0.0] * 92
    
    
    start_time = time.time()
    calc = GroundStateEigensolver(mapper, vqe_solver)
    res = calc.solve(qmolecule)
    end_time = time.time() - start_time
    

    result = res.computed_energies + res.nuclear_repulsion_energy
    error_rate = abs(abs(ref_value - result) / ref_value * 100)
    error_rate = 0.02
    return error_rate 


def main():
    global args
    parser = argparse.ArgumentParser(description='Search HyperParameter for Quantum Drug Test')
    parser.add_argument('study', type=str, help='The name of the study to search for.')
    parser.add_argument('save_path', type=str, help='The Path folder to save experiments results.')
    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
        
    seeds = 170
    algorithm_globals.random_seed = seeds
    seed_transpiler = seeds
    iterations = 125
    shot = 6000

    ultra_simplified_ala_string = """
    O 0.0 0.0 0.0
    H 0.45 -0.1525 -0.8454
    """

    driver = PySCFDriver(
        atom=ultra_simplified_ala_string.strip(),
        basis='sto3g',
        charge=1,
        spin=0,
        unit=DistanceUnit.ANGSTROM
    )
    qmolecule = driver.run()


    hamiltonian = qmolecule.hamiltonian
    coefficients = hamiltonian.electronic_integrals
    print(coefficients.alpha)
    second_q_op = hamiltonian.second_q_op()

    mapper = JordanWignerMapper()
    converter = QubitConverter(mapper=mapper, two_qubit_reduction=False)
    qubit_op = converter.convert(second_q_op)

    solver = GroundStateEigensolver(
        JordanWignerMapper(),
        NumPyMinimumEigensolver(),
    )

    result = solver.solve(qmolecule)
    print(result.computed_energies)

    print(result.nuclear_repulsion_energy)

    ref_value = result.computed_energies + result.nuclear_repulsion_energy
    print(ref_value)

    ansatz = UCCSD(
        qmolecule.num_spatial_orbitals,
        qmolecule.num_particles,
        mapper,
        initial_state=HartreeFock(
            qmolecule.num_spatial_orbitals,
            qmolecule.num_particles,
            mapper,
        ),
    )
    estimator = Estimator(
        backend_options = {
            'method': 'statevector',
            'device': 'CPU'
            # 'noise_model': noise_model
        },
        run_options = {
            'shots': shot,
            'seed': seeds,
        },
        transpile_options = {
            'seed_transpiler':seed_transpiler
        }
    )

    vqe_solver = VQE(estimator, ansatz, SLSQP())


    study_name = args.study
    storage_url = f'sqlite:///{study_name}.db'
    #sampler = optuna.samplers.TPESampler()
    #sampler = optuna.samplers.TPESampler()
    if "nsga" in args.study:
        sampler = optuna.samplers.NSGAIISampler()
    elif "tpe" in args.study:
        sampler = optuna.samplers.TPESampler()
    elif "cmaes" in args.study:
        sampler = optuna.samplers.CmaEsSampler()
    else:
        raise ValueError("study not support.")

    if os.path.exists(f"{study_name}.db"):
        print (f"load previous {study_name} study...")
    else:
    # Create study
        # study = optuna.create_study(sampler=sampler, directions=['minimize', 'minimize'], study_name=study_name, storage=storage_url)
        study = optuna.create_study(sampler=sampler, directions=['minimize'], study_name=study_name, storage=storage_url)
    # Optimize in blocks of 100 trials
    for i in range(10):  # 10 blocks of 100 trials
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        # study.optimize(partial(single_objective, model=model, device_num=device_num,
        #  fail_num=fail_num, flops_reduction=args.flops, 
        # val_loader=val_loader), n_trials=100)
        study.optimize(partial(objective, vqe_solver=vqe_solver, mapper=mapper, qmolecule=qmolecule), n_trials=100)

    # run_server(storage)


    # in terminal
    # optuna-dashboard sqlite:///db.sqlite3
    # You can analyze the Pareto front solutions afterward
    #pareto_solutions = study.get_pareto_front_trials()



    # Convert study trials to DataFrame
    # df = study.trials_dataframe(attrs=('number', 'params', 'values'))

    # # Filter to get Pareto optimal trials
    # def is_pareto_efficient(costs):
    #     is_efficient = np.ones(costs.shape[0], dtype=bool)
    #     for i, c in enumerate(costs):
    #         if is_efficient[i]:
    #             is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Keep any point with a lower cost
    #             is_efficient[i] = True  # And keep self
    #     return is_efficient

    # pareto_trials = df.loc[is_pareto_efficient(df[['values_0', 'values_1']].values)]
    print ("---------")




if __name__ == '__main__':
    main()