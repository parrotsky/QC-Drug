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

from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
 
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
        'device': 'GPU'
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
vqe_solver.initial_point = [0.0] * ansatz.num_parameters
# print (ansatz.num_parameters)
# print (vqe_solver.initial_point)
start_time = time.time()
calc = GroundStateEigensolver(mapper, vqe_solver)
res = calc.solve(qmolecule)
end_time = time.time() - start_time
print(res, f", time: {end_time}")

result = res.computed_energies + res.nuclear_repulsion_energy
error_rate = abs(abs(ref_value - result) / ref_value * 100)
print("Error rate: %f%%" % (error_rate))



# from qiskit.providers.fake_provider import *
# backend = FakeMontreal()

# with pulse.build(backend) as my_program1:
#   pulse.call(ansatz)

# print (f"Duration: {my_program1.duration}")



