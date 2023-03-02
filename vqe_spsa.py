from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, execute, IBMQ, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.tools import job_monitor

from qiskit.circuit.library import EfficientSU2
from qiskit.circuit.library import RealAmplitudes

from qiskit.aqua.components.optimizers import Optimizer, SLSQP
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.operators.gradients import GradientBase
from qiskit.aqua.components.optimizers import SPSA

from qiskit.test.mock import FakeMumbai

from qiskit.aqua.algorithms import NumPyEigensolver

from qiskit.compiler import transpile

from qiskit.algorithms.optimizers import SPSA

import numpy as np
from skquant.opt import minimize
import pickle
import copy
import csv

from term_grouping import *
import Reconstruction_Functions as RF

#################### Function to get Pauli strings and Coefficients from the parsed hamiltonian #####################
def give_paulis_and_coeffs(hamiltonian, num_qubits):
    '''
    hamiltonian: A list containing all hamiltonian terms along with their weights
    num_qubits: The number of qubits in the hamiltonian
    '''
    paulis = []
    coeffs = []
    
    for idx, term in enumerate(hamiltonian):
        
        #the coefficient
        coeffs.append(term[0])
        
        #the pauli string
        pauli_string = num_qubits*'I'
        all_gates = term[1]
        #print(non_id_gates)
        
        for _, gate in enumerate(all_gates):
            pauli = gate[0]
            location = int(gate[1])
            #print('location: ', location, 'pauli_string: ', pauli_string, 'pauli: ', pauli)
            pauli_string = pauli_string[0:location] + pauli + pauli_string[location+1:]
            #print(pauli_string, len(pauli_string))
        
        paulis.append(pauli_string)
    
    return coeffs, paulis

##################### Function to create parameterized quantum state ################################
def quantum_state_preparation(circuit, parameters):
    '''
    Args:
    circuit: The input circuit to which we append the parameterized state
    parameters: The parameters of the rotations
    
    Returns:
    Circuit with /home/siddharthdangwal/JigSaw+VQE/Data/Experiment 2/TFIM-4-full/noisy_jigsaw_params.csvthe ansatz for a generalized state appended to it
    '''
    num_qubits = circuit.num_qubits
    
    #the number of repetitions of a general ansatz block
    p = (len(parameters)/(2*num_qubits)) - 1
    
    #make sure that p is an integer and then change the format
    assert int(p) == p
    p = int(p)
    
    #create an EfficientSU2 ansatz
    ansatz = EfficientSU2(num_qubits = num_qubits, entanglement = 'full', reps = p, insert_barriers = True)
    ansatz.assign_parameters(parameters = parameters, inplace = True)
    circuit.compose(ansatz, inplace = True)
    
    return circuit

##################### Prepare a VQE circuit for a given single-element hamiltonian ################################
def vqe_circuit(n_qubits, parameters, hamiltonian):
    '''
    Args:
    n_qubits: The number of qubits in the circuit
    parameters: The parameters for the vqe circuit
    hamiltonian: The hamiltonian string whose expectation would be measured
    using this circuit
    
    Returns:
    The VQE circuit for the given Pauli tensor hamiltonian 
    '''
    from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
    
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_qubits)
    circuit = QuantumCircuit(qr, cr)
    
    #append the circuit with the state preparation ansatz
    circuit = quantum_state_preparation(circuit, parameters)
    
    #add the measurement operations
    for i, el in enumerate(hamiltonian):
        if el == 'I':
            #no measurement for identity
            continue
        elif el == 'Z':
            circuit.measure(qr[i], cr[i])
        elif el == 'X':
            circuit.u(np.pi/2, 0, np.pi, qr[i])
            circuit.measure(qr[i], cr[i])
        elif el == 'Y':
            circuit.u(np.pi/2, 0, np.pi/2, qr[i])
            circuit.measure(qr[i], cr[i])
    
    return circuit

##################### Compute expectation of each term in the hamiltonian ################################
def compute_expectations(parameters, paulis, shots, backend, mode):
    '''
    Args:
    parameters: The parameters of the VQE ansatz
    paulis: The paulis tensor hamiltonians
    backend: The backend on which the vqe is run
    mode: Specifies if we have to run a noisy simulation or ideal simulation or run the circuit on a device
    shots: The total shots budget
    
    Returns:
    A list of expectations for each circuit
    '''
    
    #the number of qubits
    n_qubits = len(paulis[0])
    
    #get all the vqe circuits
    circuits = [vqe_circuit(n_qubits, parameters, pauli) for pauli in paulis]
    
    #evaluate the circuits
    if mode == 'no_noisy_sim' or mode == 'device_execution':
        job = execute(circuits, backend = backend, optimization_level = 3, shots = 8192)
        result = job.result()
        all_counts = []
        for __, _id in enumerate(paulis):
            if _id == len(_id)*'I':
                all_counts.append({len(_id)*'0':8192})
            else:
                all_counts.append(result.get_counts(__))
    elif mode == 'noisy_sim':
        sim_device = AerSimulator.from_backend(backend)
        tcircs = transpile(circuits, sim_device, optimization_level = 3)
        result_noise = sim_device.run(tcircs, shots = 8192).result()
        #all_counts = result_noise.get_counts()
        all_counts = []
        for __, _id in enumerate(paulis):
            if _id == len(_id)*'I':
                all_counts.append({len(_id)*'0':8192})
            else:
                all_counts.append(result_noise.get_counts(__))
    else:
        raise Exception('Invalid circuit execution mode')
    
    #compute the expectations
    expectations = []
    
    for i, count in enumerate(all_counts):
        
        #initiate the expectation value to 0
        expectation_val = 0
        
        #compute the expectation
        for el in count.keys():
            sign = 1
            
            #change sign if there are an odd number of ones
            if el.count('1')%2 == 1:
                sign = -1
            
            expectation_val += sign*count[el]/8192
        
        expectations.append(expectation_val)
    
    return expectations

##################### Compute vqe loss ################################
def compute_loss(parameters, paulis, coeffs, shots, backend, mode):
    '''
    Args:
    parameters: The parameters of the VQE ansatz
    paulis: The paulis tensor hamiltonians
    coeffs: The coefficients corresponding to each pauli tensor
    backend: The backend on which the vqe is run
    mode: Specifies if we have to run a noisy simulation or ideal simulation or run the circuit on a device
    shots: The number of shots for which each circuit is executed
    
    Returns:
    The loss for the entire VQE hamiltonian
    '''
    
    expectations = compute_expectations(parameters, paulis, shots, backend, mode)
    
    loss = 0
    
    for i, el in enumerate(expectations):
        loss += coeffs[i]*el
    
    return loss

##################### Loss corresponding to one iteration of VQE circuit ################################
def vqe(parameters, paulis, coeffs, shots, backend, mode, loss_filename = None, params_filename = None):
    '''
    Args:
    parameters: The parameters of the VQE ansatz
    paulis: The paulis tensor hamiltonians
    coeffs: The coefficients corresponding to each pauli tensor
    backend: The backend on which the vqe is run
    mode: Specifies if we have to run a noisy simulation or ideal simulation or run the circuit on a device
    shots: The number of shots for which each circuit is executed
    
    Returns:
    Loss for one iteration of the VQE
    '''
    
    #number of qubits in the VQE ansatz
    n_qubits = len(paulis[0])
    
    #making sure that the number of elements in each pauli tensor is the same
    for i in paulis:
        assert len(i) == n_qubits
    
    loss =  compute_loss(parameters, paulis, coeffs, shots, backend, mode)
    print('Loss computed by VQE is: {}'.format(loss))

    return loss

############################ Callback function ############################################################################################################
def callback_function(nfev, x_next, fx_next, norm, boolean, loss_list = None, params_list = None, loss_filename = None, params_filename = None):
    '''
    Args:
    We do not need to worry about any other argument except:
    1)x_next: The parameter value
    2)fx_next: The value of the function
    3)loss_list: A global list in which we store the values of losses that the callback function prints
    4)loss_filename: The name of the file where the loss is stored
    5)params_filename: The name of the file where the params are stored
    
    Returns:
    Prints loss and parameters and saves them in a file
    '''
    
    global current_iteration
    
    #Print the loss and parameters
    print('The loss value at iteration number ', current_iteration, ' is: ', fx_next)
    #print('The parameters are: ', x_next)
    
    #store loss in the list
    if not (loss_list == None):
        loss_list.append(fx_next)
    
    #store params in the list
    if not (params_list == None):
        params_list.append(x_next)
    
    #Save them 
    if not (loss_filename == None):
        with open(loss_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([fx_next])
    
    if not(params_filename == None):
        with open(params_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(x_next)
    
    current_iteration += 1
    
    return 

################## Run VQE #####################
tol = 1e-5

simulator = Aer.get_backend('qasm_simulator')
device = FakeMumbai()

shots_sim = 8192
shots_dev = 8192

#get the seed as the first argument
seed = int(sys.argv[1])
np.random.seed(seed)

p = 2

#get simulation type -- noiseless or noisy
sim_type = sys.argv[2]

#get the system hamiltonian
hamiltonian_string = sys.argv[3]
hamiltonian_string_elements = hamiltonian_string.split('/')
hamil = parseHamiltonian(hamiltonian_string)

max_length = 0
for i in hamil:
    if int(i[1][-1][1]) + 1 > max_length:
        max_length = int(i[1][-1][1]) + 1

#Number of qubits in the hamiltonian
n_qubits = max_length

#get paulis and coefficients
coeffs, paulis = give_paulis_and_coeffs(hamil, n_qubits)

#define optimizer bounds, initial_point and budget
bounds = np.array([[0, np.pi*2]]*2*n_qubits*(p+1))
initial_point = np.array([np.pi]*2*n_qubits*(p+1))

#initiate the iteration number
global current_iteration
current_iteration = 0

#get the vqe result
if sim_type == 'no_noisy_sim':

    loss_filename = hamiltonian_string_elements[-1] +'_vqe_noiseless_loss_' + sys.argv[1] + '.csv'
    params_filename = hamiltonian_string_elements[-1] +'_vqe_noiseless_params_' + sys.argv[1] + '.csv'
    
    #the callback function
    callback_func = lambda a, b, c, d, e: callback_function(a, b, c, d, e, loss_filename = loss_filename, params_filename = params_filename)
    spsa = SPSA(maxiter = 1000, callback = callback_func)

    objective_function = lambda c:vqe(c, paulis = paulis, coeffs = coeffs, shots = shots_sim, backend = simulator, mode = sim_type, loss_filename = None, params_filename = None)
    print('Starting the simulation')	
    vqe_result = spsa.optimize(num_vars = 2*n_qubits*(p+1), objective_function = objective_function, variable_bounds = bounds, initial_point = initial_point)
    
elif sim_type == 'noisy_sim':
    
    loss_filename = hamiltonian_string_elements[-1] +'_vqe_noisy_loss_' + sys.argv[1] + '.csv'
    params_filename = hamiltonian_string_elements[-1] +'_vqe_noisy_params_' + sys.argv[1] + '.csv'
        
    #the callback function
    callback_func = lambda a, b, c, d, e: callback_function(a, b, c, d, e, loss_filename = loss_filename, params_filename = params_filename)
    spsa = SPSA(maxiter = 1000, callback = callback_func)

    objective_function = lambda c: vqe(c, paulis = paulis, coeffs = coeffs, shots = shots_dev, backend = device, mode = sim_type, loss_filename = None, params_filename = None)
    print('Starting the simulation')	
    vqe_result = spsa.optimize(num_vars = 2*n_qubits*(p+1), objective_function = objective_function, variable_bounds = bounds, initial_point = initial_point)

else:
    raise Exception('Invalid simulation type!')



