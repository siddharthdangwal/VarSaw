from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, execute, IBMQ, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.models import BackendProperties
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

############################# Functions to generate marginal Paulis (with or without repeated terms) ###########################
def generate_marginal_paulis(paulis, meas_size):
    '''
    Args:
    paulis:
    meas_size:
    
    Returns: All marginal paulis. Eliminates repeated terms
    
    '''
    marginal_paulis = []
    length_pauli = len(paulis[0])
    
    for idx, el in enumerate(paulis):
        for i in range(length_pauli - meas_size + 1):
            marginal = i*'I' + el[i:i+meas_size] + (length_pauli - i - meas_size)*'I'
            if marginal not in marginal_paulis:
                marginal_paulis.append(marginal)
    
    return marginal_paulis

def generate_marginal_paulis2(paulis, meas_size):
    '''
    Args:
    paulis:
    meas_size:
    
    Returns: All marginal paulis. Does not eliminate repeated terms
    
    '''
    marginal_paulis = []
    length_pauli = len(paulis[0])
    
    for idx, el in enumerate(paulis):
        for i in range(length_pauli - meas_size + 1):
            marginal = i*'I' + el[i:i+meas_size] + (length_pauli - i - meas_size)*'I'
            marginal_paulis.append(marginal)
    
    return marginal_paulis
    
####################################### Functions to perform bayesian reconstruction of the main distribution using marginals ############################################
def bayesian_reconstruction(main_dist, marginal_dists, max_recur_count):
    '''
    Args:
    main_dist: The main distribution that needs to be changed
    marginal_dists: A dictionary of marginal distributions, where the key is the hamiltonian of the marginal
    max_recur_count: Maximum number of times the reconstruction function needs to be called
    
    Returns:
    The bayesian reconstructed output
    '''
    
    _baseline = copy.deepcopy(main_dist)
    _partials = copy.deepcopy(marginal_dists)
    
    reconstructed = reconstruct(_baseline, _partials)
    
    for i in range(max_recur_count):
        _baseline = [reconstructed]
        reconstructed = reconstruct(_baseline[0], _partials)
        if RF.Compute_Helinger(_baseline[0], reconstructed) < 0.0001:
            break
        
    return reconstructed

def reconstruct(main_dist, marginal_dists):
    '''
    Args:
    main_dist: The main distribution that needs to be changed
    marginal_dists: A dictionary of marginal distributions, where the key is the hamiltonian of the marginal
    
    Returns:
    The result of one iteration of bayesian update
    '''
    #normalize the main dictionary
    pout = RF.normalize_dict(copy.deepcopy(main_dist))
    
    for hamil in marginal_dists:
        ppost = bayesian_update(main_dist, RF.normalize_dict(marginal_dists[hamil]), hamil)
        pout = add_dists(ppost, pout)
        pout = RF.normalize_dict(pout)
    
    return pout

def add_dists(d1, d2):
    '''
    Args:
    d1: dictionary 1 
    d2: dictionary 2
    
    Returns:
    The sum of the two dictioanies
    '''
    updated_dict = {}
    
    keys = set(list(d1.keys()) + list(d2.keys()))
    all_values_sum = sum(list(d1.values())) + sum(list(d2.values()))
    
    for key in keys:
        if key in d1.keys() and key not in d2.keys():
            updated_dict[key] = d1[key]/all_values_sum
        elif key in d2.keys() and key not in d1.keys():
            updated_dict[key] = d2[key]/all_values_sum
        else:
            updated_dict[key] = (d2[key] + d1[key])/all_values_sum
    
    return updated_dict

def bayesian_update(main_dist, marginal_dist, hamil):
    '''
    Args:
    main_dist: The main distribution that needs to be changed
    marginal_dists: A dictionary of marginal distributions, where the key is the hamiltonian of the marginal
    hamil: The hamiltonian corresponding to the marignal_dist
    '''
    
    #track the location of 
    i_locs = []
    length = len(hamil)
    for idx, el in enumerate(hamil):
        if el != 'I':
            i_locs.append(length - idx - 1)
    
    #sort i_locs
    i_locs.sort()
    
    #created an updated marginal dict
    updated_marginal = {}
    marginal_keys = list(marginal_dist.keys())
    processed_keys = set() #the set of processed marginal keys
    
    for idx, key in enumerate(marginal_keys):
        if key in processed_keys:
            continue
        else:
            processed_keys.add(key)
            track_object = ''.join(list(np.array(list(key))[i_locs]))
            val = marginal_dist[key]
            processed_keys, extra_val = marginalize(marginal_dist, marginal_keys[idx+1:], track_object, processed_keys, i_locs)
            updated_marginal[track_object] = val + extra_val
    
    #print('updated marginal: ', updated_marginal)
    
    #now update the main dict
    updated_main_dict = copy.deepcopy(main_dist)
    for pry in updated_marginal:
        candidates = []
        running_sum = 0
        for prx in main_dist:
            if ''.join(list(np.array(list(prx))[i_locs])) == pry:
                running_sum += main_dist[prx]
                candidates.append([main_dist[prx], prx])
    
        for idx, el in enumerate(candidates):
            normalized_val = el[0]/running_sum
            updated_value = (normalized_val*updated_marginal[pry])/(1 - updated_marginal[pry])
            updated_main_dict[el[1]] = updated_value
    
    #normalize updated_main_dict
    updated_main_dict = RF.normalize_dict(updated_main_dict)
    
    return updated_main_dict

def marginalize(marginal_dict, keys, track_object, processed_keys, locs):
    '''
    Args:
    marginal_dict: The marginal dictionary
    relevant_keys: The potential candidates
    track_object: The relevant substring of the reference key
    processed_keys: a dictionary keeping track of which keys have been processed
    locs: Locations to trace similarity
    
    Returns:
    
    '''
    extra_val = 0
    for key in keys:
        if track_object == ''.join(list(np.array(list(key))[locs])):
            extra_val += marginal_dict[key]
            processed_keys.add(key)
    
    return processed_keys, extra_val
    
################################## Compute the expectation of the hamiltonian with JigSaw ############################## 
def compute_expectations_with_jigsaw(parameters, paulis, shots, meas_size, backend,  mode, opt_level, external_dists = None, dict_filename = None, noise_model = None, seed = 0):
    '''
    Args:
    parameters: The parameters of the VQE ansatz
    paulis: The paulis tensor hamiltonians
    backend: The backend on which the vqe is run
    noise_model: The noise model on which we want to run the simulation -- overrides the backend noise model if passed
    mode: Specifies if we have to run a noisy simulation or ideal simulation or run the circuit on a device
    shots: The total shots budget
    meas_size: An integer specifying the number of terms each new pauli hamiltonian consists of
    external_dists: A dictionary containing distributions for each term in the hamiltonian. If none, then compute
    the distributions, else use the ones given in external_dist
    
    Returns:
    expectations: A list of expectations for each circuit
    reconstructed_counts: A list of counts of the reconstructed circuit
    '''
    #define the simulator
    simulator = Aer.get_backend('qasm_simulator')

    #gloabl variable keeping track of iteration number
    global iter_num
    
    #get the new marginal paulis
    marginal_paulis = generate_marginal_paulis(paulis, meas_size)
    print('Number of marginal paulis: ', len(marginal_paulis))
    
    #the number of qubits in the ansatz
    n_qubits = len(paulis[0])
    
    #get all the original vqe circuits
    if external_dists == None:
        circuits = [vqe_circuit(n_qubits, parameters, pauli) for pauli in paulis]
    
    #get all the marginal vqe circuits
    marginal_circuits = [vqe_circuit(n_qubits, parameters, hamil) for hamil in marginal_paulis]
    marginal_circuits = {k:v for k, v in zip(marginal_paulis, marginal_circuits)}
    
    sim_device = None
    if mode == 'noisy_sim':
        sim_device = AerSimulator.from_backend(backend)
    
    #evaluate the original circuits
    if external_dists == None:
        if mode == 'no_noisy_sim' or mode == 'device_execution':
            job = execute(circuits, backend = backend, optimization_level = 3, shots = 8192)
            all_counts = job.result().get_counts()
        elif mode == 'noisy_sim':
            #tcircs = transpile(circuits, sim_device, optimization_level = 3)
            #result_noise = sim_device.run(tcircs, shots = shots_per_circuit_global).result()
            basis_gates = backend.configuration().to_dict()['basis_gates']
            coupling_map = backend.configuration().to_dict()['coupling_map']
            result_noise = execute(circuits, backend = simulator, noise_model = noise_model, coupling_map = coupling_map, basis_gates = basis_gates, optimization_level = 3, shots = 8192).result()
            #all_counts = result_noise.get_counts()
            all_counts = []
            for __, _id in enumerate(paulis):
                if _id == len(_id)*'I':
                    all_counts.append({len(_id)*'0':8192})
                else:
                    all_counts.append(result_noise.get_counts(__))
        else:
            raise Exception('Invalid circuit execution mode')
    
    #evaluate all the marginal circuits
    if mode == 'no_noisy_sim' or mode == 'device_execution':
        job = execute(list(marginal_circuits.values()), backend = backend, shots = 8192, optimization_level = 3)
        all_marginal_counts = job.result().get_counts()
    elif mode == 'noisy_sim':
        #tcircs = transpile(list(marginal_circuits.values()), sim_device, optimization_level = 3)
        #result_noise = sim_device.run(tcircs, shots = shots_per_circuit_marginal).result()
        basis_gates = backend.configuration().to_dict()['basis_gates']
        coupling_map = backend.configuration().to_dict()['coupling_map']
        result_noise = execute(list(marginal_circuits.values()), backend = simulator, noise_model = noise_model, coupling_map = coupling_map, basis_gates = basis_gates, optimization_level = 3, shots = 8192).result()
        #all_marginal_counts = result_noise.get_counts()
        all_marginal_counts = []
        for __, _id in enumerate(marginal_paulis):
            if _id == len(_id)*'I':
                all_marginal_counts.append({len(_id)*'0':8192})
            else:
                all_marginal_counts.append(result_noise.get_counts(__))
    else:
        raise Exception('Invalid circuit execution mode')
    
    all_marginal_counts = {k:v for k, v in zip(marginal_paulis, all_marginal_counts)}
    
    #reconstruct the original distributions
    reconstructed_counts = []
    for idx, pauli in enumerate(paulis):
        relevant_marginals = generate_marginal_paulis([pauli], meas_size)
        relevant_marginal_counts = {}
        for marginal in relevant_marginals:
            relevant_marginal_counts[marginal] = all_marginal_counts[marginal]
        
        if external_dists == None:
            reconstructed_counts.append(bayesian_reconstruction(all_counts[idx], relevant_marginal_counts, max_recur_count = 10))
        else:
            reconstructed_counts.append(bayesian_reconstruction(external_dists[idx], relevant_marginal_counts, max_recur_count = 10))
    
    #dump the reconstructed counts
    #if iter_num%100 == 0:
        #pickle.dump(reconstructed_counts, open(dict_filename[:-4] + '_seed_' + str(seed) + '_global_opt_level_' + str(opt_level) + '_iter_num_' + str(iter_num) + '_ro_only' + '.pkl', 'wb'))
    iter_num += 1

    #compute the expectations
    expectations = []
    
    for i, count in enumerate(reconstructed_counts):
        
        #initiate the expectation value to 0
        expectation_val = 0
        
        #compute the expectation
        for el in count.keys():
            sign = 1
            
            #change sign if there are an odd number of ones
            if el.count('1')%2 == 1:
                sign = -1
            
            expectation_val += sign*count[el]
        
        expectations.append(expectation_val)
    
    return expectations, reconstructed_counts

########################################## Compute VQE loss ###################################################### 
def compute_loss_with_jigsaw(parameters, paulis, coeffs, shots, meas_size, backend, mode, opt_level, external_dists, dict_filename = None, noise_model = None, seed = 0):
    '''
    Args:
    parameters: The parameters of the VQE ansatz
    paulis: The paulis tensor hamiltonians
    coeffs: The coefficients corresponding to each pauli tensor
    backend: The backend on which the vqe is run
    noise_model: The noise model on which we want to run the simulation -- overrides the backend noise model if passed
    mode: Specifies if we have to run a noisy simulation or ideal simulation or run the circuit on a device
    shots: The number of shots for which each circuit is executed
    external_dists: A dictionary containing distributions for each term in the hamiltonian. If none, then compute
    the distributions, else use the ones given in external_dist
    
    Returns:
    The loss for the entire VQE hamiltonian
    '''
    
    expectations, reconstructed_counts = compute_expectations_with_jigsaw(parameters, paulis,  shots, meas_size, backend, mode, opt_level, external_dists = external_dists, dict_filename = dict_filename, noise_model = noise_model, seed = seed)
    
    loss = 0
    
    for i, el in enumerate(expectations):
        loss += coeffs[i]*el
    
    return loss, reconstructed_counts

####################################################### VQE function ###################################################################
def vqe_with_jigsaw(parameters, paulis, coeffs, shots, meas_size, backend, external_dists = None, mode = 'no_noisy_sim', opt_level = 1, loss_filename = None, params_filename = None, dict_filename = None, noise_model = None, seed = 0):
    '''
    Args:
    parameters: The parameters of the VQE ansatz
    paulis: The paulis tensor hamiltonians
    coeffs: The coefficients corresponding to each pauli tensor
    backend: The backend on which the vqe is run
    noise_model: The noise model on which we want to run the simulation -- overrides the backend noise model if passed
    external_dists: A dictionary containing distributions for each term in the hamiltonian. If none, then compute
    the distributions, else use the ones given in external_distA dictionary containing distributions for each term in the hamiltonian. If none, then compute
    the distributions, else use the ones given in external_dist
    mode: Specifies if we have to run a noisy simulation or ideal simulation or run the circuit on a device
    shots: The number of shots for which each circuit is executed
    
    Returns:
    Loss for one iteration of the A dictionary containing distributions for each term in the hamiltonian. If none, then compute
    the distributions, else use the ones given in external_distVQE
    '''
    
    #number of qubits in the VQE ansatz
    n_qubits = len(paulis[0])
    
    #making sure that the number of elements in each pauli tensor is the same
    for i in paulis:
        assert len(i) == n_qubits
    
    loss, reconstructed_counts =  compute_loss_with_jigsaw(parameters, paulis, coeffs, shots, meas_size, backend, mode, opt_level, external_dists = external_dists, dict_filename = dict_filename, noise_model = noise_model, seed = seed)
    print('Loss computed by VQE is: {}'.format(loss))
    
    if not (loss_filename == None):
        with open(loss_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([loss])

    if not(params_filename == None):
        with open(params_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(parameters)

    return loss,reconstructed_counts

############################# Wrapper Function to compute VQE with JigSaw where the distribution from the previous iteration is reused ###########################
def vqe_with_jigsaw_wrapper(parameters, paulis, coeffs, shots, meas_size, backend, mode = 'no_noisy_sim', opt_level = 1, loss_filename = None, params_filename = None, dict_filename = None, noise_model = None, seed = 0):
    
    #get the reconstructed counts and the loss value
    global external_dists
    loss, reconstructed_counts = vqe_with_jigsaw(parameters, paulis = paulis, coeffs = coeffs, shots = shots_dev, meas_size = meas_size, backend = backend, external_dists = external_dists, mode = mode, opt_level = opt_level, loss_filename = loss_filename, params_filename = params_filename, dict_filename = dict_filename, noise_model = noise_model, seed = seed)
    external_dists = reconstructed_counts
    
    return loss

########################## Function to scale backend properties #####################################################

def alter_properties_dict(backend, scaling_factor):
    '''
    Args:
    backend: The qiskit backend whose properties dictionary we have to modify
    scaling_factor: The factor by which we want to scale the properties
    
    Returns:
    A new dictionary with scaled noise parameters 
    '''
    import copy
    
    conf_dict = backend.configuration().to_dict()
    prop_dict = backend.properties().to_dict()
    
    new_prop_dict = copy.deepcopy(prop_dict)
    
    #alter the qubit based properties - readout error rates and t1, t2 times
    qubits = prop_dict['qubits']
    props_to_change = ['T1', 'T2', 'readout_error', 'prob_meas0_prep1', 'prob_meas1_prep0']
    for idx, qubit in enumerate(qubits):
        
        #a qubit is represented by 8 properties
        assert len(qubit) == 8
        
        for idx2, prop in enumerate(qubit):
            if prop['name'] in props_to_change:
                
                if prop['name'] == 'T1' or prop['name'] == 'T2':
                    new_prop_value = prop['value']*(1/scaling_factor)
                    #print(prop['name'], prop['value'], new_prop_value)
                else:
                    new_prop_value = prop['value']*scaling_factor
                    #print(prop['name'], prop['value'], new_prop_value)
                
                new_prop_dict['qubits'][idx][idx2]['value'] = new_prop_value
    
    #alter the gate based properties - gate error
    gates = prop_dict['gates']
    for idx, gate in enumerate(gates):
        
        for idx2 in range(len(gate['parameters'])):
            
            #a gate is represented by a dicrionary of 4 items
            gate_error_dict = gate['parameters'][idx2]

            #change the value of the gate error
            if gate_error_dict['name'] == 'gate_error':
                new_gate_error_val = gate_error_dict['value']*scaling_factor
                new_prop_dict['gates'][idx]['parameters'][idx2]['value'] = new_gate_error_val
                #print(gate_error_dict['name'], gate_error_dict['value'], new_gate_error_val)
    
    
    return new_prop_dict

################## Run VQE #####################
tol = 1e-5

simulator = Aer.get_backend('qasm_simulator')
device = FakeMumbai()

shots_sim = 9999
shots_dev = 8192

#get the seed as the first argument
seed = int(sys.argv[1])
np.random.seed(seed)

p = 2

#get the optimization level
opt_level = int(sys.argv[2])

#get the system hamiltonian
hamiltonian_string = sys.argv[3]
hamiltonian_string_elements = hamiltonian_string.split('/')
hamil = parseHamiltonian(hamiltonian_string)

#get the number of qubits in the hamiltonian
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
budget = 1000


#create the noise model 
scaling_factor = float(sys.argv[4])
scaled_props_dict = alter_properties_dict(backend = device, scaling_factor = scaling_factor)
scaled_props = BackendProperties.from_dict(scaled_props_dict)
scaled_noise_model = NoiseModel.from_backend(scaled_props, gate_error = False, thermal_relaxation = False)

#get the vqe result
if opt_level == 1:

    loss_filename = hamiltonian_string_elements[-1] + '_vqe_with_jigsaw_ro_opt1_loss_' + sys.argv[1] + '_scaling_factor_' + sys.argv[4] + '.csv'
    params_filename = hamiltonian_string_elements[-1] + '_vqe_with_jigsaw_ro_opt1_params_' + sys.argv[1] + '_scaling_factor_' + sys.argv[4] + '.csv'
    global_dict_name_prefix = hamiltonian_string_elements[-1]
    iter_num = 0
    
    objective_function = lambda c:vqe_with_jigsaw(c, paulis = paulis, coeffs = coeffs, shots = shots_dev, meas_size = 2, backend = device, external_dists = None, mode = 'noisy_sim', opt_level = 1, loss_filename = loss_filename, params_filename = params_filename, dict_filename = global_dict_name_prefix, noise_model = scaled_noise_model, seed = seed)[0]
    vqe_result_jigsaw = minimize(objective_function, initial_point, bounds, budget, method = 'imfil')

elif opt_level == 2:

    #converge second iteration onwards
    loss_filename = hamiltonian_string_elements[-1] + '_vqe_with_jigsaw_ro_opt2_loss_' + sys.argv[1] + '_scaling_factor_' + sys.argv[4] + '.csv'
    params_filename = hamiltonian_string_elements[-1] + '_vqe_with_jigsaw_ro_opt2_params_' + sys.argv[1] + '_scaling_factor_' + sys.argv[4] + '.csv'
    global_dict_name_prefix = hamiltonian_string_elements[-1]
    iter_num = 0

    #first iteration normally
    loss, reconstructed_counts = vqe_with_jigsaw(initial_point, paulis = paulis, coeffs = coeffs, shots = shots_dev, meas_size = 2, backend = device, external_dists = None, mode = 'noisy_sim', opt_level = 2, loss_filename = loss_filename, params_filename = params_filename, dict_filename = global_dict_name_prefix, seed = seed)
    external_dists = reconstructed_counts

    objective_function = lambda c:vqe_with_jigsaw(c, paulis = paulis, coeffs = coeffs, shots = shots_dev, meas_size = 2, backend = device, external_dists = None, mode = 'noisy_sim', opt_level = 1, loss_filename = loss_filename, params_filename = params_filename, dict_filename = global_dict_name_prefix, noise_model = scaled_noise_model, seed = seed)[0]
    vqe_result_jigsaw = minimize(objective_function, initial_point, bounds, budget, method = 'imfil')

else:
    raise Exception('Invalid optimization level. Optimization level can only be 1 or 2.')




