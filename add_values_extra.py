#! /usr/bin/python3
from pennylane import numpy as np
import pennylane as qml

numbers = [5, 7, 8, 9, 1]

address_register_size = len(numbers) # Number of bits for the one-hot encoding
data_register_size = 5 # Number of bits needed for the binary encoding

# Different registers for the quantum circuit
address_register = range(address_register_size)
data_register = range(address_register_size, address_register_size + data_register_size)
ancilla = address_register_size + data_register_size

def oracle():
    # QFT applied to wires 5, 6, 7, 8, 9
    qml.QFT(wires=data_register)
    
    # Addition
    for i, number in enumerate(numbers):
        direction = number / (2 ** data_register_size)
        for wire in reversed(data_register):
            qml.ControlledPhaseShift(2 * np.pi * direction, wires=[i, wire])
            direction *= 2 

    qml.QFT(wires=data_register).inv()

    # Check if addition gives 16 (binary representation: 10000)
    qml.MultiControlledX(control_wires=data_register, wires=ancilla, control_values="10000")

    qml.QFT(wires=data_register)

    # Reverse everything 
    for i, number in enumerate(reversed(numbers)):
        direction = -number / (2 ** data_register_size)
        for wire in reversed(data_register):
            qml.ControlledPhaseShift(2 * np.pi * direction, wires=[data_register_size - i - 1, wire])
            direction *= 2

    qml.QFT(wires=data_register).inv()

dev = qml.device('default.qubit', wires=11)

@qml.qnode(dev)
def circuit(num_iterations=1):
    # Phase kickback qubit
    qml.PauliX(wires=ancilla)
    qml.Hadamard(wires=ancilla)

    # Equal superposition of all possible summations
    for wire in address_register:
        qml.Hadamard(wires=wire)

    for _ in range(num_iterations):
        oracle()
        qml.templates.GroverOperator(wires=address_register)

    return qml.probs(wires=address_register)

def binary_list(m):
    arr = []
    for i in range(address_register_size):
        arr = [m % 2] + arr
        m = m // 2
    return arr

def decode(probs):
    # This function decodes the probability distribution returned by circuit()
    # to find the possible solutions
    combinations = []

    max_prob = max(probs)
    max_positions = [i for i, j in enumerate(probs) if np.allclose(j, max_prob)]
    for max_position in max_positions:
        combination = []
        encoding = binary_list(max_position)
        for i, bit in enumerate(encoding):
            if bit == 1:
                combination.append(numbers[i])

        if not len(combination) == 0:
            combinations.append(combination)

    return combinations

if __name__ == "__main__": 
    probs = circuit(num_iterations=7)
    combinations = decode(probs)
    print(combinations)