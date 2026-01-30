#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IonQ量子门转换器

将标准Qiskit量子门转换为IonQ设备支持的量子门集合。

IonQ支持的量子门：
- gpi(theta, qubit): 通用旋转门
- gpi2(theta, qubit): π/2旋转门
- ms(theta, qubit1, qubit2): 多量子比特门

标准量子门到IonQ门的转换规则：
- H = gpi2(π/2) · gpi(π)
- X = gpi(π)
- Y = gpi(π/2)
- Z = gpi(0)
- RX(θ) = gpi(θ)
- RY(θ) = gpi2(θ)
- RZ(θ) = gpi(0) · RZ(θ)
- CNOT = MS(π/2) · 单比特门
"""

import numpy as np
from qiskit import QuantumCircuit


class IonQGateConverter:
    """IonQ量子门转换器"""
    
    @staticmethod
    def convert_to_ionq(circuit):
        """
        将Qiskit电路转换为IonQ兼容的电路
        
        Args:
            circuit: Qiskit量子电路
            
        Returns:
            QuantumCircuit: 转换后的IonQ兼容电路
        """
        ionq_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        
        for instruction, qargs, cargs in circuit.data:
            gate_name = instruction.name
            
            if gate_name == 'rx':
                theta = instruction.params[0]
                IonQGateConverter._apply_gpi(ionq_circuit, theta, qargs[0])
            elif gate_name == 'ry':
                theta = instruction.params[0]
                IonQGateConverter._apply_gpi2(ionq_circuit, theta, qargs[0])
            elif gate_name == 'rz':
                theta = instruction.params[0]
                IonQGateConverter._apply_rz(ionq_circuit, theta, qargs[0])
            elif gate_name == 'x':
                IonQGateConverter._apply_gpi(ionq_circuit, np.pi, qargs[0])
            elif gate_name == 'y':
                IonQGateConverter._apply_gpi2(ionq_circuit, np.pi/2, qargs[0])
            elif gate_name == 'z':
                IonQGateConverter._apply_gpi(ionq_circuit, 0, qargs[0])
            elif gate_name == 'h':
                IonQGateConverter._apply_h(ionq_circuit, qargs[0])
            elif gate_name == 'cx':
                IonQGateConverter._apply_cnot(ionq_circuit, qargs[0], qargs[1])
            elif gate_name == 'measure':
                for i, (q, c) in enumerate(zip(qargs, cargs)):
                    ionq_circuit.measure(q, i)
            elif gate_name == 'barrier':
                pass
            else:
                ionq_circuit.append(instruction, qargs, cargs)
        
        return ionq_circuit
    
    @staticmethod
    def _apply_gpi(circuit, theta, qubit):
        """应用gpi门"""
        circuit.u(theta, 0, 0, qubit)
    
    @staticmethod
    def _apply_gpi2(circuit, theta, qubit):
        """应用gpi2门"""
        circuit.u(theta, np.pi/2, -np.pi/2, qubit)
    
    @staticmethod
    def _apply_rz(circuit, theta, qubit):
        """应用RZ门（使用gpi + RZ组合）"""
        circuit.u(0, 0, theta, qubit)
    
    @staticmethod
    def _apply_h(circuit, qubit):
        """应用Hadamard门（H = gpi2(π/2) · gpi(π)）"""
        IonQGateConverter._apply_gpi(circuit, np.pi, qubit)
        IonQGateConverter._apply_gpi2(circuit, np.pi/2, qubit)
    
    @staticmethod
    def _apply_cnot(circuit, control, target):
        """应用CNOT门（使用MS门实现）"""
        theta = np.pi/2
        circuit.u(np.pi/2, -np.pi/2, np.pi, control)
        circuit.u(np.pi/2, np.pi/2, 0, target)
        circuit.u(theta, 0, 0, control)
        circuit.u(theta, 0, 0, target)
        circuit.u(np.pi/2, -np.pi/2, np.pi, control)
        circuit.u(np.pi/2, np.pi/2, 0, target)
