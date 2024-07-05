# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:46:02 2024

@author: BM109X32G-10GPU-02
"""

from collections import defaultdict
import numpy as np
from rdkit import Chem
import torch
import pickle
import pandas as pd
atom_dict1 = defaultdict(lambda: len(atom_dict1))
bond_dict1 = defaultdict(lambda: len(bond_dict1))
fingerprint_dict1 = defaultdict(lambda: len(fingerprint_dict1))
edge_dict1 = defaultdict(lambda: len(edge_dict1))
radius=1
bond_dict2 = defaultdict(lambda: len(bond_dict2))
fingerprint_dict2 = defaultdict(lambda: len(fingerprint_dict2))
edge_dict2 = defaultdict(lambda: len(edge_dict2))
atom_dict2 = defaultdict(lambda: len(atom_dict2))

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)
device = torch.device('cuda')        
# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     print('The code uses a GPU!')
# else:
#     device = torch.device('cpu')
#     print('The code uses a CPU...')
# 	
def create_atoms(mol, atom_dict):
    """Transform the atom types in a molecule (e.g., H, C, and O)
    into the indices (e.g., H=0, C=1, and O=2).
    Note that each atom index considers the aromaticity.
    """
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol, bond_dict):
    """Create a dictionary, in which each key is a node ID
    and each value is the tuples of its neighboring node
    and chemical bond (e.g., single and double) IDs.
    """
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(radius, atoms, i_jbond_dict,
                         fingerprint_dict, edge_dict):
    """Extract the fingerprints from a molecular graph
    based on Weisfeiler-Lehman algorithm.
    """

    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges.
            The updated node IDs are the fingerprint IDs.
            """
            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(fingerprint_dict[fingerprint])

            """Also update each edge ID considering
            its two nodes on both sides.
            """
            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))

            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

    return np.array(nodes)

def create_dataset(filename,path,dataname):
    dir_dataset = path+dataname
    print(filename)
    """Load a dataset."""
    with open(dir_dataset + filename, 'r') as f:
        smiles_property = f.readline().strip().split()
        data_original = f.read().strip().split('\n')

        """Exclude the data contains '.' in its smiles.排除含.的数据"""
    data_original = [data for data in data_original
                        ]
    dataset1 = []
    dataset2 = []
    for data in data_original:



            try:
                smiles1,smiles2, property = data.strip().split()
                mol1 = Chem.AddHs(Chem.MolFromSmiles(smiles1))
                mol2 = Chem.AddHs(Chem.MolFromSmiles(smiles2))
            except:
                print(data)
                continue
            atoms1 = create_atoms(mol1, atom_dict1)
            molecular_size1 = len(atoms1)
            i_jbond_dict1 = create_ijbonddict(mol1, bond_dict1)
            fingerprints1 = extract_fingerprints(radius, atoms1, i_jbond_dict1,
                                                    fingerprint_dict1, edge_dict1)
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            adjacency1 = np.float32((Chem.GetAdjacencyMatrix(mol1)))
    #Transform the above each data of numpy to pytorch tensor on a device (i.e., CPU or GPU).
            fingerprints1 = torch.LongTensor(fingerprints1).to(device)
            adjacency1 = torch.FloatTensor(adjacency1).to(device)
            property = torch.FloatTensor([[float(property)]]).to(device)
            dataset1.append((smiles1,fingerprints1, adjacency1, molecular_size1, property))

            atoms2 = create_atoms(mol2, atom_dict2)
            molecular_size2 = len(atoms2)
            i_jbond_dict2 = create_ijbonddict(mol2, bond_dict2)
            fingerprints2 = extract_fingerprints(radius, atoms2, i_jbond_dict2,
                                                    fingerprint_dict2, edge_dict2)
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            adjacency2 = np.float32((Chem.GetAdjacencyMatrix(mol2)))
    #Transform the above each data of numpy to pytorch tensor on a device (i.e., CPU or GPU).
            fingerprints2 = torch.LongTensor(fingerprints2).to(device)
            adjacency2 = torch.FloatTensor(adjacency2).to(device)
            property = torch.FloatTensor([[float(property)]]).to(device)
            dataset2.append((smiles2,fingerprints2, adjacency2, molecular_size2, property))

    dump_dictionary(fingerprint_dict1, dir_dataset +dataname+ '-fingerprint_dict1.pickle')
    dump_dictionary(atom_dict1, dir_dataset +dataname+ '-atom_dict1.pickle')
    dump_dictionary(bond_dict1, dir_dataset  +dataname+ '-bond_dict1.pickle')
    dump_dictionary(edge_dict1, dir_dataset +dataname+ '-edge_dict1.pickle')
    dump_dictionary(fingerprint_dict2, dir_dataset +dataname+ '-fingerprint_dict2.pickle')
    dump_dictionary(atom_dict2, dir_dataset +dataname+ '-atom_dict2.pickle')
    dump_dictionary(bond_dict2, dir_dataset  +dataname+ '-bond_dict2.pickle')
    dump_dictionary(edge_dict2, dir_dataset +dataname+ '-edge_dict2.pickle')    

    return dataset1,dataset2

def create_dataset_randomsplit(x,y,path,dataname):
    dir_input = path + 'SMRT-'
    with open(dir_input + 'atom_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            atom_dict.get(k)
            atom_dict[k]=c[k]
    with open(dir_input+ 'bond_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            bond_dict.get(k)
            bond_dict[k]=c[k]

    with open(dir_input + 'edge_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            edge_dict.get(k)
            edge_dict[k]=c[k]

    with open(dir_input + 'fingerprint_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            fingerprint_dict.get(k)
            fingerprint_dict[k]=c[k]    
    dataset = []  
    for i in range(len(x)):
        smiles=x[i]
        property=y[i]         
        """Create each data with the above defined functions."""
        mol = Chem.MolFromInchi(smiles)     
        mol = Chem.AddHs(Chem.MolFromInchi(smiles))
        atoms = create_atoms(mol, atom_dict)
        molecular_size = len(atoms)
        i_jbond_dict = create_ijbonddict(mol, bond_dict)
        fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                fingerprint_dict, edge_dict)
        adjacency = np.float32((Chem.GetAdjacencyMatrix(mol)))
#Transform the above each data of numpy to pytorch tensor on a device (i.e., CPU or GPU).
        fingerprints = torch.LongTensor(fingerprints).to(device)
        adjacency = torch.FloatTensor(adjacency).to(device)
        property = torch.FloatTensor([[float(property)]]).to(device)

        dataset.append((smiles,fingerprints, adjacency, molecular_size, property))
    dir_dataset=path
    dump_dictionary(fingerprint_dict, dir_dataset +dataname+ '-fingerprint_dict.pickle')
    dump_dictionary(atom_dict, dir_dataset +dataname+ '-atom_dict.pickle')
    dump_dictionary(bond_dict, dir_dataset  +dataname+ '-bond_dict.pickle')
    dump_dictionary(edge_dict, dir_dataset +dataname+ '-edge_dict.pickle')
    return dataset




def transferlearning_dataset_predict(x,path):
    dir_input = path+'SMRT-'
    with open(dir_input + 'atom_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            atom_dict.get(k)
            atom_dict[k]=c[k]
    with open(dir_input+ 'bond_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            bond_dict.get(k)
            bond_dict[k]=c[k]

    with open(dir_input + 'edge_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            edge_dict.get(k)
            edge_dict[k]=c[k]

    with open(dir_input + 'fingerprint_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            fingerprint_dict.get(k)
            fingerprint_dict[k]=c[k]
    dataset = []
    for i in range(len(x)):
        smiles=x[i]
        """Create each data with the above defined functions."""       
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue           
        else:
            smi = Chem.MolToSmiles(mol)            
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        atoms = create_atoms(mol, atom_dict)
        molecular_size = len(atoms)
        i_jbond_dict = create_ijbonddict(mol, bond_dict)
        fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                fingerprint_dict, edge_dict)
        adjacency = np.float32((Chem.GetAdjacencyMatrix(mol)))
#Transform the above each data of numpy to pytorch tensor on a device (i.e., CPU or GPU).
        fingerprints = torch.LongTensor(fingerprints).to(device)
        adjacency = torch.FloatTensor(adjacency).to(device)
        dataset.append((smiles,fingerprints, adjacency, molecular_size)) 
    return dataset