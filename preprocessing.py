import lzma
import pandas as pd
import seaborn as sns
from rdkit import Chem
from pyscf import gto, scf
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
# %%

def select_protein_class(data, protein_data, classes=[{"l5": "Adenosine receptor"}]): 
    """ 
    Filter the input dataset based on specific protein classifications. 
    Parameters: 
        data (DataFrame): A DataFrame containing biological data. It 
            must include the 'target_id' column. 
        protein_data (DataFrame): A DataFrame containing protein  
            classification data, with required columns 'target_id' and  
            'Classification'. 
        classes (list of dict): A list of dictionaries specifying      
            filters for different classification levels.  
    Returns: 
        DataFrame: A filtered DataFrame containing only the rows from  
            'data' whose 'target_id' matches those in the filtered  
            protein_data. The returned DataFrame is merged with the  
            corresponding 'Classification' and 'Organism' columns from  
            protein_data. 
    """ 
    # If no classification filters are provided, return the original dataset. 
    if not classes: 
        return data 
    # Split the 'Classification' column into separate levels using "->"as the delimiter. 
    # Ensure that we have exactly 8 columns, filling in any missing levels with empty strings. 
    split_classifications = ( 
        protein_data["Classification"] 
        .str.split("->", expand=True) 
        .reindex(columns=range(8), fill_value="") 
    ) 
    protein_data[["l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8"]] =     split_classifications 
    # Define a helper function to determine whether a given row matches all provided classification filters. 
    def match_classification(row): 
        for class_filter in classes: 
            if all( 
                str(row.get(level, "")).lower() == str(value).lower() 
                for level, value in class_filter.items() 
            ): 
                return True 
        return False 

    # Apply the matching function to filter protein_data based on the specified classifications. 
    filtered_proteins = protein_data[protein_data.apply( 
                        match_classification, axis=1)] 
    target_ids = filtered_proteins["target_id"].unique() 
    # Return the filtered rows from 'data' that match the target_ids and  merge in the 'Classification' and 'Organism' details. 
    return data[data["target_id"].isin(target_ids)].merge( 
        filtered_proteins[["target_id", "Classification", "Organism"]], 
        on="target_id")

# Validate SMILES strings
def validate(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


# %%

# LOAD DATA
dataset_file = "05.5_combined_set_without_stereochemistry.tsv.xz"
with lzma.open(dataset_file, "rt") as f:
    df = pd.read_csv(f, sep="\t", low_memory=False)

print(f"Dimensioni del dataset: {df.shape}")
print("Prime righe del dataset:")
print(df.head())

protein_file = "05.5_combined_set_protein_targets.tsv.xz"
with lzma.open(protein_file, "rt") as f:
    protein_data = pd.read_csv(f, sep="\t", low_memory=False)

print(f"Dimensioni del dataset: {protein_data.shape}")
print("Prime righe del dataset:")
print(protein_data.head())

print("Dataset Information:")
print(df.info())
print("First few rows of the dataset:")
print(df.head())

target = "Adenosine receptor"
adenosine = select_protein_class(df, protein_data, classes=[{"l5": target}])
adenosine = adenosine[adenosine['Organism'] == 'Homo sapiens (Human)']

target_id = adenosine["target_id"].unique()
df_clean = df.dropna(subset=["SMILES", "pchembl_value"]).drop_duplicates(
    subset=["Activity_ID"]
)
print("Dataset size after cleaning:", len(df_clean))

# Apply SMILES validation and remove invalid entries
df_clean["valid_smiles"] = df_clean["SMILES"].apply(validate)
df_clean = df_clean[df_clean["valid_smiles"]].drop(columns=["valid_smiles"])
print("Dataset size after SMILES validation:", len(df_clean))
df_clean["target_id"].value_counts()

delimiter = "_"
df_target = df_clean[df_clean["target_id"].isin(target_id)]
print(f"Number of molecules with target{target_id}: {len(df_target)}")

df_quality = df_target[df_target["Quality"].str.lower() != "low"]
print(f"Number of molecules with high quality: {len(df_quality)}")


morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)


# Function to calculate molecular descriptors
def calc_desc(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Convert to explicit bit vector
        fp_bit_vect = morgan_gen.GetFingerprint(mol)
        return {
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "RotB": Descriptors.NumRotatableBonds(mol),
            "FP": list(fp_bit_vect.ToBitString()),  # Convert to list of bits
        }
    return None


# Apply the function to the dataset
df_quality["Desc"] = df_quality["SMILES"].apply(calc_desc)
df_desc = pd.json_normalize(df_quality["Desc"]).set_index(df_quality.index)

numeric_cols = df_desc.drop(columns=["FP"])
correlation_matrix = numeric_cols.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Molecular Descriptors")
plt.show()

assert len(df_desc["FP"].iloc[0]) == 2048
fp_df = (
    pd.DataFrame(
        df_desc["FP"].tolist(),
        columns=[f"FP_{i}" for i in range(len(df_desc["FP"].iloc[0]))],
    )
    .astype(int)
    .set_index(df_desc.index)
)

df_desc_exp = pd.concat([df_desc.drop(columns=["FP"]), fp_df], axis=1)
df_final = pd.concat(
    [
        df_quality[["Year", "target_id", "pchembl_value_Mean"]],
        df_desc_exp,
    ],
    axis=1,
)



def calculate_hf_descriptors(index, smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol) 
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())  
        AllChem.MMFFOptimizeMolecule(mol)  

        
        coords = mol.GetConformer().GetPositions()
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

        
        mol_pyscf = gto.M(
            atom=[(symbols[i], tuple(coords[i])) for i in range(len(symbols))],
            basis="sto-3g",
        )

        # SCF con Hartree-Fock
        mf = scf.RHF(mol_pyscf)  # Restricted Hartree-Fock
        mf.kernel()  # COmpute SCF

        # Compute HOMO, LUMO e gap
        mo_energies = mf.mo_energy
        num_electrons = mol_pyscf.nelectron
        homo_index = num_electrons // 2 - 1
        lumo_index = homo_index + 1
        homo_energy = mo_energies[homo_index]
        lumo_energy = mo_energies[lumo_index]
        homo_lumo_gap = lumo_energy - homo_energy
        dipole = mf.dip_moment()

        # Return results
        return {
            "Idx": index,
            "SCF Energy": mf.e_tot,
            "HOMO": homo_energy,
            "LUMO": lumo_energy,
            "HOMO-LUMO Gap": homo_lumo_gap,
            "Dipole Moment X": dipole[0],
            "Dipole Moment Y": dipole[1],
            "Dipole Moment Z": dipole[2],
            "Dipole Moment Magnitude": (
                dipole[0] ** 2 + dipole[1] ** 2 + dipole[2] ** 2
            )
            ** 0.5,
        }
    except Exception as e:
        print(f"Errore per SMILES: {smiles}, {e}")
        return None



# Parallel execution of the calculate_hf_descriptors function
results = Parallel(n_jobs=16)(
    delayed(calculate_hf_descriptors)(index, smile)
    for index, smile in df_quality["SMILES"].items()
)


res = [d for d in results if d and isinstance(d, dict)]
df_quantum = (pd.DataFrame(res)).set_index("Idx")

df_quantum = df_quantum[
    [
        "SCF Energy",
        "HOMO",
        "LUMO",
        "Dipole Moment X",
        "Dipole Moment Y",
        "Dipole Moment Z"
    ]
]

df_final_quantum = pd.concat([df_final, df_quantum], axis=1)


df_final_quantum = df_final_quantum.dropna()