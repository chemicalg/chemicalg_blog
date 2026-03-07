## Decoding Molecular Complexity: nSPS and Assembly Theory

Companion notebook for the blog post at [chemicalg.github.io](https://chemicalg.github.io).

This notebook compares two quantitative complexity metrics — **CM\*** (Proudfoot, 2017) and **nSPS** (Waldmann, 2023) — applied to the same set of drug candidates used in the original [Molecular Complexity post](https://chemicalg.github.io/2025/03/27/MolecularComplexity.html).

**References:**
- Proudfoot (2017): https://doi.org/10.1016/j.bmcl.2017.03.008
- Ertl, Schuhmann & Waldmann (2023): https://doi.org/10.1021/acs.jmedchem.3c01024
- Marshall, Cronin et al. (2021): https://doi.org/10.1038/s41467-021-23258-x

### 1. Imports


```python
import numpy as np
import pandas as pd

from collections import Counter
from typing import Dict, Iterator, List, Tuple

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.SpacialScore import SPS  # built-in since RDKit 2023.09

from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import io
```

### 2. Load Data

We reuse the same compound set from the original Molecular Complexity post.
The Excel file lives in the sibling directory `2025-03-27-MolecularComplexity/`.


```python
data = pd.read_excel('../2025-03-27-MolecularComplexity/data.xlsx', header=0)

data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>SMILES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RPT193</td>
      <td>ClC1=C([C@@H](C)NC2=NC(N3CC([C@@H]4CN([C@@H]5C...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GDC-0853</td>
      <td>O=C1C(NC2=CC=C(N3[C@@H](C)CN(C4COC4)CC3)C=N2)=...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Orfoglipron</td>
      <td>O=C(N1CCC2=NN(C3=CC(C)=C(F)C(C)=C3)C(N4C(N(C5=...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GS-6207</td>
      <td>CC(S(C)(=O)=O)(C)C#CC1=NC(C(CC2=CC(F)=CC(F)=C2...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PF-07258669</td>
      <td>CC1=C(C2=NC=CC=N2)C=C(CC[C@]3(CN(C([C@@H](C4=C...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>14</td>
      <td>CC(C=C1CN)=NN1[C@H]2C[C@H](C3=CC4=CC(C)=CC=C4N...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PC0371</td>
      <td>O=C1NC(C2=CC=C(OC(F)(F)F)C=C2)=NC13CCN(S(CCC4=...</td>
    </tr>
  </tbody>
</table>
</div>



### 3. CM* Implementation (Proudfoot, 2017)

The CM\* metric rewards molecular environments that are locally complex — diverse bond paths emanating from each heavy atom.

**Per-atom complexity:**
$$C_A = -\sum_i p_i \log_2 p_i + \log_2 N$$

where $p_i$ is the fractional occurrence of each path type and $N$ is the total number of paths from that atom.

**Molecular complexity:**
$$CM^* = \log_2 \left( \sum_A 2^{C_A} \right)$$

Identical implementation to the original Molecular Complexity post — it hasn't changed.


```python
AtomType = Tuple[str, int, int]
Atom = Tuple[int, AtomType]
AtomDict = Dict[Atom, List[Atom]]


def _non_h_items(data: Dict[Atom, any]) -> Iterator[Tuple[Atom, any]]:
    """
    Generator for non-H items from a dictionary where the keys are atom tuples.
    Expected keys: (index, (symbol, total degree, non-h degree))
    """
    for key, val in data.items():
        if key[1][0] != 'H':
            yield key, val


def _collect_atom_paths(neighbors: AtomDict) -> List[List[tuple]]:
    """Returns list of atom paths for each atom (1 and 2 bonds away)."""
    atom_paths = []
    for atom, nbs in _non_h_items(neighbors):
        paths = []
        for nb in nbs:
            if nb[1][0] == 'H' or neighbors[nb] == [atom]:
                paths.append((atom[1], nb[1]))
            else:
                paths.extend((atom[1], nb[1], nb2[1]) for nb2 in neighbors[nb] if nb2 != atom)
        atom_paths.append(paths)
    return atom_paths


def get_atom_type(atom: Chem.rdchem.Mol) -> AtomType:
    """Return a tuple describing the atom type: (symbol, total_degree, non_h_degree)."""
    symbol = atom.GetSymbol()
    degree = atom.GetTotalDegree()
    h_count = atom.GetTotalNumHs(includeNeighbors=True)
    non_h = degree - h_count
    return (symbol, degree, non_h)


def fractional_occurrence(data: list) -> np.ndarray:
    """Calculate the fractional occurrence of unique items in the input list."""
    counter = Counter(data)
    counts = np.array(list(counter.values()))
    return counts / len(data)


def calculate_cm_star(mol: Chem.rdchem.Mol) -> float:
    """
    Calculate the CM* molecular complexity metric (Proudfoot, 2017).
    https://doi.org/10.1016/j.bmcl.2017.03.008
    """
    atoms = [(atom.GetIdx(), get_atom_type(atom)) for atom in mol.GetAtoms()]
    neighbors = {
        atom: [atoms[neighbor.GetIdx()] for neighbor in mol.GetAtomWithIdx(atom[0]).GetNeighbors()]
        for atom in atoms
    }
    atom_paths = _collect_atom_paths(neighbors)
    cas = np.zeros(len(atom_paths))
    for i, paths in enumerate(atom_paths):
        total_paths = len(paths)
        pi = fractional_occurrence(paths)
        cas[i] = -np.sum(pi * np.log2(pi)) + np.log2(total_paths)
    cm_star = np.log2(np.sum(2**cas))
    return float(cm_star)


def cm_star_from_smiles(smiles: str) -> float:
    """Calculate CM* from a SMILES string. Returns NaN on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        return calculate_cm_star(mol)
    except Exception:
        return np.nan
```

### 4. nSPS Implementation (Waldmann, 2023)

The **Spacial Score** (Waldmann's exact spelling) assigns a score to each heavy atom based on four per-atom factors:

$$\text{atom\_score}(i) = \text{hybridization} \times \text{stereo} \times \text{ring} \times \text{branching}$$

SPS is the sum over all heavy atoms; **nSPS** normalizes by dividing by the number of heavy atoms.

Available natively in RDKit since 2023.09 via `rdkit.Chem.SpacialScore`.


```python
def nsps_from_smiles(smiles: str) -> float:
    """
    Calculate the normalized Spatial Score (nSPS) using RDKit's built-in module.
    SPS() returns the raw (unnormalized) spatial score; dividing by heavy atom count gives nSPS.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        sps_value = SPS(mol)
        n_heavy = mol.GetNumHeavyAtoms()
        return sps_value / n_heavy if n_heavy > 0 else np.nan
    except Exception:
        return np.nan
```

### 5. Calculate Both Metrics


```python
data['CM*']  = [cm_star_from_smiles(smi)  for smi in data['SMILES']]
data['nSPS'] = [nsps_from_smiles(smi)     for smi in data['SMILES']]

data[['Name', 'CM*', 'nSPS']].round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>CM*</th>
      <th>nSPS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RPT193</td>
      <td>9.960</td>
      <td>0.771</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GDC-0853</td>
      <td>10.323</td>
      <td>0.422</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Orfoglipron</td>
      <td>10.702</td>
      <td>0.333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GS-6207</td>
      <td>10.436</td>
      <td>0.276</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PF-07258669</td>
      <td>9.577</td>
      <td>0.590</td>
    </tr>
    <tr>
      <th>5</th>
      <td>14</td>
      <td>10.059</td>
      <td>0.395</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PC0371</td>
      <td>9.928</td>
      <td>0.457</td>
    </tr>
  </tbody>
</table>
</div>



### 6. Scatter Plot: CM* vs nSPS

Both metrics should broadly agree — compounds with more intricate ring systems and sp3-rich scaffolds should score higher on both axes.

Divergences are informative:
- High CM\* / lower nSPS → chemically diverse environments, but relatively less ring/stereo architecture
- High nSPS / lower CM\* → spatially intricate (bridged rings, stereocentres) but an atom-type palette that is less varied


```python
fig, ax = plt.subplots(figsize=(7, 5))

for _, row in data.iterrows():
    ax.scatter(row['CM*'], row['nSPS'], s=60, zorder=3)
    ax.annotate(row['Name'], (row['CM*'], row['nSPS']),
                textcoords='offset points', xytext=(6, 3), fontsize=8)

ax.set_xlabel('CM*  (Proudfoot, 2017)  →  More complex environments')
ax.set_ylabel('nSPS  (Waldmann, 2023)  →  More spatially complex')
ax.set_title('CM* vs nSPS for a Set of Recent Drug Candidates', loc='left', pad=10)
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
```


    
![png](output_12_0.png)
    


### 7. Molecular Grid with Both Scores Annotated


```python
mols = [Chem.MolFromSmiles(smi) for smi in data['SMILES']]

legends = [f"{row['Name']}\nCM*: {row['CM*']:.2f}  |  nSPS: {row['nSPS']:.2f}"
           for _, row in data.iterrows()]

img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(450, 350), legends=legends)
img
```




    
![png](output_14_0.png)
    



### 8. Atom-Level Complexity Heatmap (CM* per atom)

For a single molecule, we can visualise which atoms contribute most to the CM\* score — a direct callback to the original post.


```python
def atom_complexity_heatmap(smiles: str, threshold: float = 0.8):
    mol = Chem.MolFromSmiles(smiles)
    atoms_list = [(atom.GetIdx(), get_atom_type(atom)) for atom in mol.GetAtoms()]
    neighbors = {
        atom: [atoms_list[neighbor.GetIdx()]
               for neighbor in mol.GetAtomWithIdx(atom[0]).GetNeighbors()]
        for atom in atoms_list
    }
    atom_paths = _collect_atom_paths(neighbors)
    cas = np.zeros(len(atom_paths))
    for i, paths in enumerate(atom_paths):
        pi = fractional_occurrence(paths)
        cas[i] = -np.sum(pi * np.log2(pi)) + np.log2(len(paths))
    cas_norm = (cas - np.min(cas)) / (np.max(cas) - np.min(cas))
    cas_processed = [v if v >= threshold else 0 for v in cas_norm]
    drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
    SimilarityMaps.GetSimilarityMapFromWeights(mol, cas_processed, draw2d=drawer, alpha=0.2)
    drawer.FinishDrawing()
    bio = io.BytesIO(drawer.GetDrawingText())
    img = mpimg.imread(bio)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(img)
    ax.axis('off')
    # Remove all padding/margins so the Cairo image fills the figure exactly
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig
```


```python
# Heatmap for each compound in the set
for _, row in data.iterrows():
    print(f"--- {row['Name']} ---")
    fig = atom_complexity_heatmap(row['SMILES'])
    plt.show()
```

    --- RPT193 ---
    


    
![png](output_17_1.png)
    


    --- GDC-0853 ---
    


    
![png](output_17_3.png)
    


    --- Orfoglipron ---
    


    
![png](output_17_5.png)
    


    --- GS-6207 ---
    


    
![png](output_17_7.png)
    


    --- PF-07258669 ---
    


    
![png](output_17_9.png)
    


    --- 14 ---
    


    
![png](output_17_11.png)
    


    --- PC0371 ---
    


    
![png](output_17_13.png)
    


### 9. Assembly Theory — Conceptual Context

Assembly Theory (Marshall, Cronin et al., *Nature Communications*, 2021) defines the **Molecular Assembly Index (MA)** as the minimum number of bond-forming steps required to construct a molecule, given that previously built fragments can be reused.

Key properties:
- **Low MA**: simple or highly symmetric molecules (methane ≈ 1, benzene is low due to repeating units)
- **High MA**: complex biological molecules — amino acids, nucleotides, secondary metabolites — consistently exceed a threshold of ~15 that simple abiotic chemistry rarely crosses
- This threshold has been proposed as a potential **biosignature** for detecting life

Unlike CM\* and nSPS (which measure the current structure), MA is an *information-theoretic* measure of construction history.

**To compute MA in Python:**
```bash
pip install assembly-theory
```

The `assembly-theory` package wraps a high-performance Rust backend and is RDKit-compatible. The original paper used a Go implementation (`croningp/assembly_go`), but the pip package is the most accessible entry point.

> **Note:** Exact MA calculation for large drug-like molecules is computationally expensive — the assembly space grows rapidly. The pip package uses efficient heuristics. Be mindful of this when working with larger structures.

### 10. Summary Table

| Metric | Origin | What it measures | Normalized? | Code availability |
|---|---|---|---|---|
| CM\* | Proudfoot (2017) | Diversity & depth of local atomic environments | No | Pure Python / RDKit |
| nSPS | Waldmann (2023) | Spatial & topological character (hybridization, rings, stereo, branching) | Yes (by heavy atom count) | `rdkit.Chem.SpacialScore` |
| MA | Cronin (2021) | Minimum construction steps from atomic building blocks | Context-dependent | `pip install assembly-theory` |


All three are complementary rather than competing. Running all three on a candidate structure gives a richer picture than any single number can.

---

**Full blog post:** https://chemicalg.github.io/2026/04/01/nSPS-AssemblyTheory.html  
**Code:** https://github.com/chemicalg/chemicalg_blog/tree/main/2026-04-01-nSPS-AssemblyTheory


```python

```
