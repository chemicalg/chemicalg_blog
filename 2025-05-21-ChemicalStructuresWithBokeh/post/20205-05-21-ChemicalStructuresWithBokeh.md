### Chemical Structures with Bokeh

Bokeh is a well documented and versatile visualisation package that has a python API. Check out their documentation [here](https://docs.bokeh.org/en/latest/).

I have been looking for a way to be able to include chemical structures as a hover effect on plots, diagramms, figures interactively and had experimented with a few packages before settling on Bokeh.

This post demonstrates a proof of concept for working with Bokeh to achieve that. I hope you enjoy it and please let me know if you know of different ways of achieving the same with different interactive visualisation packages in python.

First let's define what we will be visualising. I think something simple and amenamble to change will serve us just fine.

As an example, let's look at the chemical synthesis of [paracetamol](https://en.wikipedia.org/wiki/Paracetamol#Chemistry) and how the molecular weight changes at each step of the route.


```python
#import modules
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import rdkit.Chem.Descriptors as Desc

import io
import base64

from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
from bokeh.models import LinearAxis, Range1d
```


```python
#read in the route data

route_data = pd.read_csv('Paracetamol.csv')

route_data
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
      <th>Step</th>
      <th>SMILES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>C1=CC=CC(=C1)OC</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>C1=C(C=CC(=C1)O[H])[N+](=O)[O-]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>C1=C(C=CC(=C1)O[H])N([H])[H]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>C1=C(C=CC(=C1)O[H])N([H])C(C)=O</td>
    </tr>
  </tbody>
</table>
</div>



This is a short 3 step route involving a nitration, reduction to the amine, and acylation to prepare paracetamol starting from phenol.

Let's calculate the molecular weight of each compound in a new column


```python
mols = [Chem.MolFromSmiles(smi) for smi in route_data['SMILES']]

mws = [Desc.ExactMolWt(mol) for mol in mols]

route_data['MW'] = mws

route_data
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
      <th>Step</th>
      <th>SMILES</th>
      <th>MW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>C1=CC=CC(=C1)OC</td>
      <td>108.057515</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>C1=C(C=CC(=C1)O[H])[N+](=O)[O-]</td>
      <td>139.026943</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>C1=C(C=CC(=C1)O[H])N([H])[H]</td>
      <td>109.052764</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>C1=C(C=CC(=C1)O[H])N([H])C(C)=O</td>
      <td>151.063329</td>
    </tr>
  </tbody>
</table>
</div>



Great! Now we can visualise the above in a Bokeh plot:

1. let's get the image for each molecule from its mol object that we generated above


```python
imgs = [Draw.MolToImage(mol) for mol in mols]
```

2. get a mock urls in a list that we can feed to the Bokeh API:


```python
urls = []

for img in imgs:
    buffer = io.BytesIO() #initialise the buffer
    img.save(buffer, format='PNG') #use the buffer to save the image to memory
    byte_im = buffer.getvalue() #retrieve the image from memory
    url = 'data:image/png:base64' #base string
    url += base64.b64encode(byte_im).decode('utf-8') #add unique string after encoding and decoding
    urls.append(url) #append to the list
```


```python
source = ColumnDataSource(data=dict(x=route_data['Step'], y=route_data['MW'], imgs=urls))

TOOLTIPS = """
    <div>
        <div>
            <img
                src="@imgs" height="150"
                style="float: left; margin: 2px 2px 2px 2px;"
                border="2"
            ></img>
        </div>
            <span style="font-size: 15px; font-weight: bold; ">Step: </span>
            <span style="font-size: 15px; ">@x </span>
        </div>
        <div>
            <span style="font-size: 15px; font-weight: bold; ">MW: </span>
            <span style="font-size: 15px; ">@y </span>
        </div>
    </div>
"""

#create the figure
p = figure(plot_width=800, plot_height=800, 
           x_range=(-0.2, route_data['Step'].max()+0.2), 
           y_range=(route_data['MW'].min()-10, route_data['MW'].max()+10), 
           tools='hover, pan, wheel_zoom, box_zoom, reset, save',
           tooltips = TOOLTIPS,
           title = f'Route Intermediate Visualisation') # create the figure

p.circle('x', 'y', fill_alpha=0.8, size=10, source = source) #scatter plot with circles

p.xaxis.axis_label = 'Step number'
p.yaxis.axis_label = 'MW for each compound at each step'

output_notebook()

show(p)   
```


<div class="bk-root">
        <a href="https://bokeh.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="1545">Loading BokehJS ...</span>
    </div>







<div class="bk-root" id="a966571e-49aa-43b2-9ec1-b8143f5a6376" data-root-id="1506"></div>





I find it very useful to be able to hover over a point and the chemical structure to be coming up. This allows for direct interaction and can facilitate faster understanding of your data.

I hope that you find this useful.
