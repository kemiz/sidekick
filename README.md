# sidekick [![Build Status](https://travis-ci.com/Peltarion/sidekick.svg?token=nkS94uQqBVFyK1JitpGf&branch=master)](https://travis-ci.com/Peltarion/sidekick)
This code is the sidekick to the superhero that is the Peltarion Platform,
handling the mundane tasks like bundling up data into the Platform's preferred
format or sending data examples to the Deployment endpoints to get predictions.

The aim is to make it easier to get started with getting data in and getting
predictions out, which are parts of the workflow that are currently left for
the user to figure out.

This should help more people experience the end-to-end flow of a deep learning
project and appreciate the value that the Platform provides.


## Installation
Sidekick requires python 3.5+. Install package and dependencies with pip
directly from GitHub (requires valid SSH keys to the Peltarion organization).
When doing this we recommend using a separate virtual environment, see e.g.
[this tutorial](https://realpython.com/python-virtual-environments-a-primer/)).

### Integrating Jupiter notebooks with virtual envs
If you're using a virtual env, then you need to make it available to jupiter through this
```
ipython kernel install --user --name=sidekick
```
where sidekick is the name of your virtual env.

```console
> pip install git+ssh://git@github.com/Peltarion/sidekick.git#egg=sidekick
```

Another option is to create a GitHub access token here https://github.com/settings/tokens with repo permissions, and using this command instead
```console
pip install git+https://USERNAME:TOKEN@github.com/Peltarion/sidekick.git#egg=sidekick
```
The token can be revoked from the same GitHub page, and this doesn't require copying your SSH keys around.

## Create a Platform compatible dataset
When creating a dataset zip you may load data the data in two separate ways.
Both require loading the data in a Pandas `DataFrame` and assume all columns
only contain one type of data with the same shape.

#### 1) In memory objects
Store objects directly in the `Series` (columns of your `DataFrame`). This
works for all scalars (floats, integers and strings of one dimension) as well
as [Pillow](https://pillow.readthedocs.io/en/stable/) images and numpy arrays.
This is such an example with a progressbar enabled:

```python
df.head()
```
```
float_column          image_column                numpy_column
    0.248851  <PIL.Image.Image ...  [0.18680, 0.61951, 0.83...
    0.523621  <PIL.Image.Image ...  [0.75213, 0.44948, 0.82...
    0.647844  <PIL.Image.Image ...  [0.41525, 0.63858, 0.34...
    0.447717  <PIL.Image.Image ...  [0.79373, 0.24514, 0.94...
    0.194222  <PIL.Image.Image ...  [0.12636, 0.40554, 0.66...
```

```python
import sidekick

# Create dataset
sidekick.create_dataset(
    'path/to/dataset.zip',
    df,
    progress=True
)
```

#### 2) Paths to objects

Columns may also point to paths of object. Which columns are paths should be
indicated in the `path_columns`. Like the in-memory version these may also be
preprocessed. This is an example where all images are loaded from a path,
preprocessed to have the same shape and type and then placed in the dataset:

```python
df.head()
```
```text
float_column string_column                                  image_file_column
    0.248851           foo  /var/folders/7t/80jfy0rd3l7f31xdd3rw0_jw0000gn...
    0.523621           foo  /var/folders/7t/80jfy0rd3l7f31xdd3rw0_jw0000gn...
    0.647844           foo  /var/folders/7t/80jfy0rd3l7f31xdd3rw0_jw0000gn...
    0.447717           foo  /var/folders/7t/80jfy0rd3l7f31xdd3rw0_jw0000gn...
    0.194222           foo  /var/folders/7t/80jfy0rd3l7f31xdd3rw0_jw0000gn...
```

```python
import functools
import sidekick

# Create preprocessor for images, cropping to 32x32 and formatting as png
image_processor = functools.partial(
    sidekick.process_image, crop_size=(32, 32), format='png')
 
# Create dataset
sidekick.create_dataset(
    'path/to/dataset.zip',
    df,
    path_columns=['image_file_column'],
    preprocess={
        'image_file_column': image_processor
    }
)
```


## Using deployments
To connect to an enabled deployment use the `sidekick.Deployment` class. This
takes the information you find on the deployment page of an experiment. Given
the following deployment here's how to query it.

![deployment example](static/image/deployment_example.png "Deployment example")

Use the `url` and `token` displayed in the dark box. Then, create a dictionary
of the `feature` and `type` fields from the table of input and output
parameters to specify `dtypes_in` and `dtypes_out`.

```python
import sidekick

client = sidekick.Deployment(
    url='<url>',
    token='<token>',
    dtypes_in={'image': 'Image (32x32x3)'},
    dtypes_out={'image': 'Image (32x32x3)'}
)
```

This deployment client may now be used to get predictions for images.

```python
from PIL import Image

# Load image
image = Image.open('test.png')

# Get predictions from model
client.predict(image=image)
```

To efficiently predict the results of multiple input examples use
`predict_many`:

```python
client.predict_many([
    {'image': Image.open('test1.png')},
    {'image': Image.open('test2.png')}
])
```

For interactive exploration of data it us useful to use the `predict_lazy`
method, which returns a generator that lazily polls the deployment when
needed. Here's an example exploring outputs of an autoencoder interactively:

```python
import IPython.display as ipd
import numpy as np
import pandas as pd
import sidekick
import zipfile
from ipywidgets import interact_manual
from PIL import Image


# Open Peltarion compatible cifar10 dataset
DATASET_PATH = 'cifar10_full.zip'
with zipfile.ZipFile(DATASET_PATH, 'r') as z:
    index_file = z.open('index.csv')
    dataset = pd.read_csv(index_file)
    dataset['image'] = dataset['image'].apply(
        lambda path: Image.open(z.open(path))
    )

# Connect to autoencoder deployment
client = sidekick.Deployment(
    url='<url>',
    token='<token>',
    dtypes_in={'image': 'Image (32x32x3)'},
    dtypes_out={'image': 'Image (32x32x3)'}
)

# Get predictions for all images in dataset
predictions = client.predict_lazy(dataset.to_dict('records'))

# Show grid of given size
def show_next_grid(n_rows, n_columns):
    grid = Image.new('RGB', (n_columns * 32, n_rows * 32))
    for column in range(n_columns):
        for row in range(n_rows):
            grid.paste(next(predictions)['image'], (column * 32, row * 32))
    return grid

interact_manual(show_next_grid, n_rows=(1, 10), n_columns=(1, 10))
```

Here's an example of it running. Notice that this allows you to immediatly
start exploring the results instead of waiting for all predictions to finnish.

![interactive example](static/image/interactive_use_sidekick.gif)

#### Note
If you're on Ubuntu, remember to run
```
jupyter nbextension enable --py widgetsnbextension
```
to get the ipywidgets working


### Compatibility
Which object types are compatible with each field type may be show by printing
the `sidekick.encode.DTYPE_COMPATIBLITY` dictionary.

```python
print(sidekick.encode.DTYPE_COMPATIBILITY)

{'Float': {float},
 'Int': {int},
 'Numpy': {numpy.ndarray},
 'Image': {PIL.Image.Image}}
```
