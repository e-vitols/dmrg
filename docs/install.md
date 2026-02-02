# Installation instructions

We highly recommend that you install vlx-dmrg in a conda environment.
If you don't already have conda, install [Miniconda](https://www.anaconda.com/download/success) 
following the [instructions](https://www.anaconda.com/docs/getting-started/miniconda/install) for your operating system. 
Once you have miniconda, create a new conda environment using the vlxdmrg_env.yml file.
In a terminal (or Anaconda Powershell prompt for Windows), run:

```
conda env create -f vlxdmrg_env.yml
```

Activate the conda environment:

```
conda activate vlxdmrgenv
```

Clone the repository with:

```
git clone https://github.com/e-vitols/dmrg.git
```

or

```
git clone git@github.com:e-vitols/dmrg.git
```

Pip install the package:

```
cd dmrg
python3 -m pip install .
```

If on Linux you get an error with building wheel, and a final error message like:

```
error: command 'g++' failed: Not a directory
```

Then try:

```
conda install gcc gxx_linux-64
```

and again try:

```
python3 -m pip install .
```

The package should now be importable with:

```
import dmrg
```

## Testing

After successful installation it is recommended you run the tests:

```
python3 -m pytest tests -m "not slow" 
```

The above runs only the fast tests, if you wish to run only the slower ones then do:

```
python3 -m pytest tests -m "slow" 
```


