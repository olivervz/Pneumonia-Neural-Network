# Keras Pneumonia X-ray neural network

#### This was my final project for CS 4100, Artificial Intelligence.\

#### See **_Report.pdf_** for a more detailed project description as well as final outputs.\

## Usage

### Required Packages:

#### NOTE: this has only been tested in a linux environment

This project is based on python 3.7.2 and requires several additional packages to be installed via pip. This can be accomplished through the steps below.

<details><summary><b>Show instructions</b></summary>
    <br />
    Creating a virtual environment is recommended if you do not wish to install
    various packages required to run this project into your python environment.
    To create a temporary environment local to this project:
    python3 -m venv env

1. (optional) Create a local virtual environment:

   ```
   python3 -m venv env
   source env/bin/activate
   ```

2. Install the packages listed in requirements.txt

   ```
   pip install -r requirements.txt
   ```

3. To exit the virtual environment:
   `deactivate`
   </details>

### Running the Project:

To run this project, you first must load the data from the raw images into a numpy array usable by the neural network. This can be done with the following command:

```
make load
```

Once the data is loaded, the model can be trained using this data with the following command:

```
make train
```

To run the inference stage:

```
make run
```

To remove the model and weights:

```
make clean
```

To remove the locally made Virtual Environment:

```
make removeVENV
```

##### Note: This README, was taken from a previous personal project I completed
