### SVHN → MNIST

This is a very common adaptation benchmark. Following the standard protocol, we used the whole training sets of both datasets,
resized MNIST images to 32 × 32 and converted SVHN to grayscale.

### The code
* The script ``download.sh`` collects both MNIST and SVHN. Default downloaing directory is ``./``.
  ```
  sudo chmod a+x download.sh
  ./download.sh
  ``` 

* ``prepro.py`` saves a rescaled copy of MNIST into ``train.pkl`` and ``test.pkl``.
    ```
    python prepro.py
    ``` 

* ``model.py`` deploys the net and losses definitions and the training operations.

* ``solver.py`` defines the ``Solver`` class which does the actual training and testing

* ``main.py`` creates the net and launches the training/testing/plotting, acccording to the ``--mode`` options. 

    For example:
    ```
    python main.py --mode='train' --method='log-d-coral' --alpha=1. --device='/gpu:0'
    python main.py --mode='test' --method='log-d-coral' --alpha=1. --device='/gpu:0'
    python main.py --mode='tsne' --method='log-d-coral' --alpha=1. --device='/gpu:0'
    ```

Below are the t-SNE visualization of the feature space learned by geodesic aligment. *Left*: blue and red dots indicate SVHN (source) and MNIST (target) features,respectively. *Right*: different colors indicate the ten different classes
![tsne](./svhn2mnist/tsne.png)
