PTLearn - High-level API for PyTorch
====================================

PTLearn is a deep learning library built on top of PyTorch_. Inspired by
TFLearn_ for TensorFlow_, PTLearn provides a high-level API to PyTorch for
quick and easy experimentation.


Features
--------

* High-level API allows you to build deep neural networks with less friction.
* Automatic device placement - use GPU to accelerate tensor computations
  if CUDA is available.
* Seamless interoperability with PyTorch.


Example
-------

.. code-block:: python

    class Net(nn.Module):
        def __init__(self):
            self.fc1 = nn.Linear(784, 512)
            self.fc2 = nn.Linear(512, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = F.dropout(x, 0.5)
            x = self.fc2(x)
            return F.softmax(x)

    net = Net()

    model = ptlearn.DNN(net, loss_fn='CrossEntropy', optimizer='Adam')
    model.fit(X, Y)

    labels = model.predict(data)


.. toctree::
    :hidden:
    :maxdepth: 2

    tutorials


.. _PyTorch: http://pytorch.org/
.. _TFLearn: http://tflearn.org/
.. _TensorFlow: https://www.tensorflow.org/
