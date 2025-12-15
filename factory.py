"""Legacy NeuronFactory placeholder.

This module intentionally raises an ImportError to signal that the factory
pattern has been removed in favor of directly instantiating ``models.Neuron``.
"""

raise ImportError(
    "`factory.NeuronFactory` was removed. Instantiate `models.Neuron` "
    "directly with the desired activation instead."
)
