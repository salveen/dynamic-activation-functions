"""Legacy activation strategy placeholder.

Activation strategies are now baked directly into ``models.Neuron``. Importing
this module raises an informative error to help migrate existing scripts.
"""

raise ImportError(
    "`activation_functions` has been removed. Use the built-in activations "
    "handled by `models.Neuron` instead."
)
