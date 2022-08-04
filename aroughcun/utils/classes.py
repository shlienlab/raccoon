"""
Auxiliary classes for RACCOON
F. Comitani     @2022
"""

from math import sqrt

class IdentityProjection:

    """ To be used when the target space dimensionality corresponds to the input space
        and the dimensionality reduction step should be skipped. """

    def __init__(self, **kwargs):
        """ Initialize the the class.
        
        Args:
            kwargs: keyword arguments will be ignored.
        """

        self.n_neighbors  = 0
        self.n_components = 1

    def identity(self, data):
        """ Identity function. Returns the input data.

        Args:
            data (any): object to be returned.

        Returns:    
            data (any): object to be returned.
        """

        return data

    def fit(self, data, *args, **kwargs):
        """ Initialize the the class, set the number of 
            neighbors as square root of the dataset size
            and dimensionality of the dataset.

            args: arguments will be ignored.
            kwargs: keyword arguments will be ignored.
        """

        self.n_neighbors  = int(sqrt(data.shape[0]))
        self.n_components = data.shape[1]

    def transform(self, data):
        """ Empty transform function

        Args:
            data (any): object to be returned.

        Returns:    
            data (any): object to be returned.
        """

        return self.identity(data)

    def fit_transform(self, data, **kwargs):
        """ Empty fit_transform function

        Args:
            data (any): object to be returned.

        Returns:    
            data (any): object to be returned.
            kwargs: keyword arguments will be ignored.
        """

        self.fit(data)
        return self.transform(data)