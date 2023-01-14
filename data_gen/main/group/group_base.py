import numpy as np

class GroupBase():
    def __init__(self):
        self.order = 1
        raise NotImplementedError("This is abstract class")

    def get_element(self, index) -> np.ndarray:
        """pick index-th element in group
        
        Arguments:
            index {int} -- index of element
        
        Returns:
            np.ndarray -- matrix representation of index-th element
        """
        raise NotImplementedError("This is abstract class")

    def sampling(self, count : int) -> list:
        """randomly choose <code>count</code> of elements
        
        Arguments:
            count {int} -- number of samples
        
        Returns:
            list -- list of chosen elements
        """
        indices = np.random.randint(self.order, size=count, dtype=np.int64)
        elements = [self.get_element(index) for index in indices]
        return elements

    def enumerate_all(self) -> list:
        """enumerate all element
        
        Returns:
            list -- list of all the elements in group
        """
        indices = np.arange(self.order)
        elements = [self.get_element(index) for index in indices]
        return elements
