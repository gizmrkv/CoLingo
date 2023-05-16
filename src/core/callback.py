class Callback:
    """
    This class represents a callback interface that can be used to execute certain 
    functions at specific points during an operation, such as the beginning, update, 
    and end of a task.
    """

    def on_begin(self):
        """
        This method is called at the beginning of a task.
        """
        pass

    def on_update(self):
        """
        This method is called during each update of a task.
        """
        pass

    def on_end(self):
        """
        This method is called at the end of a task.
        """
        pass