import torch
from torch.utils.data import DataLoader

class BatchLoader:
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Simple class that allows you to get one batch at a time using a DataLoader
    - If a call to next_batch finishes an epoch, sets just_finished_epoch to True, otherwise sets it to False
    - shuffle will shuffle the dataset, drop_last will drop the last batch if it's less than batch_size and num_workers is for multiprocessing
    - Keep num_workers=0 for large datasets like coco otherwise your memory is going to explode
    - Resizing the dataset does not work if num_workers > 0. It will only resize once an epoch.
    ----------
    """

    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False, num_workers=0):
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
        self.loader_iter = iter(self.loader)

        self.just_finished_epoch = False

    def next_batch(self):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Grabs the next batch from the dataset
        - If a call finishes an epoch, sets just_finished_epoch to True, otherwise sets it to False
        ----------
        """

        try:
            self.just_finished_epoch = False
            batch = next(self.loader_iter)
        except StopIteration:
            self.just_finished_epoch = True
            self.loader_iter = iter(self.loader)
            batch = next(self.loader_iter)

        return batch
