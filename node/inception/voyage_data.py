from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from inception.dataset import Dataset


class VoyageData(Dataset):
  """Flowers data set."""

  def __init__(self, subset):
    super(VoyageData, self).__init__('Voyage', subset)

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 4

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data subset."""
    if self.subset == 'train':
      return 2362
    if self.subset == 'validation':
      return 1012

  def download_message(self):
    return ""
