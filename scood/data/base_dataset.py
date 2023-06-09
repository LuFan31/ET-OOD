import logging
import random
import traceback

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, psudo_index=-1, skip_broken=False, new_index="next"):
        super(BaseDataset, self).__init__()
        self.psudo_index = -1
        self.skip_broken = skip_broken
        self.new_index = new_index
        if new_index not in ("next", "rand"):
            raise ValueError('new_index not one of ("next", "rand")')

    def __getitem__(self, index):  
        # in some pytorch versions, input index will be torch.Tensor
        index = int(index)

        # if sampler produce psudo_index, randomly sample an index, and mark it as psudo
        if index == self.psudo_index:
            index = random.randrange(len(self)) #def randrange(self, start, stop=None, step=1, _int=int):
            psudo = 1  #mark it as psudo
        else:
            psudo = 0

        while True:
            try:
                sample = self.getitem(index)
                break
            except Exception as e:
                if self.skip_broken and not isinstance(e, NotImplementedError):
                    if self.new_index == "next":
                        new_index = (index + 1) % len(self)
                    else:
                        new_index = random.randrange(len(self))
                    logging.warn(
                        "skip broken index [{}], use next index [{}]".format(
                            index, new_index
                        )
                    )
                    index = new_index
                else:
                    logging.error("index [{}] broken".format(index))
                    traceback.print_exc()
                    logging.error(e)
                    raise e

        sample["index"] = index
        sample["psudo"] = psudo  
        return sample

    def getitem(self, index):
        raise NotImplementedError
