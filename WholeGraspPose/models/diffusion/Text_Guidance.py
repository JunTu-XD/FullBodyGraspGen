import torch


class Text2Label:
    prompt=["Grasp from top"]
    label = [1]

    def __init__(self):
        self.sbert

    def to_label(self, to_tensor=True):
        # self sbert get embedding
        # ind = arg min dist
        # label = prompt[ind]
        # if one_hot(torch.tensor(label), 2)
        pass
