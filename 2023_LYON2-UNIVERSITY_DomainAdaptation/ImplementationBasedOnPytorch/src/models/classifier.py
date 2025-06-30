import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc3 = nn.Linear(100, 10)

    def forward(self, input):
        logits = self.fc3(input)
        return logits