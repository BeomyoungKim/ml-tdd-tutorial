import os

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.models.network import Net
from src.data.preprocess import testset
from src.constants import MODEL_DIR_PATH, CLASSES


class Evaluator:
    def __init__(self):
        self.load(os.path.join(MODEL_DIR_PATH, 'model.pt'))

        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=4, shuffle=False
        )
        self.writer = SummaryWriter('runs/evaluate')

    def load(self, checkpoint_path):
        self.net = Net()
        self.net.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def evaluate(self):
        class_probs = []
        class_label = []
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                output = self.net(images)
                class_probs_batch = [F.softmax(el, dim=0) for el in output]

                class_probs.append(class_probs_batch)
                class_label.append(labels)

        test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        test_label = torch.cat(class_label)

        return test_label, test_probs

    def add_pr_curve_tensorboard(self, global_step=0):
        test_label, test_probs = self.evaluate()

        for class_index in range(len(CLASSES)):
            tensorboard_truth = test_label == class_index
            tensorboard_probs = test_probs[:, class_index]

            self.writer.add_pr_curve(
                CLASSES[class_index],
                tensorboard_truth,
                tensorboard_probs,
                global_step=global_step
            )
            self.writer.close()
