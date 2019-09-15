import torch
import torch.nn as nn
import torch.nn.functional as F


def entangled_loss(targets, receiver_output_1, receiver_output_2):
    acc_1 = (receiver_output_1.argmax(dim=1) == targets[:, 0]).detach().float()
    acc_2 = (receiver_output_2.argmax(dim=1) == targets[:, 1]).detach().float()
    loss_1 = F.cross_entropy(receiver_output_1, targets[:, 0], reduction="none")
    loss_2 = F.cross_entropy(receiver_output_2, targets[:, 1], reduction="none")
    acc = (acc_1 * acc_2).mean(dim=0)
    loss = loss_1 + loss_2
    return loss, {f'accuracy': acc.item(),
                  f'first_accuracy': acc_1.mean(dim=0).item(),
                  f'second_accuracy': acc_2.mean(dim=0).item()}


def disentangled_loss(target, output, prefix):
    acc = (output.argmax(dim=1) == target).detach().float().mean(dim=0)
    loss = F.cross_entropy(output, target, reduction="none")
    return loss, {f'{prefix}_accuracy': acc.item()}


def sample(message_size):
    batch_size, _, vocab_size = message_size
    random_symbols = torch.randint(0, vocab_size, size=(batch_size,))
    return F.one_hot(random_symbols, num_classes=vocab_size).float().unsqueeze(dim=1)


class PretrainingmGameGS(nn.Module):
    def __init__(
            self,
            senders,
            receiver,
            padding=True
    ):
        super(PretrainingmGameGS, self).__init__()
        self.sender_1, self.sender_2 = senders
        self.receiver = receiver
        self.padding = padding

    def forward(self, sender_input, target):
        message_1 = self.sender_1(sender_input)
        if self.padding:
            message_2 = sample(message_1.size())
            message = torch.cat([message_1, message_2], dim=1)
        else:
            message = message_1
        first_receiver_output, second_receiver_output = self.receiver(message)
        loss_1, rest_info_1 = disentangled_loss(target[:, 0], first_receiver_output[:, -1, ...], prefix='first')

        message_2 = self.sender_2(sender_input)
        if self.padding:
            message_1 = sample(message_2.size())
            message = torch.cat([message_1, message_2], dim=1)
        else:
            message = message_2
        first_receiver_output, second_receiver_output = self.receiver(message)
        loss_2, rest_info_2 = disentangled_loss(target[:, 1], second_receiver_output[:, -1, ...], prefix='second')
        rest = {
            'first_accuracy': rest_info_1['first_accuracy'],
            'second_accuracy': rest_info_2['second_accuracy'],
            'accuracy': (rest_info_1['first_accuracy'] + rest_info_2['second_accuracy'])/2
        }
        return (loss_1 + loss_2).mean(), rest


class CompositionalGameGS(nn.Module):
    def __init__(
            self,
            sender,
            receiver,
    ):
        super(CompositionalGameGS, self).__init__()
        self.sender = sender
        self.receiver = receiver

    def forward(self, sender_input, target):
        message = self.sender(sender_input)
        first_receiver_output, second_receiver_output = self.receiver(message)
        loss, rest_info = entangled_loss(target, first_receiver_output[:, -1, ...], second_receiver_output[:, -1, ...])
        return loss.mean(), rest_info
