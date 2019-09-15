import collections

from neptunecontrib.monitoring.utils import send_figure
import neptune
import seaborn as sns
import matplotlib.pyplot as plt
from egg.core import Callback, EarlyStopperAccuracy
import torch
from tabulate import tabulate

from compositionality.metrics import compute_concept_symbol_matrix, compute_context_independence, compute_representation_similarity


class NeptuneMonitor(Callback):

    def __init__(self, prefix=None):
        self.epoch_counter = 0
        self.prefix = prefix + '_' if prefix else ''

    def on_epoch_end(self, loss, rest):
        self.epoch_counter += 1
        if self.epoch_counter % 10 == 0:
            neptune.send_metric(f'{self.prefix}train_loss', self.epoch_counter, loss)
            for metric, value in rest.items():
                neptune.send_metric(f'{self.prefix}train_{metric}', self.epoch_counter, value)

    def on_test_end(self, loss, rest):
        neptune.send_metric(f'{self.prefix}test_loss', self.epoch_counter, loss)
        for metric, value in rest.items():
            neptune.send_metric(f'{self.prefix}test_{metric}', self.epoch_counter, value)


class CompositionalityMetric(Callback):

    def __init__(self, dataset, sender, opts, vocab_size, prefix=''):
        self.dataset = dataset
        self.sender = sender
        self.epoch_counter = 0
        self.opts = opts
        self.vocab_size = vocab_size
        self.prefix = prefix

        self.epoch_counter = 0

    def on_epoch_end(self, *args):
        self.epoch_counter += 1
        if self.epoch_counter % 2 == 0:
            self.input_to_message = collections.defaultdict(list)
            self.message_to_output = collections.defaultdict(list)
            train_state = self.trainer.game.training  # persist so we restore it back
            self.trainer.game.train(mode=False)
            for _ in range(10):
                self.run_inference()
            self.concept_symbol_matrix, concepts = compute_concept_symbol_matrix(
                self.input_to_message,
                input_dimensions=[self.opts.n_features] * self.opts.n_attributes,
                vocab_size=self.vocab_size
            )
            self.trainer.game.train(mode=train_state)
            self.print_table_input_to_message()
            self.draw_concept_symbol_matrix()

            # Context independence metrics
            context_independence_scores, v_cs = compute_context_independence(
                self.concept_symbol_matrix,
                input_dimensions=[self.opts.n_features] * self.opts.n_attributes,
            )
            neptune.send_metric(self.prefix + 'context independence', self.epoch_counter, context_independence_scores.mean(axis=0))
            neptune.send_text(self.prefix + 'v_cs', str(v_cs.tolist()))
            neptune.send_text(self.prefix + 'context independence scores', str(context_independence_scores.tolist()))

            # RSA
            correlation_coeff, p_value = compute_representation_similarity(
                self.input_to_message,
                input_dimensions=[self.opts.n_features] * self.opts.n_attributes
            )
            neptune.send_metric(self.prefix + 'RSA', self.epoch_counter, correlation_coeff)
            neptune.send_metric(self.prefix + 'RSA_p_value', self.epoch_counter, p_value)

    def on_train_end(self):
        self.on_epoch_end(self)

    def run_inference(self):
        raise NotImplementedError()

    def print_table_input_to_message(self):
        table_data = [['x'] + list(range(self.opts.n_features))] + [[i] + [None] * self.opts.n_features for i in range(self.opts.n_features)]
        for (input1, input2), messages in self.input_to_message.items():
            table_data[input1 + 1][input2 + 1] = '  '.join((' '.join((str(s) for s in message)) for message in set(messages)))
        for a, b in zip(range(self.opts.n_features), range(self.opts.n_features)):
            if a == b:
                table_data[a+1][(b % self.opts.n_features) + 1] = '*' + table_data[a+1][(b % self.opts.n_features) +1]
        filename = f'{self.prefix}input_to_message_{self.epoch_counter}.txt'
        with open(file=filename, mode='w', encoding='utf-8') as file:
            file.write(tabulate(table_data, tablefmt='fancy_grid'))
        neptune.send_artifact(filename)
        with open(file='latex' + filename, mode='w', encoding='utf-8') as file:
            file.write(tabulate(table_data, tablefmt='latex'))
        neptune.send_artifact('latex' + filename)

    def draw_concept_symbol_matrix(self):
        figure, ax = plt.subplots(figsize=(20, 5))
        figure.suptitle(f'Concept-symbol matrix {self.epoch_counter}')
        g = sns.heatmap(self.concept_symbol_matrix, annot=True, fmt='.2f', ax=ax)
        g.set_title(f'Concept-symbol matrix {self.epoch_counter}')
        send_figure(figure, channel_name=self.prefix + 'concept_symbol_matrix')
        plt.close()


class CompositionalityMetricGS(CompositionalityMetric):

    def run_inference(self):
        with torch.no_grad():
            ran_inference_on = collections.defaultdict(int)
            for (input, target) in self.dataset:
                target = tuple(target.tolist())
                if ran_inference_on[target] < 5:
                    message = self.sender(input.unsqueeze(dim=0))[0]
                    message = tuple(message.argmax(dim=1).tolist())
                    neptune.send_text(self.prefix + 'messages', f'{target} -> {message}')
                    self.input_to_message[target].append(message)
                    ran_inference_on[target] += 1


class EarlyStopperAccuracy(EarlyStopperAccuracy):

    def __init__(self, threshold: float, field_name: str = 'acc', delay=5, train: bool = True) -> None:

        super(EarlyStopperAccuracy, self).__init__(threshold, field_name)
        self.delay = delay
        self.train = train

    def should_stop(self) -> bool:
        data = self.train_stats if self.train else self.validation_stats
        if len(data) < self.delay:
            return False
        assert data is not None, 'Validation/Train data must be provided for early stooping to work'
        return all(logs[self.field_name] > self.threshold for _, logs in data[-self.delay:])

    def on_train_end(self):
        if self.should_stop():
            print(f'Stopped early on epoch {self.epoch}')