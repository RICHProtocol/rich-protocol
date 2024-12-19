import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BertAgent(nn.Module):
    def __init__(self, num_labels=2, hidden_width=256, depth=2):
        """
        Initialize the BERT-based model with customizable depth and width.
        :param num_labels: Number of output classes.
        :param hidden_width: Width of the fully connected hidden layers.
        :param depth: Number of additional fully connected layers.
        """
        super(BertAgent, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.depth = depth
        self.hidden_width = hidden_width

        # Dynamically create linear layers using _make_layer
        self.fc_layers = self._make_layer(self.bert.config.hidden_size, hidden_width, depth, num_labels)

    def _make_layer(self, input_size, hidden_width, depth, num_labels):
        """
        Create fully connected layers dynamically based on depth and width.
        :param input_size: Size of the input features.
        :param hidden_width: Width of the hidden layers.
        :param depth: Number of hidden layers.
        :param num_labels: Number of output classes.
        """
        layers = []
        in_features = input_size
        for _ in range(depth - 1):
            layers.append(nn.Linear(in_features, hidden_width))
            layers.append(nn.ReLU())
            in_features = hidden_width
        layers.append(nn.Linear(in_features, num_labels))  # Final output layer
        return nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the BERT model.
        :param input_ids: Tokenized input IDs.
        :param attention_mask: Attention masks for input.
        :return: Output logits.
        """
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output  # Get the pooled [CLS] token output
        return self.fc_layers(pooled_output)
