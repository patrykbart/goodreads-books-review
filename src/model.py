import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, Accuracy

from src.utils import visible_print


class ReviewModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.mode = config["model"]["mode"]

        self.num_directions = 2 if config["model"]["bidirectional"] else 1
        self.hidden_size = config["model"]["hidden_size"]
        self.padding_idx = config["model"]["padding_idx"]

        self.embedding = nn.Embedding(
            num_embeddings=config["model"]["vocab_size"],
            embedding_dim=config["model"]["embedding_size"],
            padding_idx=self.padding_idx,
        )
        self.lstm = nn.LSTM(
            input_size=config["model"]["embedding_size"],
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            bidirectional=config["model"]["bidirectional"],
            batch_first=True,
        )
        self.fc1 = nn.Linear(
            in_features=self.num_directions * self.hidden_size,
            out_features=config["model"]["hidden_size"],
        )
        self.fc2 = nn.Linear(
            in_features=config["model"]["hidden_size"],
            out_features=1 if self.mode == "regression" else config["model"]["output_size"],
        )

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=config["model"]["dropout"])
        self.softmax = nn.Softmax(dim=1)

        if self.mode == "regression":
            self.loss = nn.MSELoss()
            self.metric = MeanSquaredError()
        elif self.mode == "classification":
            self.loss = nn.CrossEntropyLoss()
            self.metric = Accuracy(task="multiclass", num_classes=config["model"]["output_size"])

        self.test_results = []

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )

    def forward(self, x):
        h0, c0 = self.init_hidden(x.shape[0])

        lengths = torch.sum(x != self.padding_idx, dim=1)

        x = self.embedding(x)
        x, _ = self.lstm(x, (h0, c0))

        x = x[torch.arange(x.shape[0]), lengths - 1, :]

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)

        if self.mode == "regression":
            return x.squeeze()
        elif self.mode == "classification":
            return self.softmax(x)

    def training_step(self, batch, batch_idx):
        x, y_true = batch

        y_pred = self(x)
        loss = self.loss(y_pred, y_true)

        self.log("loss", loss, prog_bar=True)
        self.log(self.metric.__class__.__name__, self.metric(y_pred, y_true), prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x = batch

        y_pred = self(x)

        if self.mode == "regression":
            y_pred = torch.round(y_pred).detach().tolist()
            self.test_results.extend(y_pred)
        elif self.mode == "classification":
            y_pred = torch.argmax(y_pred, dim=1).detach().tolist()
            self.test_results.extend(y_pred)

    def on_train_start(self):
        visible_print("Training encoder")

    def on_test_start(self):
        visible_print("Testing encoder")

    def init_hidden(self, batch_size):
        return (
            torch.zeros(
                self.num_directions * self.config["model"]["num_layers"],
                batch_size,
                self.hidden_size,
                device=self.embedding.weight.device,
            ),
            torch.zeros(
                self.num_directions * self.config["model"]["num_layers"],
                batch_size,
                self.hidden_size,
                device=self.embedding.weight.device,
            ),
        )
