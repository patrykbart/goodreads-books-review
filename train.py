import torch
import pandas as pd
import pytorch_lightning as pl

from src.model import ReviewModel
from src.data_provider import GoodreadsDataset
from src.utils import get_config_yaml, visible_print

if __name__ == "__main__":
    config = get_config_yaml()

    visible_print("Loading data")
    train_dataset = GoodreadsDataset(config["data"]["train_path"], config)
    valid_dataset = GoodreadsDataset(config["data"]["valid_path"], config) if config["training"]["validate"] else None
    test_dataset = GoodreadsDataset(config["data"]["test_path"], config)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["data"]["batch_size"], shuffle=True, num_workers=config["data"]["num_workers"]
    )
    valid_loader = (
        torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
        )
        if valid_dataset is not None
        else None
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"]
    )

    visible_print("Trainer setup")
    callbacks = []
    if config["training"]["early_stopping"]:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="val_loss" if config["training"]["validate"] else "loss",
                patience=config["training"]["patience"],
                verbose=True,
                mode="min",
            )
        )

    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        val_check_interval=config["training"]["val_check_interval"],
        num_nodes=1,
        accelerator="cpu",
        enable_model_summary=True,
        logger=False,
        callbacks=callbacks,
    )

    visible_print("Model summary")
    model = ReviewModel(config)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    trainer.test(model=model, dataloaders=test_loader)

    visible_print("Show results")
    print(f"First 10 test outputs: {model.test_results[:10]}")
    print(f"Average test output: {sum(model.test_results) / len(model.test_results)}")
    print(f"Minimum test output: {min(model.test_results)}")
    print(f"Maximum test output: {max(model.test_results)}")

    if not config["data"]["subsample"]:
        visible_print("Save as CSV")
        df = pd.read_csv(config["data"]["test_path"])
        review_id = df["review_id"].tolist()

        df = pd.DataFrame({"review_id": review_id, "rating": model.test_results})
        df.to_csv(config["data"]["output_path"], index=False)
