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
    test_dataset = GoodreadsDataset(config["data"]["test_path"], config)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["data"]["batch_size"], shuffle=True, num_workers=config["data"]["num_workers"]
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"]
    )

    visible_print("Trainer setup")
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        num_nodes=1,
        accelerator="cpu",
        enable_model_summary=True,
        logger=False,
    )

    visible_print("Model summary")
    model = ReviewModel(config)
    trainer.fit(model=model, train_dataloaders=train_loader)

    trainer.test(model=model, dataloaders=test_loader)

    visible_print("Done, saving results")
    print(f"Average test output: {sum(model.test_results) / len(model.test_results)}")
    print(f"Minimum test output: {min(model.test_results)}")
    print(f"Maximum test output: {max(model.test_results)}")

    # Commit results to the dataset and save as CSV
    df = pd.read_csv(config["data"]["test_path"])
    review_id = df["review_id"].tolist()

    df = pd.DataFrame({"review_id": review_id, "rating": model.test_results})
    df.to_csv(config["data"]["output_path"], index=False)
