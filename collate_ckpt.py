from pathlib import Path
from typing import Optional

from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
import typer

# lightning deepspeed has saved a directory instead of a file


def main(checkpoint_dir: Path, output_file: Optional[Path] = None):
    if output_file is None:
        output_file = checkpoint_dir / "collated.pt"
    convert_zero_checkpoint_to_fp32_state_dict(
        checkpoint_dir=checkpoint_dir, output_file=output_file
    )


if __name__ == "__main__":
    typer.run(main)
