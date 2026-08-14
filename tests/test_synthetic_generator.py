import torch

from datasets.synthetic.collision import (
    generate_collision,
)


def test_collision_generation():

    sample = generate_collision()


    assert sample["positions"].shape == (
        20,
        2
    )


    assert sample["velocities"].shape == (
        20,
        2
    )


    assert sample["interaction"] == (
        "elastic_collision"
    )


    assert torch.isclose(
        sample["momentum_before"],
        sample["momentum_after"]
    )