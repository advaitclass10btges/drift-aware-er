import torch


def generate_collision(
    steps: int = 20,
    dt: float = 0.1,
):
    """
    Simulate a 1D elastic collision.

    Two objects:
    Object A:
        moves right

    Object B:
        stationary

    Outputs:
        positions
        velocities
        collision event
        momentum before/after
    """

    mass_a = 1.0
    mass_b = 1.0

    velocity_a = 2.0
    velocity_b = 0.0

    collision_step = steps // 2

    positions = []
    velocities = []

    momentum_before = None
    momentum_after = None


    pos_a = 0.0
    pos_b = 2.0


    for step in range(steps):

        if step == collision_step:

            momentum_before = torch.tensor(
                mass_a * velocity_a
                +
                mass_b * velocity_b,
                dtype=torch.float32
            )

            # Equal mass elastic collision
            velocity_a, velocity_b = (
                velocity_b,
                velocity_a
            )

            momentum_after = torch.tensor(
                mass_a * velocity_a
                +
                mass_b * velocity_b,
                dtype=torch.float32
            )


        pos_a += velocity_a * dt
        pos_b += velocity_b * dt


        positions.append(
            torch.tensor(
                [
                    pos_a,
                    pos_b
                ],
                dtype=torch.float32
            )
        )


        velocities.append(
            torch.tensor(
                [
                    velocity_a,
                    velocity_b
                ],
                dtype=torch.float32
            )
        )


    return {
        "positions": torch.stack(
            positions
        ),

        "velocities": torch.stack(
            velocities
        ),

        "collision_step": collision_step,

        "momentum_before":
            momentum_before,

        "momentum_after":
            momentum_after,

        "interaction":
            "elastic_collision",
    }