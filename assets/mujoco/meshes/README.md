Robot mesh groups live here:

- `quadruped/`: checked-in STL meshes referenced by `../quadruped.xml`.
- `battlebot/`: STEP-generated STL meshes referenced by `../battlebot.xml`.

Shared scene pieces such as the ground grid stay in `assets/mujoco/` so robot
scenes can include them without duplicating geometry.
