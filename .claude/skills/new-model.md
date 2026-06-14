# Skill: new-model

Checklist for adding a new generator or discriminator architecture.

## Steps

1. Ask the user for:
   - **model_name** — snake_case, e.g. `resnet_generator`
   - **type** — `generator` | `discriminator`

2. Create `src/gan_pipeline/models/<model_name>.py`:
   - Inherit from `BaseGenerator` or `BaseDiscriminator` (`models/base.py`)
   - Implement `forward(self, x: torch.Tensor) -> torch.Tensor`
   - Follow weight init convention: `nn.init.normal_(m.weight, 0.0, 0.02)`
   - Use `ReLU(inplace=False)` in any decoder — never inplace (corrupts skip-connection gradients)

3. Export from `src/gan_pipeline/models/__init__.py`:
   ```python
   from .model_name import ClassName
   ```

4. Add `configs/model/<model_name>.yaml` with all constructor hyperparameters.

5. Wire into the relevant training script (`train_pix2pix.py` or `train.py`) under a config branch.

6. Write tests in `tests/test_models.py` or `tests/test_pix2pix.py`:
   - Output shape for expected input shape
   - Forward pass produces finite values (no NaN)
   - Gradient flows back through the model

7. Run the full check:
   ```bash
   make lint && make typecheck && pytest -k <model_name> -v
   ```

8. Run `/smoke-test` to confirm the model integrates with the full training loop.

## Notes

- `MultiScaleDiscriminator.forward()` returns `list[Tensor]`, not a single tensor — if replacing it, update `multiscale_*_loss` callers in `losses.py`
- `BaseGenerator.sample()` calls `self(z)` — ensure `forward` accepts the right input shape
- All new models must pass `make typecheck` (mypy strict mode)
