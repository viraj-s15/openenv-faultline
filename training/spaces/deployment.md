## Required Space secrets

- `HF_TOKEN`
- `MODEL_REPO_ID`
- `BASE_MODEL_NAME` when `artifact_mode=adapter`

## Deployment flow

1. Push or sync the Space repo.
2. Copy `training/spaces/README.template.md` to the Space README.
3. Set the required secrets in Space settings.
4. Confirm the runtime listens on port `8000`.
5. Start the Space and verify `/health` and `/state`.
