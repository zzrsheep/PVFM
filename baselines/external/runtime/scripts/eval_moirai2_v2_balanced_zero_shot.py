# Mechanically extracted; original definitions and internal namespaces retained.
from __future__ import annotations

def _make_predictor(module, *, context_length: int, prediction_length: int, covariate_dim: int, batch_size: int, device: str):
    from uni2ts.model.moirai2 import Moirai2Forecast

    forecast = Moirai2Forecast(
        module=module,
        prediction_length=prediction_length,
        target_dim=1,
        feat_dynamic_real_dim=covariate_dim,
        past_feat_dynamic_real_dim=0,
        context_length=context_length,
    )
    return forecast.create_predictor(batch_size=batch_size, device=device)

