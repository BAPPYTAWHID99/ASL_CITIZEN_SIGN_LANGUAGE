"""Django REST Framework views for ASL inference."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .apps import DEVICE, EXPECTED_SHAPE, INFERENCE_LOCK, LABEL_MAP, MODEL


def parse_sequence(payload: Any) -> torch.Tensor:
    if not isinstance(payload, dict) or "sequence" not in payload:
        raise ValueError("Request JSON must contain a 'sequence' field.")
    sequence = np.asarray(payload["sequence"], dtype=np.float32)
    if sequence.shape == (1, *EXPECTED_SHAPE):
        sequence = sequence[0]
    if sequence.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"'sequence' must have shape {EXPECTED_SHAPE}; received {sequence.shape}."
        )
    if not np.isfinite(sequence).all():
        raise ValueError("'sequence' must contain only finite numbers.")
    return torch.from_numpy(sequence).unsqueeze(0).to(DEVICE)


class PredictView(APIView):
    """Return the top-1 ASL word for one 200-frame sequence."""

    def post(self, request: Request) -> Response:
        try:
            sequence = parse_sequence(request.data)
            with INFERENCE_LOCK, torch.inference_mode():
                started_at = time.time()
                probabilities = torch.softmax(MODEL(sequence), dim=1)
                print(
                    f"Model inference time: {(time.time() - started_at) * 1000:.2f} ms",
                    flush=True,
                )
                confidence, predicted_index = probabilities.max(dim=1)
            class_id = int(predicted_index.item())
            return Response(
                {
                    "word": LABEL_MAP[class_id],
                    "confidence": round(float(confidence.item()), 6),
                }
            )
        except (TypeError, ValueError, OverflowError, KeyError) as error:
            return Response({"error": str(error)}, status=status.HTTP_400_BAD_REQUEST)


predict = PredictView.as_view()
