from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt


def plt_to_html(fig) -> str:
    """Serialize a Matplotlib figure into an inline HTML image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{img}"/>'


__all__ = [
    "plt_to_html",
]
