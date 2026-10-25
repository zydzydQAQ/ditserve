"""xDiT execution adapters; model and distributed imports stay lazy."""

XDIT_REVISION = "16c34e7f070de65d9d2c67a6d74e0007dd8b426a"
ENGINE_NAME = "xdit"


def engine_info():
    return {"name": ENGINE_NAME, "revision": XDIT_REVISION}
