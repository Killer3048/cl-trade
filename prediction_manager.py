import asyncio
import logging
import queue
from threading import Thread, Event
from typing import Dict

from moment_classifier import MomentClassifier

logger = logging.getLogger(__name__)


class _PredictionRequest:
    def __init__(self, df, loop: asyncio.AbstractEventLoop):
        self.df = df
        self.loop = loop
        self.future = loop.create_future()


class PredictionManager(Thread):
    """Runs predictions in a dedicated thread owning the model."""

    def __init__(self, model_config: Dict):
        super().__init__(daemon=True)
        self.model_config = model_config
        self.queue: queue.Queue[_PredictionRequest | None] = queue.Queue()
        self.model: MomentClassifier | None = None
        self.running = True
        self.ready = Event()

    def run(self) -> None:
        try:
            self.model = MomentClassifier(self.model_config)
            self.model.load_model()
            logger.info("PredictionManager: model loaded and ready")
        except Exception as e:
            logger.exception("PredictionManager failed to load model: %s", e)
            self.model = None
        finally:
            self.ready.set()

        while self.running:
            req = self.queue.get()
            if req is None:
                break
            try:
                result = self.model.predict(req.df) if self.model else {}
                req.loop.call_soon_threadsafe(req.future.set_result, result)
            except Exception as e:
                logger.exception("PredictionManager: error during prediction: %s", e)
                req.loop.call_soon_threadsafe(req.future.set_exception, e)
            finally:
                self.queue.task_done()

    async def predict(self, df) -> Dict[str, str]:
        loop = asyncio.get_running_loop()
        req = _PredictionRequest(df, loop)
        self.queue.put(req)
        return await req.future

    def stop(self):
        self.running = False
        self.queue.put(None)
