"""Wyoming TTS server for Phonon (gradium phonon-31m-en, CPU), with streamed text and audio."""
import argparse
import asyncio
import logging
import re
from functools import partial

import numpy as np
import ptts
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped

_LOGGER = logging.getLogger("wyoming_phonon")
_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+")


def split_sentences(text: str) -> tuple[list[str], str]:
    """Complete sentences in text, and the unfinished rest."""
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts[:-1] if p.strip()], parts[-1]


class Handler(AsyncEventHandler):
    def __init__(self, info: Info, args, tts: ptts.TTS, lock: asyncio.Lock, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.info, self.args, self.tts, self.lock = info, args, tts, lock
        self.known = {v.name.lower() for v in info.tts[0].voices}
        self.streaming = False
        self.started = False
        self.voice = args.voice
        self.pending = ""

    async def speak(self, text: str) -> None:
        """Synthesize one piece of text, writing audio chunks as Phonon produces them."""
        text = text.strip()
        if not text:
            return
        rate = self.tts.sample_rate
        if not self.started:
            await self.write_event(AudioStart(rate=rate, width=2, channels=1).event())
            self.started = True
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def produce() -> None:
            try:
                with self.tts.stream(text, voice=self.voice) as audio:
                    for pcm in audio:
                        pcm16 = (np.clip(np.asarray(pcm, dtype=np.float32), -1, 1) * 32767).astype("<i2")
                        loop.call_soon_threadsafe(queue.put_nowait, pcm16.tobytes())
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        async with self.lock:
            job = loop.run_in_executor(None, produce)
            while (chunk := await queue.get()) is not None:
                await self.write_event(AudioChunk(rate=rate, width=2, channels=1, audio=chunk).event())
            await job
        _LOGGER.debug("Spoke: %s", text)

    async def finish(self) -> None:
        if not self.started:
            await self.write_event(AudioStart(rate=self.tts.sample_rate, width=2, channels=1).event())
        await self.write_event(AudioStop().event())
        self.started = False

    def pick_voice(self, voice) -> str:
        """The requested voice if Phonon has it, else the default (HA may still ask for a Piper voice)."""
        name = voice.name if voice and voice.name else None
        if name and name.lower() in self.known:
            return name
        if name:
            _LOGGER.debug("No voice %s, using %s", name, self.args.voice)
        return self.args.voice

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True
        if Synthesize.is_type(event.type):
            if self.streaming:
                return True  # HA sends the whole text again after a stream, for older servers
            synth = Synthesize.from_event(event)
            self.voice = self.pick_voice(synth.voice)
            sentences, rest = split_sentences(synth.text)
            for s in sentences + [rest]:
                await self.speak(s)
            await self.finish()
            return True
        if SynthesizeStart.is_type(event.type):
            self.streaming = True
            self.voice = self.pick_voice(SynthesizeStart.from_event(event).voice)
            self.pending = ""
            return True
        if SynthesizeChunk.is_type(event.type):
            self.pending += SynthesizeChunk.from_event(event).text
            sentences, self.pending = split_sentences(self.pending)
            for s in sentences:
                await self.speak(s)
            return True
        if SynthesizeStop.is_type(event.type):
            await self.speak(self.pending)
            self.pending = ""
            await self.finish()
            await self.write_event(SynthesizeStopped().event())
            self.streaming = False
            return True
        return True


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--uri", default="tcp://0.0.0.0:10202")
    p.add_argument("--model", default="hf://gradium/phonon-31m-en")
    p.add_argument("--quant", default="q8")
    p.add_argument("--voice-catalog", default=None, help="hf://owner/name[@revision] or a local voices.jsonl")
    p.add_argument("--voice", default="Maeve")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    tts = ptts.TTS(args.model, lang="en", quant=args.quant, voice_catalog=args.voice_catalog)
    tts.synth("Ready.", voice=args.voice)  # load the voice and warm the model before taking requests
    attribution = Attribution(name="Gradium", url="https://huggingface.co/gradium/phonon-31m-en")
    voices = [
        TtsVoice(
            name=c["name"],
            description=c.get("description") or c["name"],
            attribution=attribution,
            installed=True,
            languages=["en", "en_US", "en_GB"],  # HA matches the pipeline's en_US exactly, as Piper reported
            version="1",
        )
        for c in (ptts.catalog_voices(args.voice_catalog) if args.voice_catalog else ptts.catalog_voices())
        if c.get("language") == "en"
    ]
    info = Info(
        tts=[
            TtsProgram(
                name="phonon",
                description="Phonon 31M on the CPU",
                attribution=attribution,
                installed=True,
                version=ptts.__version__,
                voices=voices,
                supports_synthesize_streaming=True,
            )
        ]
    )
    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready on %s with voice %s", args.uri, args.voice)
    await server.run(partial(Handler, info, args, tts, asyncio.Lock()))


if __name__ == "__main__":
    asyncio.run(main())
