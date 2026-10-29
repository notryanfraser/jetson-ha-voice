"""Wyoming STT bridge to a local whisper.cpp whisper-server (HTTP /inference)."""
import argparse
import asyncio
import io
import json
import logging
import urllib.request
import uuid
import wave
from functools import partial

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger("wyoming_whispercpp")


def _post_wav(url: str, wav: bytes, language: str) -> str:
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in (("response_format", "json"), ("language", language), ("temperature", "0.0")):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n".encode() + wav + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        url, data=b"".join(parts), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp).get("text", "").strip()


class Handler(AsyncEventHandler):
    def __init__(self, info: Info, args, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.info = info
        self.args = args
        self.converter = AudioChunkConverter(rate=16000, width=2, channels=1)
        self.audio = bytearray()
        self.language = args.language

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True
        if Transcribe.is_type(event.type):
            self.language = Transcribe.from_event(event).language or self.args.language
            return True
        if AudioChunk.is_type(event.type):
            self.audio += self.converter.convert(AudioChunk.from_event(event)).audio
            return True
        if AudioStop.is_type(event.type):
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(bytes(self.audio))
            self.audio = bytearray()
            text = await asyncio.get_running_loop().run_in_executor(
                None, _post_wav, self.args.server, buf.getvalue(), self.language
            )
            _LOGGER.debug("Transcript: %s", text)
            await self.write_event(Transcript(text=text).event())
            return False
        return True


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--uri", default="tcp://0.0.0.0:10301")
    p.add_argument("--server", default="http://127.0.0.1:8910/inference")
    p.add_argument("--model", default="small.en")
    p.add_argument("--language", default="en")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    attribution = Attribution(name="ggml-org", url="https://github.com/ggml-org/whisper.cpp")
    info = Info(
        asr=[
            AsrProgram(
                name="whisper.cpp",
                description="whisper.cpp with CUDA",
                attribution=attribution,
                installed=True,
                version="1",
                models=[
                    AsrModel(
                        name=args.model,
                        description=args.model,
                        attribution=attribution,
                        installed=True,
                        languages=["en"],
                        version="1",
                    )
                ],
            )
        ]
    )
    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready on %s", args.uri)
    await server.run(partial(Handler, info, args))


if __name__ == "__main__":
    asyncio.run(main())
