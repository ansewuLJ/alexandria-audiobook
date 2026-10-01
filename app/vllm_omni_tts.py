"""Minimal vLLM-Omni TTS client for the OpenAI-compatible speech API."""

import io
import os
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
import soundfile as sf


VOICE_TYPES = {"custom": "CustomVoice", "clone": "Base", "design": "VoiceDesign"}


class VLLMOmniTTS:
    # Tried in order whenever the preferred format is not accepted. PCM leads because
    # it is lossless, accepted by every OpenAI-compatible speech endpoint tested, and
    # costs the server no encoding work; MP3 covers the services that only speak it;
    # WAV trails because cloud gateways tend to reject it outright (OpenRouter's
    # request schema has no "wav" value at all).
    FORMAT_FALLBACK = ("pcm", "mp3", "wav")

    def __init__(self, config):
        self.base_url = (config.get("base_url") or "").rstrip("/")
        self.api_key = config.get("api_key") or ""
        self.model = (config.get("model") or "").strip()
        # vLLM-Omni routes between a model's variants with this field, and falls back
        # to CustomVoice when it is absent -- which makes a VoiceDesign deployment
        # hunt for a preset voice it does not have. Not part of the OpenAI spec:
        # cloud gateways ignore it, but *any* model served by vLLM-Omni reacts to it
        # (BreezeTTS2 produces different audio for "VoiceDesign" than for the
        # default). Empty means "do not send it", which is the model's own default.
        self.model_type = (config.get("model_type") or "").strip().lower()
        # Only the *first* candidate -- see _candidates() for what actually gets sent.
        self.response_format = (config.get("response_format") or "pcm").strip().lower()
        self.concurrency = max(1, min(int(config.get("concurrency", 16)), 32))
        # Whichever format last succeeded, so a long batch pays the fallback search
        # once rather than on every chunk.
        self._resolved_format = None

    def _url(self, path):
        if not self.base_url or not self.model:
            raise ValueError("Set vLLM-Omni Base URL and Model Name first")
        base = self.base_url if self.base_url.endswith("/v1") else f"{self.base_url}/v1"
        parsed = urlparse(base)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("vLLM-Omni Base URL must be an HTTP(S) address")
        return f"{base}/{path.lstrip('/')}"

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def status(self):
        response = requests.get(self._url("models"), headers=self._headers(), timeout=8)
        response.raise_for_status()
        names = [item.get("id") for item in response.json().get("data", [])]
        # Some services only list a subset of models here: OpenRouter, for example,
        # returns text models by default and hides TTS ones behind
        # ?output_modalities=speech. Absence from this listing does not mean the model
        # is unusable, so treat a successful /v1/models response as "ready" either way.
        listed = self.model in names
        return {"connected": True, "model_ready": True,
                "model": self.model, "listed": listed,
                "detail": "Model is ready" if listed
                else "Model not listed by /v1/models, but the endpoint responded"}

    @staticmethod
    def _reference_data_url(path):
        if not path:
            raise ValueError("Voice Clone needs a reference recording")
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
        with open(path, "rb") as source:
            encoded = base64.b64encode(source.read()).decode("ascii")
        mime = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac"}.get(
            os.path.splitext(path)[1].lower(), "audio/wav"
        )
        return f"data:{mime};base64,{encoded}"

    def _item(self, chunk, voices, language, seed):
        speaker = chunk.get("speaker", "")
        voice = voices.get(speaker, {})
        text = (chunk.get("text") or "").strip()
        if not text:
            raise ValueError(f"Chunk {chunk['index']} has no text")
        # Send only OpenAI spec fields plus per-model optional extensions. There is no
        # "model type" switch: the server interprets these fields according to whichever
        # model it has loaded, so the same client works against any compatible service.
        # "response_format" is deliberately absent: _request owns it and may retry with
        # a different container, so it decides per attempt rather than per item.
        item = {"input": text, "non_streaming_mode": True, "max_new_tokens": 2048}

        # vLLM-Omni variant selector. Omitting it lets the server pick its own
        # default (CustomVoice), so only send it when the config names a mode.
        task_type = VOICE_TYPES.get(self.model_type)
        if task_type:
            item["task_type"] = task_type

        # Preset voice name (lookup-style models such as Qwen3-TTS CustomVoice).
        if voice.get("voice"):
            item["voice"] = voice["voice"]

        # Voice description plus emotion/style instruction (design-style models).
        parts = [voice.get("description", ""), chunk.get("instruct", "")]
        instructions = ", ".join(p.strip() for p in parts if p and p.strip())
        if instructions:
            item["instructions"] = instructions

        # Reference audio (clone-style models such as Qwen3-TTS Base, MOSS-TTS-Nano).
        if voice.get("ref_audio"):
            item["ref_audio"] = self._reference_data_url(voice["ref_audio"])
            item["ref_text"] = voice.get("ref_text") or ""

        # "language" is not an OpenAI spec field and some backends (e.g. BreezeTTS2)
        # reject it outright. Only send it when a concrete language was requested;
        # "Auto" and empty values are omitted so the model infers the language itself.
        if language and str(language).strip().lower() not in ("auto", ""):
            item["language"] = language

        if seed is not None and int(seed) >= 0:
            item["seed"] = int(seed)
        return item

    def _candidates(self):
        """Formats to try, best guess first.

        The configured preference leads; whichever format last succeeded is then
        promoted to the front so a batch does not re-run the search on every chunk.
        """
        order = [self.response_format]
        order += [fmt for fmt in self.FORMAT_FALLBACK if fmt != self.response_format]
        if self._resolved_format in order:
            order.remove(self._resolved_format)
            order.insert(0, self._resolved_format)
        return order

    def _request(self, item):
        failure = ""
        for fmt in self._candidates():
            response = requests.post(self._url("audio/speech"),
                                     headers=self._headers(),
                                     json={"model": self.model, **item,
                                           "response_format": fmt},
                                     timeout=600)
            if response.ok:
                content = response.content
                if not content:
                    raise ValueError("vLLM-Omni returned empty audio")
                self._resolved_format = fmt
                return self._normalise_audio(content,
                                             response.headers.get("Content-Type", ""))
            failure = f"HTTP {response.status_code}: {response.text[:300]}"
            # 400 is how these APIs report an unsupported response_format (Gemini:
            # 'only supports response_format="pcm"'; OpenRouter rejects unknown enum
            # values before routing at all). Anything else -- auth, rate limit, bad
            # input, 5xx -- will not be fixed by changing the container, so fail fast
            # instead of tripling the load on a service that is already unhappy.
            if response.status_code != 400:
                break
        raise RuntimeError(f"vLLM-Omni {failure}")

    @classmethod
    def _normalise_audio(cls, content, content_type=""):
        """Normalise whatever the service returned into a real RIFF/WAV payload.

        The downstream pipeline reads every chunk back with ``AudioSegment.from_wav``
        (see project.py), which parses the RIFF header rather than sniffing the
        container -- so raw MP3 or headerless PCM fails there even though the bytes
        are perfectly good audio. Three cases:

        * WAV            -- already RIFF, pass through untouched.
        * headerless PCM -- prepend a 44-byte RIFF header. Sample rate and channel
          count differ per backend (Gemini 24 kHz, Fish Audio 44.1 kHz, ...), so
          read what the service reports instead of assuming, and fall back to
          24 kHz mono.
        * MP3            -- decode to PCM and re-wrap. The decode itself is
          lossless: whatever quality was lost, the service lost it when it encoded
          the MP3, and re-wrapping can neither undo nor add to that.
        """
        if content[:4] == b"RIFF":
            return content
        if "pcm" in (content_type or "").lower():
            sample_rate, channels = cls._parse_pcm_params(content_type)
            return cls._wrap_pcm_as_wav(content, sample_rate=sample_rate,
                                        channels=channels)
        if cls._looks_like_mp3(content, content_type):
            return cls._mp3_to_wav(content)
        return content

    @staticmethod
    def _looks_like_mp3(content, content_type=""):
        """Identify MP3 by Content-Type first, then by magic bytes.

        Not every service labels its response honestly, so fall back to sniffing:
        either an ID3v2 tag, or a bare MPEG frame sync (the first 11 bits set).
        """
        if any(token in (content_type or "").lower() for token in ("mpeg", "mp3")):
            return True
        if content[:3] == b"ID3":
            return True
        return len(content) >= 2 and content[0] == 0xFF and (content[1] & 0xE0) == 0xE0

    @staticmethod
    def _mp3_to_wav(mp3):
        """Decode an MP3 payload and re-wrap it as WAV."""
        from pydub import AudioSegment

        try:
            segment = AudioSegment.from_file(io.BytesIO(mp3), format="mp3")
        except Exception as exc:
            raise ValueError(
                f"Service returned MP3 but decoding it failed ({exc}). This path needs "
                f"ffmpeg on PATH; requesting response_format 'pcm' or 'wav' avoids it."
            ) from exc
        if len(segment) == 0:
            raise ValueError("Service returned an MP3 payload with no audio in it")
        # pydub always hands back 16-bit PCM, so reuse the PCM wrapper rather than
        # spawning a second ffmpeg just to prepend a 44-byte header.
        return VLLMOmniTTS._wrap_pcm_as_wav(
            segment.raw_data,
            sample_rate=segment.frame_rate,
            channels=segment.channels,
            bits=segment.sample_width * 8,
        )

    @staticmethod
    def _parse_pcm_params(content_type: str) -> tuple[int, int]:
        """Read sample rate and channel count from a Content-Type header.

        Expects the form ``audio/pcm;rate=24000;channels=1``; anything missing
        falls back to 24 kHz mono, the common case for TTS output.
        """
        sample_rate, channels = 24000, 1
        for part in (content_type or "").split(";"):
            key, _, value = part.strip().partition("=")
            key = key.lower()
            if not value:
                continue
            if key == "rate":
                try:
                    sample_rate = int(value)
                except ValueError:
                    pass
            elif key == "channels":
                try:
                    channels = int(value)
                except ValueError:
                    pass
        return sample_rate, channels

    @staticmethod
    def _wrap_pcm_as_wav(pcm: bytes, sample_rate: int = 24000,
                         channels: int = 1, bits: int = 16) -> bytes:
        """Wrap headerless 16-bit PCM in a minimal WAV container."""
        import struct

        byte_rate = sample_rate * channels * bits // 8
        block_align = channels * bits // 8
        header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
        header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels,
                                        sample_rate, byte_rate, block_align, bits)
        header += b"data" + struct.pack("<I", len(pcm))
        return header + pcm

    def generate_one(self, chunk, voices, language, seed=None):
        return self._request(self._item(chunk, voices, language, seed))

    def generate_batch(self, chunks, voices, language, seed=None,
                       on_progress=None, on_result=None):
        results = {}
        items = {}
        for chunk in chunks:
            try:
                items[chunk["index"]] = self._item(chunk, voices, language, seed)
            except Exception as exc:
                results[chunk["index"]] = (None, str(exc))
                if on_result:
                    on_result(chunk["index"], None, str(exc))

        def run(index):
            try:
                return index, (self._request(items[index]), None)
            except Exception as exc:
                return index, (None, str(exc))

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(run, index) for index in items]
            for future in as_completed(futures):
                index, result = future.result()
                results[index] = result
                if on_result:
                    on_result(index, result[0], result[1])
                if on_progress:
                    done = sum(audio is not None for audio, _ in results.values())
                    failed = sum(error is not None for _, error in results.values())
                    on_progress(done, failed, len(chunks))
        return results
