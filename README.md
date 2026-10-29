# jetson-ha-voice

An Ansible playbook that turns an NVIDIA Jetson Orin Nano into a local voice server for [Home Assistant](https://www.home-assistant.io/) Assist. It runs speech-to-text, text-to-speech and the conversation LLM on the Jetson, so a Home Assistant Voice PE (or any Assist satellite) works without the cloud.

| Piece | What runs | Port | Home Assistant integration |
|---|---|---|---|
| Speech-to-text | [whisper.cpp](https://github.com/ggml-org/whisper.cpp) `small.en`, built with CUDA, behind a small Wyoming bridge | 10300 | Wyoming Protocol |
| Text-to-speech | [Phonon](https://huggingface.co/gradium/phonon-31m-en) 31M on the CPU, voice Maeve, as a streaming Wyoming server | 10200 | Wyoming Protocol |
| Conversation | [Ollama](https://ollama.com/) with Qwen3-4B-Instruct-2507 (Q4_K_M), loaded at boot and kept on the GPU | 11434 | Ollama |

Measured on an Orin Nano Super 8 GB:

- Speech-to-text: about 0.3 s from the end of speech to the transcript.
- Text-to-speech: about 40 ms to first audio, and it starts on the LLM's first sentence.
- LLM: a median of 3.2 s per command.
- Accuracy: 22.5 of 24 on a benchmark of house commands, with no wrong actions.

## Requirements

- **Hardware:** a Jetson Orin Nano Developer Kit (8 GB), ideally the Super kit or one updated to JetPack 6.2 so the `MAXN_SUPER` power mode exists. Use an NVMe SSD with at least 15 GB free for the models, the build, Ollama and the swap file.
- **Software:** JetPack 6.x (L4T R36) with the CUDA toolkit installed, as in the stock SD card image or SDK Manager's default. Tested on JetPack 6.2.3 (L4T 36.5.2).
- **Access:** an account on the Jetson with sudo, reachable over SSH with a key.
- **Ansible:** ansible-core 2.16 or newer on the machine you run it from (Linux, macOS or WSL).

The playbook checks the board and the L4T release, and stops on anything else.

## Quick start

```sh
git clone https://github.com/notryanfraser/jetson-ha-voice.git
cd jetson-ha-voice
python3 -m pip install --user ansible-core
ansible-galaxy collection install -r requirements.yml

cp inventory/hosts.example.yml inventory/hosts.yml
$EDITOR inventory/hosts.yml        # your Jetson's address and user

ansible-playbook site.yml -K       # asks for the Jetson's sudo password
```

The first run is long, mostly compiling whisper.cpp's CUDA kernels and downloading the 2.5 GB LLM. Later runs take a minute, change nothing unless a setting changed, and restart only the services whose configuration did.

Reboot once afterwards so the text-mode boot and the freed memory take effect:

```sh
ssh <user>@<jetson> sudo reboot
```

Run one part with tags: `--tags stt`, `--tags tts`, `--tags llm` or `--tags system`. Preview changes with `--check --diff`.

## Set up Home Assistant

1. **Speech-to-text:** Settings > Devices & services > Add integration > **Wyoming Protocol**. Host: the Jetson's IP, port `10300`.
2. **Text-to-speech:** add **Wyoming Protocol** again with port `10200`. It shows up as "phonon".
3. **Conversation:** add **Ollama**, URL `http://<jetson>:11434`, model `qwen3:4b-instruct-2507-q4_K_M`. In the agent's options:
   - Control Home Assistant: **Assist**
   - Context window size: **8192**. This must equal `ollama_context`, or every request reloads the model.
   - Keep alive: **-1**
   - Think: **off**
4. **Pipeline:** Settings > Voice assistants > your assistant:
   - Conversation agent: the Ollama agent. Turn on **Prefer handling commands locally**, so simple commands like "turn on the kitchen lights" skip the LLM.
   - Speech-to-text: whisper.cpp.
   - Text-to-speech: phonon. Pick a voice, for example Maeve.
5. **Expose fewer entities.** Every exposed entity goes into the LLM's prompt, and a smaller prompt means faster answers. Settings > Voice assistants > Expose: keep one entity per thing a person would name, and add aliases for spoken names.
6. **Optional: warm the prompt at boot.** At boot the model is loaded, but Home Assistant's first request still has to read its whole prompt, about 6,000 tokens or 10 s. To skip that wait, capture the prompt once and the preload service will read it at every boot:
   ```sh
   HA_TOKEN=<long-lived token> python3 tools/capture_ha_prompt.py \
     --ha http://homeassistant.local:8123 --agent conversation.<your_agent> --jetson <user>@<jetson>
   ```
   Re-run it when you change exposed entities, aliases or areas.

## What the playbook changes on the Jetson

The Orin Nano's 8 GB is shared by the CPU and GPU, and the LLM needs about 3.9 GB of it. Most system changes free memory or keep latency steady.

| Change | Why | Variable |
|---|---|---|
| Power mode `MAXN_SUPER`, `jetson_clocks` at boot | Full GPU and memory clocks, so speed doesn't vary | `jetson_power_mode`, `jetson_lock_clocks` |
| Fan profile `cool` | Locked clocks run warm | `jetson_fan_profile` |
| Boot to text mode | The desktop takes about 800 MB | `jetson_headless` |
| Disable snapd, ModemManager, Bluetooth, the CSI camera daemon; mask PackageKit | Unused memory and CPU | `jetson_disable_services`, `jetson_mask_services` |
| 4 GB swap file with swappiness 10, instead of zram | zram compresses swap into the same RAM the GPU needs | `jetson_swap_*` |
| Persistent journal capped at 200 MB | Logs survive a reboot, for troubleshooting | `jetson_journal_max_use` |
| Packages: git, cmake, build-essential, python3-venv, tcpdump | Building whisper.cpp; the prompt capture | |

Services, all installed under `voice_base_dir` (default `/opt/ha-voice`) and run as `voice_user`:

| Unit | What |
|---|---|
| `whisper-cpp-server.service` | whisper.cpp's HTTP server on 127.0.0.1:8910, with `--suppress-nst`, so a satellite's chime is not transcribed as "\*Ding\*" |
| `wyoming-whispercpp.service` | Wyoming STT on 10300, forwarding audio to whisper-server |
| `wyoming-phonon.service` | Wyoming TTS on 10200. It runs offline, using the model and voices downloaded during setup. It takes Home Assistant's streamed text and speaks each sentence as soon as it is complete. |
| `ollama.service` | Ollama, with an override: listens on all interfaces, keeps models loaded forever, one model and one request at a time, and turns off llama-server's RAM prompt cache, which otherwise filled RAM until the OOM killer stopped Ollama |
| `ollama-preload.service` | At boot, waits for whisper.cpp and Phonon, then loads the model with the same context as Home Assistant. If Ollama put part of it on the CPU while memory was still being freed, it unloads, drops caches and retries. |

## Choosing other models or voices

Everything is in [`group_vars/all.yml`](group_vars/all.yml); override it per host in your inventory.

- **TTS voice:** `phonon_voice`. Phonon has 37 English voices, 12 of them characters (Captain Silas Blackwater, the Snow Queen and others). The server runs offline, so it can speak only the voices downloaded during setup: the default, plus any in `phonon_extra_voices`. Home Assistant lists every catalog voice, but one that wasn't downloaded fails. Add voices to `phonon_extra_voices` and re-run, then pick one per pipeline or per `tts.speak` call.
- **STT model:** `whisper_model`, plus `whisper_model_checksum` (set it to `""` to skip the check). `base.en` is faster but noticeably less accurate. `medium.en` does not leave enough memory for the LLM.
- **LLM:** `ollama_model` and `ollama_context`. Change Home Assistant's agent to match.

## Why these choices

Each choice was benchmarked on the Orin Nano Super, measured on its own and with everything else running:

- **Speech-to-text:** on 48 recorded house commands, `whisper_trt` with `base.en` got 5 of the 36 natural voices wrong; whisper.cpp `small.en` got none wrong. A TensorRT build of `small.en` runs out of memory on 8 GB.
- **Text-to-speech, on the CPU, so the GPU is left for the LLM:**

  | Engine | First audio | Speed vs real time | Peak memory |
  |---|---|---|---|
  | Phonon 31M (q8) | 54 ms | 5.4×, streamed | 542 MB |
  | Piper (medium voice) | 81 ms | 12.4× | ~210 MB |
  | Piper (high voice) | 419 ms | 2.2× | ~210 MB |
  | Kokoro v1.0 (fp32) | 0.96 s | 1.4× | 634 MB |
  | Supertonic (8 steps) | 1.21 s | 1.3× | 496 MB |

  Piper medium is the lighter alternative if memory is tight.
- **LLM:** on 24 commands in Home Assistant's exact request format, scored on whether the right tool was called:

  | Model | Score | Median | Wrong actions |
  |---|---|---|---|
  | Qwen3-4B-Instruct-2507 | 22.5/24 | 3.2 s | 0 |
  | Qwen3.5-4B | 22.0/24 | 7.8 s | 1 |
  | Home-Llama-3.2-3B (Home-LLM) | 19.5/24 | 2.7 s | 3 |
  | LFM3-1.7B (llama.cpp) | 20.5/24 | 2.3 s | 2 |

  Bare llama.cpp `llama-server` was only 0.2 s faster than Ollama, which is not worth leaving Home Assistant's built-in Ollama integration.

## Troubleshooting

- **Logs:** `journalctl -u wyoming-phonon -u wyoming-whispercpp -u whisper-cpp-server -u ollama -u ollama-preload -b`
- **Is the model fully on the GPU?** `curl -s localhost:11434/api/ps` should show `size_vram` equal to `size`. If not, `sudo systemctl restart ollama-preload`.
- **The first request after a reboot is slow:** run `tools/capture_ha_prompt.py` (step 6 above).
- **Home Assistant says the TTS language is not supported:** the server reports `en`, `en_US` and `en_GB`. Set the pipeline's TTS language to one of those.
- **Memory:** `free -m` should show 600-800 MB available with everything loaded. Avoid running other GPU or large services on the board.

## Licenses

This playbook is MIT licensed. It installs third-party software under its own licenses:

- whisper.cpp: MIT. Whisper models: MIT.
- Phonon (`ptts`): MIT or Apache-2.0. Phonon model and voices: Apache-2.0.
- Ollama: MIT.
- Qwen3-4B-Instruct-2507: Apache-2.0.
- Wyoming: MIT.
