#!/usr/bin/env python3
"""Capture the system prompt and tools Home Assistant's Ollama agent sends, and install them
on the Jetson as /var/lib/ollama-preload/ha-request.json, which ollama-preload.sh reads at
boot to warm the model. HA's first voice request then takes about 3 s instead of 13 s.

Run it from any machine that can SSH to the Jetson, after setting up the HA pipeline and
again whenever you change which entities are exposed to Assist, their aliases or areas
(a stale copy only makes the first request slow again):

    HA_TOKEN=... python3 tools/capture_ha_prompt.py --ha http://homeassistant.local:8123 \
        --agent conversation.qwen3_4b_instruct --jetson ubuntu@192.168.1.50

It asks the agent one harmless question ("What time is it?") through HA's API while tcpdump
records the request on the Jetson. HA_TOKEN is a long-lived access token (HA profile page).
The Jetson's sudo password is read from --sudo-password-file, $JETSON_SUDO_PASSWORD or a prompt.
"""
import argparse, getpass, json, os, re, socket, struct, subprocess, sys, tempfile, time, urllib.parse, urllib.request


def requests_in(pcap):
    d = open(pcap, "rb").read(); off = 24; streams = {}
    while off < len(d):
        _, _, cl, _ = struct.unpack("<IIII", d[off:off + 16]); pkt = d[off + 16:off + 16 + cl]; off += 16 + cl
        ip = pkt[20:]  # tcpdump -i any writes Linux cooked (SLL2) frames: a 20-byte header, then IP
        tcp = ip[(ip[0] & 15) * 4:]
        sport, dport, seq = struct.unpack(">HHI", tcp[:8]); pl = tcp[(tcp[12] >> 4) * 4:]
        if dport == 11434 and pl:
            streams.setdefault(sport, {})[seq] = pl
    for s in streams.values():
        data = b"".join(s[k] for k in sorted(s))
        for m in re.finditer(rb"POST /api/chat HTTP/1.1\r\n(.*?)\r\n\r\n", data, re.S):
            n = int(re.search(rb"(?i)content-length: (\d+)", m.group(1)).group(1))
            yield json.loads(data[m.end():m.end() + n])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ha", required=True, help="Home Assistant URL, e.g. http://homeassistant.local:8123")
    p.add_argument("--agent", required=True, help="the Ollama conversation entity, e.g. conversation.qwen3_4b_instruct")
    p.add_argument("--jetson", required=True, help="SSH destination of the Jetson, e.g. ubuntu@192.168.1.50")
    p.add_argument("--token-file", help="file holding the HA token (default: $HA_TOKEN)")
    p.add_argument("--sudo-password-file", help="file holding the Jetson's sudo password")
    a = p.parse_args()

    token = (open(a.token_file).read() if a.token_file else os.environ.get("HA_TOKEN", "")).strip()
    if not token:
        sys.exit("Set HA_TOKEN or pass --token-file")
    if a.sudo_password_file:
        sudo = open(a.sudo_password_file, "rb").read().strip() + b"\n"
    else:
        sudo = (os.environ.get("JETSON_SUDO_PASSWORD") or getpass.getpass(f"sudo password on {a.jetson}: ")).encode() + b"\n"

    def ssh(cmd, as_root=False):
        full = f'sudo -S -p "" sh -c {json.dumps(cmd)}' if as_root else cmd
        return subprocess.run(["ssh", a.jetson, full], input=sudo if as_root else None,
                              capture_output=True, check=True).stdout.decode()

    ha_ip = socket.gethostbyname(urllib.parse.urlparse(a.ha).hostname)
    ssh("rm -f /tmp/ha-prompt.pcap; nohup timeout 120 tcpdump -i any -s 0 -U -w /tmp/ha-prompt.pcap "
        f"tcp dst port 11434 and host {ha_ip} >/dev/null 2>&1 &", as_root=True)
    time.sleep(2)
    req = urllib.request.Request(f"{a.ha.rstrip('/')}/api/conversation/process",
                                 data=json.dumps({"text": "What time is it?", "language": "en", "agent_id": a.agent}).encode(),
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    print("HA answered:", json.load(urllib.request.urlopen(req, timeout=120))["response"]["speech"]["plain"]["speech"])
    ssh("pkill -x tcpdump; sleep 1; chmod 644 /tmp/ha-prompt.pcap", as_root=True)
    with tempfile.TemporaryDirectory() as t:
        subprocess.run(["scp", "-q", f"{a.jetson}:/tmp/ha-prompt.pcap", t + "/c.pcap"], check=True)
        ssh("rm -f /tmp/ha-prompt.pcap", as_root=True)
        first = next((r for r in requests_in(t + "/c.pcap") if r.get("tools")), None)
        if not first:
            sys.exit("No request with tools was captured. Is the agent an Ollama agent with 'Control Home Assistant' on?")
        out = {"model": first["model"], "messages": [first["messages"][0]], "tools": first["tools"],
               "options": first.get("options", {})}
        open(t + "/ha-request.json", "w").write(json.dumps(out))
        subprocess.run(["scp", "-q", t + "/ha-request.json", f"{a.jetson}:/tmp/ha-request.json"], check=True)
    ssh("install -d /var/lib/ollama-preload && install -m 644 /tmp/ha-request.json /var/lib/ollama-preload/ha-request.json "
        "&& rm /tmp/ha-request.json", as_root=True)
    print(f"Installed: model {out['model']}, system prompt {len(out['messages'][0]['content'])} chars, "
          f"{len(out['tools'])} tools, options {out['options']}")


if __name__ == "__main__":
    main()
