import json
import re
import time

import openai

NAMA_FITUR = {
    "ipk": "IPK",
    "kehadiran": "Kehadiran (%)",
    "jam_belajar": "Jam belajar mandiri per minggu",
    "tugas_terkumpul": "Tugas terkumpul (%)",
    "nilai_uts": "Nilai UTS",
    "aktif_organisasi": "Aktif organisasi",
}


def load_config(path="config.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_data(data):
    return "; ".join(f"{nama}: {data[kolom]}" for kolom, nama in NAMA_FITUR.items())


def build_messages(cfg, data):
    msgs = [{"role": "system", "content": cfg["system_prompt"]}]
    for ex in cfg.get("few_shot", []):
        msgs.append({"role": "user", "content": format_data(ex["data"])})
        msgs.append({"role": "assistant", "content": ex.get("jawaban", ex["label"])})
    msgs.append({"role": "user", "content": format_data(data)})
    return msgs


def parse_label(output, labels):
    if not output:
        return "invalid"
    peta = {label.lower(): label for label in labels}
    alt = "|".join(labels)
    teks = output.strip()

    cocok = re.findall(rf"status\s*[:=]\s*\**\s*({alt})", teks, flags=re.I)
    if cocok:
        return peta[cocok[-1].lower()]

    jawab = re.sub(r"[^a-z]", "", teks.lower())
    if jawab in peta:
        return peta[jawab]

    semua = re.findall(alt, teks, flags=re.I)
    if semua:
        return peta[semua[-1].lower()]
    return "invalid"


def classify(client, cfg, data, max_retry=4):
    for percobaan in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                messages=build_messages(cfg, data),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return parse_label(resp.choices[0].message.content, cfg["labels"])
        except openai.RateLimitError:
            time.sleep(15 * (percobaan + 1))
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError):
            time.sleep(5 * (percobaan + 1))
        except openai.APIError:
            return "invalid"
    return "invalid"