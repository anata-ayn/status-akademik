import json
import os
import time

import joblib
import openai
import pandas as pd
import streamlit as st

from classifier import classify, load_config

BATAS_LLM = 10
KOLOM_BARIS = ["ipk", "kehadiran", "jam_belajar", "tugas_terkumpul", "nilai_uts", "aktif_organisasi"]

st.set_page_config(page_title="Klasifikasi Status Akademik", page_icon="🎓", layout="wide")


@st.cache_resource
def muat_model():
    with open("model_info.json", encoding="utf-8") as f:
        info = json.load(f)
    return joblib.load("model_status_akademik.pkl"), info


@st.cache_resource
def muat_llm():
    cfg = load_config("config.json")
    try:
        kunci = st.secrets.get("NVIDIA_API_KEY")
    except Exception:
        kunci = None
    kunci = kunci or os.environ.get("NVIDIA_API_KEY")
    if not kunci:
        return cfg, None
    client = openai.OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=kunci, timeout=90)
    return cfg, client


def prediksi_dnn(model, df):
    hasil = df[KOLOM_BARIS].copy()
    hasil["prediksi_dnn"] = model.predict(hasil)
    hasil["keyakinan"] = model.predict_proba(hasil[KOLOM_BARIS]).max(axis=1).round(3)
    return hasil


def prediksi_llm(client, cfg, df, bar=None):
    out, waktu = [], []
    for i, (_, baris) in enumerate(df.iterrows()):
        mulai = time.time()
        out.append(classify(client, cfg, baris.to_dict()))
        waktu.append(round(time.time() - mulai, 1))
        if bar:
            bar.progress((i + 1) / len(df))
    hasil = df[KOLOM_BARIS].copy()
    hasil["prediksi_llm"] = out
    hasil["detik"] = waktu
    return hasil


def form_input(info):
    r = info["rentang_fitur"]
    k1, k2, k3 = st.columns(3)
    with k1:
        ipk = st.number_input("IPK", 0.0, 4.0, 3.0, step=0.01)
        kehadiran = st.number_input("Kehadiran (%)", 0, 100, 80)
    with k2:
        jam = st.number_input("Jam belajar mandiri per minggu", 0, 60, 10)
        tugas = st.number_input("Tugas terkumpul (%)", 0, 100, 80)
    with k3:
        uts = st.number_input("Nilai UTS", 0, 100, 70)
        org = st.selectbox("Aktif organisasi", info["nilai_kategori"]["aktif_organisasi"])
    return pd.DataFrame([{
        "ipk": ipk, "kehadiran": kehadiran, "jam_belajar": jam,
        "tugas_terkumpul": tugas, "nilai_uts": uts, "aktif_organisasi": org,
    }])


def cek_csv(df, info):
    kurang = [k for k in KOLOM_BARIS if k not in df.columns]
    if kurang:
        return f"Kolom berikut tidak ada di file: {', '.join(kurang)}"
    for k, nilai in info["nilai_kategori"].items():
        salah = sorted(set(df[k].astype(str)) - set(nilai))
        if salah:
            return f"Kolom {k} berisi nilai di luar {nilai}: {salah}"
    for k in info["fitur_numerik"]:
        if pd.to_numeric(df[k], errors="coerce").isna().any():
            return f"Kolom {k} harus berisi angka dan tidak boleh kosong"
    return None


model, info = muat_model()
cfg, client = muat_llm()
ev = cfg.get("evaluasi", {})

st.title("Klasifikasi Status Akademik Mahasiswa")
st.caption("Kelas: Aman, Waspada, Berisiko. Perbandingan model DNN (Modul 1B) dan LLM lewat AI model service (Modul 2B).")

with st.sidebar:
    st.subheader("Model DNN")
    st.write(info["nama_model"])
    st.metric("F1 macro uji", info["metrik"]["f1_macro"])
    st.metric("Accuracy uji", info["metrik"]["accuracy"])
    st.subheader("Model LLM")
    st.write(cfg["model"])
    if ev:
        st.metric("F1 macro uji", ev.get("f1_macro"))
        st.metric("Accuracy uji", ev.get("accuracy"))
    if client is None:
        st.warning("NVIDIA_API_KEY belum diisi, bagian LLM tidak aktif.")
    with open("contoh_data_baru.csv", "rb") as f:
        st.download_button("Unduh contoh data uji (CSV)", f, "contoh_data_baru.csv", "text/csv")

tab1, tab2, tab3 = st.tabs(["Input satu mahasiswa", "Unggah CSV", "Tentang model"])

with tab1:
    with st.form("form_mhs"):
        satu = form_input(info)
        kirim = st.form_submit_button("Prediksi")
    if kirim:
        dnn = prediksi_dnn(model, satu)
        llm_label, llm_detik = None, None
        if client is not None:
            with st.spinner("Menunggu jawaban LLM, sekitar 10 sampai 15 detik..."):
                h = prediksi_llm(client, cfg, satu)
            llm_label, llm_detik = h.loc[0, "prediksi_llm"], h.loc[0, "detik"]
        st.session_state["hasil_satu"] = {
            "dnn": dnn.loc[0, "prediksi_dnn"],
            "keyakinan": float(dnn.loc[0, "keyakinan"]),
            "proba": dict(zip(model.classes_, model.predict_proba(satu[KOLOM_BARIS])[0].round(3).tolist())),
            "luar": [k for k, (lo, hi) in info["rentang_fitur"].items() if not lo <= float(satu[k].iloc[0]) <= hi],
            "llm": llm_label,
            "detik": llm_detik,
        }
    hs = st.session_state.get("hasil_satu")
    if hs:
        if hs["luar"]:
            st.warning(
                f"Nilai {', '.join(hs['luar'])} di luar rentang data latih, prediksi model kurang dapat diandalkan."
            )
        k1, k2 = st.columns(2)
        with k1:
            st.markdown("**Model DNN**")
            st.success(hs["dnn"])
            st.caption(f"Keyakinan {hs['keyakinan']:.1%}")
            st.caption("Probabilitas: " + ", ".join(f"{k} {v:.1%}" for k, v in hs["proba"].items()))
        with k2:
            st.markdown("**Model LLM**")
            if hs["llm"] is None:
                st.info("API key belum diisi.")
            else:
                st.success(hs["llm"])
                st.caption(f"Waktu {hs['detik']} detik")

with tab2:
    st.write(
        "Unggah CSV dengan kolom: " + ", ".join(KOLOM_BARIS)
        + ". Kolom lain seperti id_mahasiswa boleh ada. Contoh file ada di sidebar."
    )
    berkas = st.file_uploader("File CSV", type="csv")
    if berkas is not None:
        try:
            data = pd.read_csv(berkas)
        except Exception as e:
            st.error(f"File tidak bisa dibaca: {e}")
            data = None
        if data is not None:
            galat = cek_csv(data, info)
            if galat:
                st.error(galat)
            else:
                st.write(f"{len(data)} baris terbaca")
                st.dataframe(data, use_container_width=True)
                pakai_llm = st.checkbox(
                    f"Sertakan prediksi LLM (maksimal {BATAS_LLM} baris pertama, sekitar 12 detik per baris)",
                    value=client is not None, disabled=client is None,
                )
                if st.button("Jalankan prediksi"):
                    gabung = prediksi_dnn(model, data)
                    for kolom in ("id_mahasiswa",):
                        if kolom in data.columns:
                            gabung.insert(0, kolom, data[kolom].values)
                    if pakai_llm and client is not None:
                        bar = st.progress(0.0)
                        h = prediksi_llm(client, cfg, data.head(BATAS_LLM), bar)
                        gabung["prediksi_llm"] = None
                        gabung.loc[h.index, "prediksi_llm"] = h["prediksi_llm"].values
                    st.session_state["hasil_csv"] = gabung
                    st.session_state["nama_csv"] = berkas.name
                if st.session_state.get("nama_csv") == berkas.name and "hasil_csv" in st.session_state:
                    gabung = st.session_state["hasil_csv"]
                    st.dataframe(gabung, use_container_width=True)
                    st.download_button(
                        "Unduh hasil (CSV)", gabung.to_csv(index=False).encode("utf-8"),
                        "hasil_prediksi.csv", "text/csv",
                    )

with tab3:
    m = info["metrik"]
    st.write(
        f"Model DNN: {info['nama_model']}, dilatih pada {m['n_train']} data dan diuji pada {m['n_test']} data. "
        f"F1 macro {m['f1_macro']} dibanding baseline {m['f1_baseline']}."
    )
    if ev:
        st.write(
            f"Model LLM: {cfg['model']}, F1 macro {ev.get('f1_macro')}, accuracy {ev.get('accuracy')}, "
            f"invalid {ev.get('invalid')}, rata-rata {ev.get('waktu_per_prediksi_detik')} detik per prediksi."
        )
    st.json(info["versi"])
