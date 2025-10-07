import streamlit as st
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
#from sklearn.preprocessing import zscore
from scipy.stats import zscore  # ✅ DOĞRU
from tensorflow.keras.models import load_model

# Modeli yükle
model = load_model("model_CONV1D.h5")

# Sayfa başlığı
st.title("🧠 Kişiye Özel Uyku Analizi ve Hormon Tahmini")

# Dosya yükleme
psg_file = st.file_uploader("PSG Dosyasını Yükleyin (.edf)", type="edf")
hypnogram_file = st.file_uploader("Hypnogram Dosyasını Yükleyin (.edf)", type="edf")

if psg_file is not None and hypnogram_file is not None:
    st.success("Dosyalar başarıyla yüklendi. Analiz başlatılıyor...")

    # Geçici dosya yoluna kaydet
    with open("temp_psg.edf", "wb") as f:
        f.write(psg_file.read())
    with open("temp_hyp.edf", "wb") as f:
        f.write(hypnogram_file.read())

    # --- Analiz Fonksiyonu ---
    def analyze_patient(psg_path, hypnogram_path):
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        raw.pick_channels(['EEG Fpz-Cz'])
        annotations = mne.read_annotations(hypnogram_path)
        raw.set_annotations(annotations)

        sfreq = raw.info['sfreq']
        signal = raw.get_data()[0]
        epoch_duration = 30
        samples_per_epoch = int(epoch_duration * sfreq)

        mapping = {
            'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3': 3,
            'Sleep stage 4': 3,
            'Sleep stage R': 4
        }

        annotations_df = pd.DataFrame({
            'onset': raw.annotations.onset,
            'description': raw.annotations.description
        })
        annotations_df = annotations_df[annotations_df['description'].isin(mapping.keys())].reset_index(drop=True)

        epochs = []
        true_labels = []

        for _, row in annotations_df.iterrows():
            start_sample = int(row['onset'] * sfreq)
            end_sample = start_sample + samples_per_epoch
            if end_sample <= len(signal):
                epoch = signal[start_sample:end_sample]
                epoch = zscore(epoch)
                epochs.append(epoch)
                true_labels.append(mapping[row['description']])

        X_individual = np.array(epochs).reshape(-1, samples_per_epoch, 1)
        y_pred = model.predict(X_individual)
        y_pred_classes = np.argmax(y_pred, axis=1)

        df_results = pd.DataFrame({'epochs': y_pred_classes})
        TST = len(df_results[df_results['epochs'] != 0]) * 30 / 60
        N3 = len(df_results[df_results['epochs'] == 3]) * 30 / 60
        REM = len(df_results[df_results['epochs'] == 4]) * 30 / 60
        total_time = len(df_results) * 30 / 60

        sleep_efficiency = (TST / total_time) * 100
        deep_sleep_percent = (N3 / TST) * 100 if TST != 0 else 0
        rem_percent = (REM / TST) * 100 if TST != 0 else 0

        melatonin = (deep_sleep_percent * 0.6) + (sleep_efficiency * 0.3) + ((REM / TST) * 100 * 0.1 if TST != 0 else 0)
        cortisol = 25 - (sleep_efficiency * 0.15) - (TST / 10)
        gh = (N3 * 0.08) + (deep_sleep_percent * 0.2)

        # --- METRİKLERİ GÖSTER ---
        st.subheader("📊 Uyku Kalitesi Metrikleri")
        st.write(f"Toplam Uyku Süresi: **{TST:.2f} dk**")
        st.write(f"Uyku Verimliliği: **%{sleep_efficiency:.2f}**")
        st.write(f"Derin Uyku (N3) Oranı: **%{deep_sleep_percent:.2f}**")
        st.write(f"REM Uyku Oranı: **%{rem_percent:.2f}**")

        # Grafik 1: Uyku Kalitesi
        st.subheader("🛌 Uyku Kalitesi Grafiği")
        fig1, ax1 = plt.subplots()
        ax1.bar(['Uyku Verimliliği', 'Derin Uyku', 'REM'], [sleep_efficiency, deep_sleep_percent, rem_percent])
        ax1.set_ylim(0, 100)
        ax1.set_ylabel("Yüzde (%)")
        ax1.set_title("Uyku Kalitesi Göstergeleri")
        st.pyplot(fig1)

        # Hormonlar
        st.subheader("🧪 Hormon Tahminleri")
        st.write(f"Tahmini **Melatonin**: {melatonin:.2f} ng/mL")
        st.write(f"Tahmini **Kortizol**: {cortisol:.2f} µg/dL")
        st.write(f"Tahmini **Büyüme Hormonu (GH)**: {gh:.2f} ng/mL")

        # Grafik 2: Hormonlar
        fig2, ax2 = plt.subplots()
        bars = ax2.bar(['Melatonin', 'Kortizol', 'GH'], [melatonin, cortisol, gh])
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{height:.2f}', ha='center')
        ax2.set_title("Tahmini Hormon Seviyeleri")
        st.pyplot(fig2)

    # Fonksiyonu çağır
    analyze_patient("temp_psg.edf", "temp_hyp.edf")
