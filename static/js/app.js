class DiseasePredictor {
  constructor() {
    this.args = {
      form: document.getElementById("gejalaForm"),
      submitButton: document.getElementById("submitBtn"),
      resultBox: document.getElementById("predictionResult"),
    };

    this.predictionResult = "";
  }

  display() {
    const { form, submitButton } = this.args;

    if (!form || !submitButton) {
      console.error("Form atau tombol submit tidak ditemukan!");
      return;
    }

    // Tangani submit form secara manual
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      this.onSubmitForm();
    });

    // Opsional, jika ada tombol khusus selain submit
    submitButton.addEventListener("click", (event) => {
      event.preventDefault();
      this.onSubmitForm();
    });
  }

  // Fungsi mengambil data gejala dari form dan mengirim ke backend
  onSubmitForm() {
    const { form, resultBox } = this.args;

    // Ambil semua input radio gejala (nama gejala1, gejala2, dst)
    let gejalaArray = [];
    let allSelected = true;

    for (let i = 1; i <= 36; i++) {
      const selected = form.querySelector(`input[name="gejala${i}"]:checked`);
      if (!selected) {
        allSelected = false;
        break;
      }
      gejalaArray.push(parseInt(selected.value));
    }

    if (!allSelected) {
      alert(
        "Silakan pilih semua gejala (Ya/Tidak) sebelum melakukan prediksi."
      );
      return;
    }

    // Kirim data gejala ke server
    fetch("/deteksi", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gejala: gejalaArray }),
    })
      .then((response) => response.json())
      .then((data) => {
        this.predictionResult = data.hasil || "Tidak ada hasil prediksi";
        this.updateResult();
      })
      .catch((error) => {
        console.error("Error saat prediksi:", error);
        this.predictionResult = "Terjadi kesalahan saat prediksi.";
        this.updateResult();
      });
  }

  // Fungsi update hasil prediksi ke dalam halaman
  updateResult() {
    const { resultBox } = this.args;
    if (resultBox) {
      resultBox.textContent = `Hasil Prediksi: ${this.predictionResult}`;
    } else {
      console.error("Elemen untuk menampilkan hasil prediksi tidak ditemukan.");
    }
  }
}

// Jalankan saat DOM siap
document.addEventListener("DOMContentLoaded", () => {
  const predictor = new DiseasePredictor();
  predictor.display();
});
