document
  .getElementById("gejala-form")
  .addEventListener("submit", function (event) {
    event.preventDefault(); // Mencegah form untuk disubmit secara default

    // Ambil semua gejala yang dipilih (checkbox yang tercentang)
    const gejala = Array.from(
      document.querySelectorAll("input[name='gejala']:checked")
    ).map((cb) => cb.value);

    // Kirim data gejala ke server Flask dengan header yang benar
    fetch("/deteksi", {
      method: "POST",
      headers: {
        "Content-Type": "application/json", // Pastikan headernya application/json
      },
      body: JSON.stringify({ gejala: gejala }), // Mengirimkan gejala dalam format JSON
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Server merespons dengan status: ${res.status}`);
        }
        return res.json(); // Mendapatkan respons dalam format JSON
      })
      .then((data) => {
        // Tampilkan hasil prediksi
        document.getElementById("prediction-result").style.display = "block";
        document.getElementById("predicted-disease").innerText =
          "Penyakit Terdeteksi: " + data.prediction;
        document.getElementById("disease-image").src =
          "/static/images/" + data.gambar; // Pastikan gambar ada di folder static/images
        document.getElementById("disease-description").innerText =
          data.disease_info;
      })
      .catch((error) => {
        console.error("Error:", error);
        alert("Terjadi kesalahan saat melakukan prediksi.");
      });
  });
