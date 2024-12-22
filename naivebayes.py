


class GaussianNaiveBayes:
    def __init__(self):
        self.fitur = None
        self.label = None
        self.kategori = None
        self.probabilitas_prior = {}
        self.mean_fitur = {}
        self.varians_fitur = {}

    def fit(self, X, y):
        self.fitur = np.array(X)
        self.label = np.array(y)

        self.kategori = np.unique(y)

        # Menghitung probabilitas prior untuk setiap kategori
        self.probabilitas_prior = {kategori: np.sum(y == kategori) / len(y) for kategori in self.kategori}

        # Menghitung rata-rata dan variansi untuk setiap fitur di masing-masing kategori
        for kategori in self.kategori:
            indeks_kategori = np.where(y == kategori)[0]
            self.mean_fitur[kategori] = np.mean(self.fitur[indeks_kategori], axis=0)
            self.varians_fitur[kategori] = np.var(self.fitur[indeks_kategori], axis=0)

        return self

    # Fungsi Probabilitas Kepadatan Gaussian
    def gaussian(self, nilai, mean, varians):
        eps = 1e-9  # nilai kecil untuk menghindari pembagian dengan nol
        koefisien = 1.0 / np.sqrt(2.0 * np.pi * (varians + eps))
        eksponen = -((nilai - mean) ** 2) / (2.0 * (varians + eps))
        return koefisien * np.exp(eksponen)

    def predict(self, X):
        X = np.array(X)

        # Memastikan jumlah kolom pada X sesuai dengan data pelatihan
        if X.shape[1] != self.fitur.shape[1]:
            raise ValueError("Dimensi fitur pada data prediksi tidak sesuai dengan data pelatihan.")

        prediksi = []
        for sampel in X:
            probabilitas_posterior = {}
            for kategori in self.kategori:
                # Mulai dengan log probabilitas prior
                probabilitas_posterior[kategori] = np.log(self.probabilitas_prior[kategori])
                # Tambahkan log likelihood untuk setiap fitur
                for indeks_fitur in range(self.fitur.shape[1]):
                    nilai_pdf = self.gaussian_pdf(
                        sampel[indeks_fitur], self.mean_fitur[kategori][indeks_fitur], self.varians_fitur[kategori][indeks_fitur]
                    )
                    probabilitas_posterior[kategori] += np.log(nilai_pdf + 1e-9)  # Tambahkan eps untuk menghindari log(0)
            # Tambahkan kategori dengan probabilitas posterior tertinggi
            prediksi.append(max(probabilitas_posterior, key=probabilitas_posterior.get))

        return np.array(prediksi)

    def score(self, X, y):
        prediksi = self.predict(X)
        return np.mean(prediksi == np.array(y))
