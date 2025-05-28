# Capstone Project Arjuna CC25-CF10 

Service API berbasis Flask yang mengintegrasikan machine learning untuk mengklasifikasikan karakter wayang dari gambar input pengguna. API ini memberikan prediksi karakter wayang berdasarkan gambar dengan menggunakan model CNN dan memberikan prediksi TOP_N yang bisa disesuaikan.


## Capstone Team

| Learning Path | ID | Nama | Universitas | Status |
|---------------|----|----|-------------|--------|
| **Machine Learning** | MC253D5Y0335 | Rizki Rahman Maulana | Universitas Mataram | Aktif |
| **Machine Learning** | MC514D5X0632 | Maulany Citra Pandini | Institut Teknologi Statistika dan Bisnis Muhammadiyah | Aktif |
| **Machine Learning** | MC253D5Y1872 | Lalu Olfata Vedora Zurji | Universitas Mataram | Aktif |
| **Front End & Back End** | FC253D5X0564 | Nazila Imkani | Universitas Mataram | Aktif |
| **Front End & Back End** | FC253D5Y2105 | M. Ilham Abdul Shaleh | Universitas Mataram | Aktif |
| **Front End & Back End** | FC808D5X2111 | Rahma | Universitas Muhammadiyah Palopo | Aktif |


## Endpoint

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form-data dengan key `image` berisi file gambar (JPG/PNG)

**Response:**
JSON dengan status dan data prediksi top-N karakter wayang

**Contoh Response:**
```json
{
  "status": "success",
  "data": [
    {"label": "arjuna", "prediction": "85.12"},
    {"label": "bima", "prediction": "10.43"},
    {"label": "gatotkaca", "prediction": "4.45"}
  ]
}
```


