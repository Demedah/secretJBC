import SkinAnalyzer from "@/components/skin-analyzer"

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-white">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">Jiabao Clinic</h1>
          <h2 className="text-2xl font-semibold text-blue-600 mb-2">Sistem Klasifikasi Jenis Kulit Wajah</h2>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Analisis mendalam jenis kulit wajah Anda menggunakan teknologi ekstraksi fitur GLCM dan histogram warna
            untuk rekomendasi perawatan yang tepat.
          </p>
        </div>

        <SkinAnalyzer />
      </div>
    </div>
  )
}
