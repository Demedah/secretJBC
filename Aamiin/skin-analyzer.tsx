"use client"

import type React from "react"

import { useState, useRef, useCallback } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, Camera, BarChart3, Zap } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import FeatureVisualization from "./feature-visualization"
import ResultsDisplay from "./results-display"

interface AnalysisResult {
  prediction: string
  confidence: number
  probabilities: { [key: string]: number }
  features: {
    glcm: {
      contrast: number
      dissimilarity: number
      homogeneity: number
      energy: number
      correlation: number
    }
    colorHistogram: {
      red: number[]
      green: number[]
      blue: number[]
    }
  }
}

export default function SkinAnalyzer() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const handleImageUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      if (file.size > 5 * 1024 * 1024) {
        setError("Ukuran file terlalu besar. Maksimal 5MB.")
        return
      }

      const reader = new FileReader()
      reader.onload = (e) => {
        setSelectedImage(e.target?.result as string)
        setError(null)
        setAnalysisResult(null)
      }
      reader.readAsDataURL(file)
    }
  }, [])

  const extractImageFeatures = useCallback((imageData: ImageData): AnalysisResult["features"] => {
    const { data, width, height } = imageData

    // Konversi ke grayscale untuk GLCM
    const grayData = new Uint8Array(width * height)
    for (let i = 0; i < data.length; i += 4) {
      const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2])
      grayData[i / 4] = gray
    }

    // Simulasi ekstraksi fitur GLCM (simplified)
    const glcmFeatures = {
      contrast: Math.random() * 100 + 50,
      dissimilarity: Math.random() * 50 + 25,
      homogeneity: Math.random() * 0.5 + 0.3,
      energy: Math.random() * 0.3 + 0.1,
      correlation: Math.random() * 0.8 + 0.1,
    }

    // Ekstraksi histogram warna
    const redHist = new Array(256).fill(0)
    const greenHist = new Array(256).fill(0)
    const blueHist = new Array(256).fill(0)

    for (let i = 0; i < data.length; i += 4) {
      redHist[data[i]]++
      greenHist[data[i + 1]]++
      blueHist[data[i + 2]]++
    }

    // Normalisasi histogram
    const totalPixels = width * height
    const normalizeHist = (hist: number[]) => hist.map((val) => val / totalPixels)

    return {
      glcm: glcmFeatures,
      colorHistogram: {
        red: normalizeHist(redHist),
        green: normalizeHist(greenHist),
        blue: normalizeHist(blueHist),
      },
    }
  }, [])

  const classifySkinType = useCallback((features: AnalysisResult["features"]): Omit<AnalysisResult, "features"> => {
    // Simulasi klasifikasi berdasarkan fitur
    const skinTypes = ["Normal", "Kering", "Berminyak", "Kombinasi", "Sensitif"]

    // Logika klasifikasi sederhana berdasarkan fitur GLCM
    let prediction = "Normal"
    const probabilities: { [key: string]: number } = {}

    if (features.glcm.energy > 0.25) {
      prediction = "Kering"
    } else if (features.glcm.contrast > 80) {
      prediction = "Berminyak"
    } else if (features.glcm.homogeneity < 0.5) {
      prediction = "Sensitif"
    } else if (features.glcm.correlation > 0.7) {
      prediction = "Kombinasi"
    }

    // Generate probabilitas acak yang realistis
    skinTypes.forEach((type) => {
      if (type === prediction) {
        probabilities[type] = 0.6 + Math.random() * 0.3
      } else {
        probabilities[type] = Math.random() * 0.3
      }
    })

    // Normalisasi probabilitas
    const total = Object.values(probabilities).reduce((sum, val) => sum + val, 0)
    Object.keys(probabilities).forEach((key) => {
      probabilities[key] = probabilities[key] / total
    })

    return {
      prediction,
      confidence: probabilities[prediction],
      probabilities,
    }
  }, [])

  const analyzeImage = useCallback(async () => {
    if (!selectedImage || !canvasRef.current) return

    setIsAnalyzing(true)
    setError(null)

    try {
      const canvas = canvasRef.current
      const ctx = canvas.getContext("2d")
      if (!ctx) throw new Error("Canvas context tidak tersedia")

      const img = new Image()
      img.crossOrigin = "anonymous"

      await new Promise((resolve, reject) => {
        img.onload = resolve
        img.onerror = reject
        img.src = selectedImage
      })

      // Resize gambar ke 128x128 seperti di kode Python
      canvas.width = 128
      canvas.height = 128
      ctx.drawImage(img, 0, 0, 128, 128)

      const imageData = ctx.getImageData(0, 0, 128, 128)

      // Simulasi delay untuk analisis
      await new Promise((resolve) => setTimeout(resolve, 2000))

      const features = extractImageFeatures(imageData)
      const classification = classifySkinType(features)

      setAnalysisResult({
        ...classification,
        features,
      })
    } catch (err) {
      setError("Gagal menganalisis gambar. Silakan coba lagi.")
      console.error("Analysis error:", err)
    } finally {
      setIsAnalyzing(false)
    }
  }, [selectedImage, extractImageFeatures, classifySkinType])

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Upload Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Camera className="h-5 w-5" />
            Upload Foto Wajah
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div
              className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors cursor-pointer"
              onClick={() => fileInputRef.current?.click()}
            >
              {selectedImage ? (
                <div className="space-y-4">
                  <img
                    src={selectedImage || "/placeholder.svg"}
                    alt="Uploaded"
                    className="max-h-64 mx-auto rounded-lg shadow-md"
                  />
                  <p className="text-sm text-gray-600">Klik untuk mengganti gambar</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <Upload className="h-12 w-12 mx-auto text-gray-400" />
                  <div>
                    <p className="text-lg font-medium">Upload foto wajah Anda</p>
                    <p className="text-sm text-gray-600">Format: JPG, PNG, JPEG (Maksimal 5MB)</p>
                  </div>
                </div>
              )}
            </div>

            <input ref={fileInputRef} type="file" accept="image/*" onChange={handleImageUpload} className="hidden" />

            {selectedImage && (
              <Button onClick={analyzeImage} disabled={isAnalyzing} className="w-full" size="lg">
                {isAnalyzing ? (
                  <>
                    <Zap className="h-4 w-4 mr-2 animate-spin" />
                    Menganalisis...
                  </>
                ) : (
                  <>
                    <BarChart3 className="h-4 w-4 mr-2" />
                    Analisis Jenis Kulit
                  </>
                )}
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Results Section */}
      {analysisResult && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ResultsDisplay result={analysisResult} />
          <FeatureVisualization features={analysisResult.features} />
        </div>
      )}

      {/* Hidden canvas for image processing */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  )
}
