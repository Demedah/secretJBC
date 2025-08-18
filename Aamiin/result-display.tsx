"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { CheckCircle, TrendingUp } from "lucide-react"

interface ResultsDisplayProps {
  result: {
    prediction: string
    confidence: number
    probabilities: { [key: string]: number }
  }
}

const skinTypeDescriptions: { [key: string]: { description: string; recommendations: string[]; color: string } } = {
  Normal: {
    description: "Kulit seimbang dengan produksi minyak yang normal dan pori-pori kecil.",
    recommendations: ["Gunakan pelembab ringan", "Rutin membersihkan wajah 2x sehari", "Gunakan sunscreen SPF 30+"],
    color: "bg-green-100 text-green-800",
  },
  Kering: {
    description: "Kulit kurang produksi minyak alami, terasa kencang dan kasar.",
    recommendations: ["Gunakan pelembab yang kaya", "Hindari sabun keras", "Gunakan serum hyaluronic acid"],
    color: "bg-blue-100 text-blue-800",
  },
  Berminyak: {
    description: "Produksi sebum berlebih, pori-pori besar, rentan berjerawat.",
    recommendations: ["Gunakan cleanser berbahan salicylic acid", "Pelembab oil-free", "Clay mask 1-2x seminggu"],
    color: "bg-yellow-100 text-yellow-800",
  },
  Kombinasi: {
    description: "T-zone berminyak, area pipi normal hingga kering.",
    recommendations: ["Multi-step skincare", "Pelembab berbeda untuk area berbeda", "Toner balancing"],
    color: "bg-purple-100 text-purple-800",
  },
  Sensitif: {
    description: "Mudah iritasi, kemerahan, dan reaktif terhadap produk tertentu.",
    recommendations: ["Produk hypoallergenic", "Patch test sebelum penggunaan", "Hindari fragrance dan alkohol"],
    color: "bg-red-100 text-red-800",
  },
}

export default function ResultsDisplay({ result }: ResultsDisplayProps) {
  const { prediction, confidence, probabilities } = result
  const skinInfo = skinTypeDescriptions[prediction]

  // Sort probabilities untuk display
  const sortedProbs = Object.entries(probabilities).sort(([, a], [, b]) => b - a)

  return (
    <div className="space-y-6">
      {/* Main Result */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CheckCircle className="h-5 w-5 text-green-600" />
            Hasil Analisis
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="text-center space-y-2">
            <Badge className={`text-lg px-4 py-2 ${skinInfo?.color || "bg-gray-100 text-gray-800"}`}>
              {prediction}
            </Badge>
            <p className="text-2xl font-bold text-gray-900">{(confidence * 100).toFixed(1)}% Confidence</p>
          </div>

          <div className="space-y-2">
            <h4 className="font-semibold">Deskripsi:</h4>
            <p className="text-gray-600 text-sm">{skinInfo?.description || "Deskripsi tidak tersedia"}</p>
          </div>
        </CardContent>
      </Card>

      {/* Probability Distribution */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Distribusi Probabilitas
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {sortedProbs.map(([type, prob]) => (
            <div key={type} className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="font-medium">{type}</span>
                <span>{(prob * 100).toFixed(1)}%</span>
              </div>
              <Progress value={prob * 100} className="h-2" />
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle>Rekomendasi Perawatan</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2">
            {skinInfo?.recommendations.map((rec, index) => (
              <li key={index} className="flex items-start gap-2 text-sm">
                <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}
