"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  LineChart,
  Line,
} from "recharts"
import { Activity, Palette, Zap } from "lucide-react"

interface FeatureVisualizationProps {
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

export default function FeatureVisualization({ features }: FeatureVisualizationProps) {
  // Prepare GLCM data for radar chart
  const glcmData = [
    { feature: "Contrast", value: features.glcm.contrast, fullMark: 150 },
    { feature: "Dissimilarity", value: features.glcm.dissimilarity, fullMark: 75 },
    { feature: "Homogeneity", value: features.glcm.homogeneity * 100, fullMark: 100 },
    { feature: "Energy", value: features.glcm.energy * 100, fullMark: 40 },
    { feature: "Correlation", value: features.glcm.correlation * 100, fullMark: 100 },
  ]

  // Prepare histogram data (sample every 16th value for visualization)
  const histogramData = Array.from({ length: 16 }, (_, i) => {
    const index = i * 16
    return {
      intensity: index,
      red: features.colorHistogram.red[index] * 1000,
      green: features.colorHistogram.green[index] * 1000,
      blue: features.colorHistogram.blue[index] * 1000,
    }
  })

  return (
    <div className="space-y-6">
      {/* GLCM Features Radar Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Fitur Tekstur GLCM
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={glcmData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="feature" tick={{ fontSize: 12 }} />
                <PolarRadiusAxis angle={90} domain={[0, "dataMax"]} tick={false} />
                <Radar
                  name="GLCM Features"
                  dataKey="value"
                  stroke="#3b82f6"
                  fill="#3b82f6"
                  fillOpacity={0.3}
                  strokeWidth={2}
                />
                <Tooltip formatter={(value: number, name: string) => [`${value.toFixed(2)}`, "Nilai"]} />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
            {Object.entries(features.glcm).map(([key, value]) => (
              <div key={key} className="flex justify-between p-2 bg-gray-50 rounded">
                <span className="capitalize">{key}:</span>
                <span className="font-mono">{value.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Color Histogram */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Palette className="h-5 w-5" />
            Histogram Warna RGB
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={histogramData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="intensity"
                  tick={{ fontSize: 10 }}
                  label={{ value: "Intensitas Warna", position: "insideBottom", offset: -5 }}
                />
                <YAxis tick={{ fontSize: 10 }} label={{ value: "Frekuensi", angle: -90, position: "insideLeft" }} />
                <Tooltip formatter={(value: number, name: string) => [value.toFixed(3), name.toUpperCase()]} />
                <Line type="monotone" dataKey="red" stroke="#ef4444" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="green" stroke="#22c55e" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="blue" stroke="#3b82f6" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Feature Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Ringkasan Analisis
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-1 gap-3 text-sm">
            <div className="p-3 bg-blue-50 rounded-lg">
              <h4 className="font-semibold text-blue-900">Tekstur Kulit</h4>
              <p className="text-blue-700">
                Contrast: {features.glcm.contrast.toFixed(1)} | Homogeneity: {features.glcm.homogeneity.toFixed(3)}
              </p>
            </div>
            <div className="p-3 bg-green-50 rounded-lg">
              <h4 className="font-semibold text-green-900">Distribusi Warna</h4>
              <p className="text-green-700">Analisis histogram RGB menunjukkan karakteristik warna kulit</p>
            </div>
            <div className="p-3 bg-purple-50 rounded-lg">
              <h4 className="font-semibold text-purple-900">Korelasi Fitur</h4>
              <p className="text-purple-700">
                Correlation: {features.glcm.correlation.toFixed(3)} | Energy: {features.glcm.energy.toFixed(3)}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
